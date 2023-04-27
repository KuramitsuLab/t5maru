import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup,
)

import pytorch_lightning as pl
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from deepspeed.ops.adam import DeepSpeedCPUAdam

from .t5data import T5TrainFileModule, T5TestFileModule
from .commons import set_seed, record, log_record, isatty, verbose_print

import json


class T5FineTuner(pl.LightningModule):
    def __init__(
        self,
        model_path,
        solver="adamw",
        output_path=None,
        learning_rate=3e-4,
        adam_epsilon=1e-8,
        weight_decay=0.0,
        warmup_steps=1,
        training_steps=100000,
    ):
        super(T5FineTuner, self).__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.output_path = output_path
        self.solver = solver
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.adam_epsilon = adam_epsilon
        self.warmup_steps = warmup_steps
        self.training_steps = training_steps

    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
    ):
        """順伝搬"""
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def _step(self, batch):
        """ロス計算"""
        labels = batch["target_ids"]
        # All labels set to -100 are ignored (masked),
        # the loss is only computed for labels in [0, ..., config.vocab_size]
        # labels[labels[:, :] == self.tokenizer.pad_token_id] = -100
        labels[labels[:, :] == 0] = -100
        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_attention_mask=batch["target_mask"],
            labels=labels,
        )
        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        """訓練ステップ処理"""
        loss = self._step(batch)
        self.log("loss", loss, sync_dist=True, prog_bar=False)
        return {"loss": loss}

    # def on_training_epoch_end(self, outputs):
    #     """バリデーション完了処理"""
    #     # print("アウトプットの確認", outputs)
    #     avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
    #     ppl = torch.exp(avg_loss)
    #     self.log("avg_loss", avg_loss, prog_bar=True)
    #     self.log("train_ppl", ppl, prog_bar=True)
    #     if self.output_path:
    #         self.model.save_pretrained(self.output_path)
    #     print(
    #         f'train epoch={self.current_epoch+1} loss={avg_loss:.5f} PPL={ppl:.5f}')

    def validation_step(self, batch, batch_idx):
        """バリデーションステップ処理"""
        loss = self._step(batch)
        self.log("val_loss", loss, sync_dist=True, prog_bar=False)
        return {"val_loss": loss}

    # def on_validation_epoch_end(self, outputs):
    #     """バリデーション完了処理"""
    #     avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
    #     ppl = torch.exp(avg_loss)
    #     self.log("avg_loss", avg_loss, prog_bar=True)
    #     self.log("val_ppl", ppl, prog_bar=False)
    #     print(
    #         f'val epoch={self.current_epoch+1} loss={avg_loss:.5f} PPL={ppl:.5f}')

    def configure_optimizers(self):
        """オプティマイザーとスケジューラーを作成する"""
        if self.solver == "deepspeed":
            return self.configure_DeepSpeedAdam()
        return self.configure_AdamW()

    def grouped_parameters(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        return optimizer_grouped_parameters

    def configure_DeepSpeedAdam(self):
        optimizer = DeepSpeedCPUAdam(
            self.grouped_parameters(), lr=self.learning_rate, eps=self.adam_epsilon
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.training_steps,
        )
        record(
            solver=self.solver,
            lr=self.learning_rate,
            warmup_steps=self.warmup_steps,
            training_steps=self.training_steps,
            adam_epsilon=self.adam_epsilon,
            weight_decay=self.weight_decay,
        )
        return [optimizer], [
            {"scheduler": scheduler, "interval": "step", "frequency": 1}
        ]

    def configure_AdamW(self):
        # self.t_total = (
        #     (len(self.train_dataset) //
        #         (self.hparams.batch_size * max(1, self.hparams.n_gpus)))
        #     // self.hparams.gradient_accumulation_steps
        #     * float(self.hparams.max_epochs)
        # )
        optimizer = AdamW(
            self.grouped_parameters(), lr=self.learning_rate, eps=self.adam_epsilon
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=max(self.training_steps,100000),
        )
        record(
            solver=self.solver,
            lr=self.learning_rate,
            warmup_steps=self.warmup_steps,
            training_steps=max(self.training_steps, 100000),
            adam_epsilon=self.adam_epsilon,
            weight_decay=self.weight_decay,
        )
        return [optimizer], [
            {"scheduler": scheduler, "interval": "step", "frequency": 1}
        ]


def get_device(device=None, gpu="cuda"):
    if device is not None:
        return device
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return gpu
    return "cpu"


def get_accelerator(accelerator=None):
    return get_device(accelerator, gpu="gpu")


class T5Model:
    def __init__(
        self,
        model_path="kkuramitsu/mt5-mini9L",
        tokenizer_path=None,
        max_length=128,
        target_max_length=None,
        accelerator=None,
        precision=32,
        strategy="auto",
        batch_size=32,
        num_of_workers=4,
    ):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path, use_fast=False
        )
        assert self.tokenizer.pad_token_id == 0
        self.accelerator = get_accelerator(accelerator)
        self.precision = precision
        self.strategy = strategy
        self.devices = "auto"
        self.max_length = max_length
        self.target_max_length = target_max_length or max_length
        self.batch_size = batch_size
        self.num_of_workers = num_of_workers  # non multitasking
        self.debug = False

    def preprocess(self, jsonl: dict):
        pass

    def transform(self, jsonl: dict):
        self.preprocess(jsonl)
        inputs = self.tokenizer.batch_encode_plus(
            [jsonl["in"]],
            max_length=self.max_length,
            truncation=True,
            pad_to_max_length=True,
            padding="max_length",
            return_tensors="pt",
        )
        source_ids = inputs["input_ids"].squeeze()
        source_mask = inputs["attention_mask"].squeeze()
        if "out" not in jsonl:
            return {
                "source_ids": source_ids.to(dtype=torch.long),
                "source_mask": source_mask.to(dtype=torch.long),
            }
        targets = self.tokenizer.batch_encode_plus(
            [jsonl["out"]],
            max_length=self.target_max_length,
            truncation=True,
            pad_to_max_length=True,
            padding="max_length",
            return_tensors="pt",
        )
        target_ids = targets["input_ids"].squeeze()
        target_mask = targets["attention_mask"].squeeze()
        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_mask": target_mask.to(dtype=torch.long),
        }

    def transform_input_only(self, jsonl: dict):
        self.preprocess(jsonl)
        inputs = self.tokenizer.batch_encode_plus(
            [jsonl["in"]],
            max_length=self.max_length,
            truncation=True,
            pad_to_max_length=True,
            padding="max_length",
            return_tensors="pt",
        )
        source_ids = inputs["input_ids"].squeeze()
        source_mask = inputs["attention_mask"].squeeze()
        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
        }

    def scale_batch_size_dm(self, datamodule, mode="power"):
        model = T5FineTuner(self.model_path)
        trainer = pl.Trainer(
            max_epochs=10,
            log_every_n_steps=0,
            #            enable_progress_bar=isatty(),
            accelerator=self.accelerator,
            devices=self.devices,
            precision=self.precision,
            strategy=self.strategy,
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            logger=False,
            # replace_sampler_ddp=False,
        )
        tuner = Tuner(trainer)
        try:
            datamodule.batch_size = 2
            tuner.scale_batch_size(
                model, datamodule=datamodule, steps_per_trial=2, init_val=2, mode=mode
            )
        except Exception as e:
            print(e)
            datamodule.batch_size //= 2
        self.batch_size = datamodule.batch_size
        return datamodule.batch_size

    def scale_batch_size(self, data_file, batch_size=None, mode="power"):
        with T5TrainFileModule(
            data_file,
            transform=self.transform,
            batch_size=batch_size or max(self.batch_size, 1),
        ) as dm:
            return self.scale_batch_size_dm(dm, mode=mode)

    def train(
        self,
        data_files,
        downsizing=None,
        max_epochs=10,
        max_time=None,
        batch_per_step=None,
        solver="adamw",
        warmup_steps=1,
        learning_rate=3e-4,
        adam_epsilon=1e-8,
        weight_decay=0.0,
        max_grad_norm=1.0,
        gradient_accumulation_steps=None,
        early_stopping=False,
        output_path="model",
        random_seed=42,
    ):
        set_seed(random_seed)  # 乱数を初期化
        with T5TrainFileModule(
            data_files,
            transform=self.transform,
            prefix=output_path,
            batch_size=self.batch_size,
            shuffle=True,
            downsizing=None,
        ) as dm:
            if self.batch_size < 1:
                self.batch_size = self.scale_batch_size_dm(dm)
            if batch_per_step is None:
                if isinstance(gradient_accumulation_steps, int):
                    batch_per_step = self.batch_size * gradient_accumulation_steps
                else:
                    gradient_accumulation_steps = 1
                    batch_per_step = max(self.batch_size, 1)
            else:
                gradient_accumulation_steps = max(batch_per_step // self.batch_size, 1)
            train_steps = dm.train_size * max_epochs // batch_per_step
            record(
                batch_size=self.batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                train_steps=train_steps,
            )
            log_every_n_steps = 50
            if log_every_n_steps > train_steps:  # 警告を消すため
                log_every_n_steps = 0
            net = T5FineTuner(
                self.model_path,
                solver=solver,
                learning_rate=learning_rate,
                adam_epsilon=adam_epsilon,
                weight_decay=weight_decay,
                warmup_steps=warmup_steps,
                output_path=output_path,
                training_steps=train_steps,
            )
            # EarlyStopping
            callbacks = []
            if early_stopping:
                early_stop_callback = EarlyStopping(
                    monitor="val_loss", patience=3, verbose=True, mode="min"
                )
                callbacks.append(early_stop_callback)
            # if checkpoint_path:
            #     # https://blog.shikoan.com/pytorch-lightning-max-time/
            #     checkpoint_callback = pl.callbacks.ModelCheckpoint(
            #         monitor="val_loss",
            #         dirpath=checkpoint_path,
            #         filename="epoch{epoch:02d}-{val_loss:.5f}",
            #         save_top_k=3,
            #         mode="max"
            #     )
            #     callbacks.append(checkpoint_callback)
            # resume_ckpt = find_latest_checkpoints(checkpoint_path)
            trainer = pl.Trainer(
                # enable_progress_bar=isatty(),
                log_every_n_steps=log_every_n_steps,
                fast_dev_run=self.debug,
                accelerator=self.accelerator,
                devices=self.devices,
                precision=self.precision,
#                strategy=self.strategy,
                max_epochs=max_epochs,
                max_time=max_time,
                gradient_clip_val=max_grad_norm,
                # k バッチ毎に勾配を蓄積する batch_size * k になる
                accumulate_grad_batches=gradient_accumulation_steps,
                callbacks=callbacks,
                # https://towardsdatascience.com/pytorch-lightning-vs-deepspeed-vs-fsdp-vs-ffcv-vs-e0d6b2a95719
                enable_progress_bar=False,
                # enable_model_summary=False,
                enable_checkpointing=False,
                # logger=False,
                # replace_sampler_ddp=False,
            )
            record(
                accelerator=self.accelerator,
                devices=self.devices,
                precision=self.precision,
                strategy=self.strategy,
                gradient_clip_val=max_grad_norm,
            )
            if max_epochs > 0:
                trainer.fit(net, dm)
                record(epoch=trainer.current_epoch, step=trainer.global_step)
            # 最終エポックのモデルを保存 output_path に保存します
            self.tokenizer.save_pretrained(output_path)
            net.model.save_pretrained(output_path)
            self.model_path = output_path
            record(saved=self.model_path)

    def predict(self, test_file):
        device = torch.device(get_device())
        record(record=str(device))
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
        model.to(device)
        with T5TestFileModule(test_file, transform=self.transform_input_only) as dm:
            dm.setup("test")
            results = []
            for batch in dm.test_dataloader():
                outputs = model.generate(
                    input_ids=batch["source_ids"].to(device),  # .cuda()
                    attention_mask=batch["source_mask"].to(device),  # .cuda()
                    max_length=self.target_max_length,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                preds = [
                    self.tokenizer.decode(
                        ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    for ids in outputs.sequences
                ]
                results.extend(preds)
            return results

    def test(self, test_file):
        results = self.predict(test_file)
        file = test_file
        if "/" in file:
            _, _, file = file.rpartition("/")
        file = file.replace(".gz", "")
        if not file.endswith(".jsonl"):
            file = file.rpartition(".")[0] + ".jsonl"
        if "_test.jsonl" in file:
            file = file.replace("_test.", "_tested.")
        else:
            file = file.replace(".jsonl", "_tested.jsonl")
        if os.path.isdir(self.model_path):
            output_file = f"{self.model_path}/{file}"
        else:
            output_file = file
        with open(output_file, "w", encoding="utf-8", errors="ignore") as w:
            with T5TestFileModule(test_file, batch_size=1) as dm:
                dm.setup("test")
                for i, batch in enumerate(dm.test_dataloader()):
                    # print(results[i], batch)
                    unlist_dict(batch)
                    batch["pred"] = results[i]
                    print(json.dumps(batch, ensure_ascii=False), file=w)
            verbose_print(f"Tested {len(results)} items. See {output_file}")
        return results

def unlist_dict(d):
    for key, value in d.items():
        if isinstance(value, list):
            value = value[0]
        d[key] = value

def setup():
    import argparse

    parser = argparse.ArgumentParser(description="t5tune script")
    parser.add_argument("files", type=str, nargs="+", help="jsonl files")
    parser.add_argument("--downsizing", default=None)

    parser.add_argument("--model_path", default="kkuramitsu/mt5-mini9L")
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--target_max_length", type=int, default=None)
    #    parser.add_argument('--float32_matmul_precision', type=str, default=None)
    parser.add_argument("--accelerator", type=str, default=None)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", default=None)
    parser.add_argument("--strategy", default="auto")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=min(4, os.cpu_count()))
    ##
    parser.add_argument("--output_path", default="local")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--batch_per_step", default=None)
    parser.add_argument("--solver", type=str, default="adamw")
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--max_time", type=str, default=None)
    parser.add_argument("--early_stopping", action="store_true", default=True)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--warmup_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)

    hparams = parser.parse_args()  # hparams になる

    # デフォルトがNoneのときは
    if hparams.tokenizer_path is None:
        hparams.tokenizer_path = hparams.model_path
    if hparams.precision is None:
        hparams.precision = 32
    if hparams.target_max_length is None:
        hparams.target_max_length = hparams.max_length
    if hparams.batch_per_step is None and hparams.gradient_accumulation_steps is None:
        hparams.gradient_accumulation_steps = 1
    torch.set_float32_matmul_precision("medium")
    return hparams


def main_train(hparams=None, files=None):
    if not hparams:
        hparams = setup()
    if not files:
        files = hparams.files
    model = T5Model(
        model_path=hparams.model_path,
        batch_size=hparams.batch_size,
        max_length=hparams.max_length,
        target_max_length=hparams.target_max_length,
        precision=hparams.precision,
        strategy=hparams.strategy,
        num_of_workers=hparams.num_workers,
    )
    record(
        model=hparams.model_path,
        max_length=hparams.max_length,
        target_max_length=hparams.target_max_length,
    )
    try:
        model.train(
            files,
            downsizing=hparams.downsizing,
            random_seed=hparams.random_seed,
            max_epochs=hparams.max_epochs,
            max_time=hparams.max_time,
            early_stopping=hparams.early_stopping,
            batch_per_step=hparams.batch_per_step,
            solver=hparams.solver,
            learning_rate=hparams.learning_rate,
            weight_decay=hparams.weight_decay,
            adam_epsilon=hparams.adam_epsilon,
            warmup_steps=hparams.warmup_steps,
            max_grad_norm=hparams.max_grad_norm,
            gradient_accumulation_steps=hparams.gradient_accumulation_steps,
            output_path=hparams.output_path,
        )
        result = "trained"
    except SyntaxError as e:
        result = "failed_train"
        record(error=repr(e))
    log_record(result, output_file=hparams.output_path)


def main_test(hparams=None, files=None):
    if not hparams:
        hparams = setup()
    if not files:
        files = hparams.files
    model = T5Model(
        model_path=hparams.model_path,
        max_length=hparams.max_length,
        target_max_length=hparams.target_max_length,
        batch_size=hparams.batch_size,
        num_of_workers=hparams.num_workers,
    )
    for test_file in files:
        model.test(test_file)


def main():
    hparams = setup()
    train_files = [file for file in hparams.files if "_test." not in file]
    test_files = [file for file in hparams.files if "_test." in file]
    if len(train_files) > 0:
        main_train(hparams, train_files)
    if len(test_files) > 0:
        main_test(hparams, test_files)


if __name__ == "__main__":
    main()
