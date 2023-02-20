# import dill as pickle
# import glob
import sys
import random
import json

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup
)

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from datasets import load_dataset
# https://www.kaggle.com/code/noriyukipy/text-classification-dataloader-from-datasets
from transformers import default_data_collator


def set_seed(seed):  # 乱数シードの設定
    random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def isatty():
    return sys.stdout.isatty()


DUP = set()


def debug_print(*args, **kwargs):
    if len(DUP) < 512:
        sep = kwargs.get('sep', ' ')
        text = ' '.join(str(a) for a in args)
        if text in DUP:
            return
        print('😱', text)
        DUP.add(text)


USE_GPU = torch.cuda.is_available()


def find_latest_checkpoints(checkpoint_dir):
    ckpts = sorted(glob.glob(checkpoint_dir+"/*.ckpt"))
    if len(ckpts) == 0:
        return None
    else:
        return ckpts[-1]


class T5FineTuner(pl.LightningModule):
    def __init__(self, model_path, solver='adamw',
                 learning_rate=3e-4, adam_epsilon=1e-8, weight_decay=0.0,
                 training_steps=100000):
        super(T5FineTuner, self).__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        debug_print('model', self.model.config)
        self.solver = solver
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.adam_epsilon = adam_epsilon
        self.warmup_steps = 1
        self.training_steps = training_steps

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, labels=None):
        """順伝搬"""
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
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
            decoder_attention_mask=batch['target_mask'],
            labels=labels
        )
        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        """訓練ステップ処理"""
        loss = self._step(batch)
        self.log("loss", loss)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        """バリデーション完了処理"""
        # print("アウトプットの確認", outputs)
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        ppl = torch.exp(avg_loss)
        self.log("avg_loss", avg_loss, prog_bar=True)
        self.log("train_ppl", ppl, prog_bar=True)
        if not isatty():
            debug_print(
                f'Epoch {self.current_epoch+1} train_loss {avg_loss:.5f} PPL {ppl:.5f}')

    def validation_step(self, batch, batch_idx):
        """バリデーションステップ処理"""
        loss = self._step(batch)
        self.log("val_loss", loss, prog_bar=False)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        """バリデーション完了処理"""
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        ppl = torch.exp(avg_loss)
        self.log("avg_loss", avg_loss, prog_bar=True)
        self.log("val_ppl", ppl, prog_bar=False)
        if not isatty():
            debug_print(
                f'Epoch {self.current_epoch+1} val_loss {avg_loss:.5f} PPL {ppl:.5f}')

    def configure_optimizers(self):
        """オプティマイザーとスケジューラーを作成する"""
        # self.t_total = (
        #     (len(self.train_dataset) //
        #         (self.hparams.batch_size * max(1, self.hparams.n_gpus)))
        #     // self.hparams.gradient_accumulation_steps
        #     * float(self.hparams.max_epochs)
        # )
        if self.solver == 'adafactor':
            return self.configure_ConstantAdafactor()
        return self.configure_AdamW()

    def grouped_parameters(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters()
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return optimizer_grouped_parameters

    def configure_AdamW(self):
        optimizer = AdamW(self.grouped_parameters(),
                          lr=self.learning_rate,
                          eps=self.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.training_steps
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def configure_Adafactor(self):
        from transformers.optimization import Adafactor, AdafactorSchedule
        optimizer = Adafactor(
            self.grouped_parameters(),
            scale_parameter=True,
            relative_step=True,
            warmup_init=True,
            lr=None)
        scheduler = AdafactorSchedule(optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def configure_ConstantAdafactor(self):
        from transformers.optimization import Adafactor
        optimizer = Adafactor(
            self.model.parameters(),
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            lr=1e-3)
        debug_print('[optimizer]', optimizer)
        return optimizer


# Note - you must have torchvision installed for this example

def transform_nop(x):
    return x


class T5DataModule(pl.LightningDataModule):
    def __init__(self, data_files: str, valid_files=None,
                 transform=transform_nop,
                 valid_split='valid', test_split='test',
                 batch_size=32, num_of_workers=0, streaming=False):
        super().__init__()
        self.file_type = 'json'
        self.data_files = data_files
        self.valid_files = valid_files
        self.valid_split = valid_split
        self.test_split = test_split
        self.transform = transform
        self.streaming = streaming
        self.batch_size = batch_size
        self.num_of_workers = num_of_workers

    def load(self, data_files, split='train'):
        ds = load_dataset(self.file_type,
                          data_files=data_files,
                          split=split, streaming=self.streaming)
        return ds.map(self.transform).with_format('torch')

    # def map(self, ds):
    #     return ds.with_format('torch')
    #     # return ds.map(transform_nop).with_format('torch')

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.ds_train = self.load(self.data_files, split='train')
            if self.valid_files:
                self.ds_valid = self.load(self.valid_files, split='train')
            else:
                self.ds_valid = self.load(
                    self.data_files, split=self.valid_split)
        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.ds_test = self.load(self.data_files, split=self.test_split)

    def train_dataloader(self):
        return DataLoader(self.ds_train,
                          collate_fn=default_data_collator,
                          num_workers=self.num_of_workers,  # drop_last=True, shuffle=True,
                          batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.ds_valid,
                          num_workers=self.num_of_workers,
                          batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.ds_test,
                          num_workers=self.num_of_workers,
                          batch_size=self.batch_size)


class T5Model(object):
    def __init__(self, model_path='kkuramitsu/mt5np_small8k',
                 tokenizer_path=None,
                 max_length=128, target_max_length=None,
                 batch_size=256, step_batch_size=0,
                 num_of_workers=0,
                 debug=False):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.tokenizer = None
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path, use_fast=False)
        assert self.tokenizer.pad_token_id == 0
        self.max_length = max_length
        self.target_max_length = target_max_length or max_length
        self.debug = debug
        self.num_of_workers = num_of_workers  # non multitasking
        self.batch_size = batch_size  # default
        self.step_batch_size = step_batch_size

    def preprocess(self, jsonl: dict):
        pass

    def transform(self, jsonl: dict):
        self.preprocess(jsonl)
        inputs = self.tokenizer.batch_encode_plus(
            [jsonl['in']],
            max_length=self.max_length,
            truncation=True,
            pad_to_max_length=True,
            padding="max_length", return_tensors="pt")
        source_ids = inputs["input_ids"].squeeze()
        source_mask = inputs["attention_mask"].squeeze()
        if 'out' not in jsonl:
            return {
                "source_ids": source_ids.to(dtype=torch.long),
                "source_mask": source_mask.to(dtype=torch.long),
            }
        targets = self.tokenizer.batch_encode_plus(
            [jsonl['out']],
            max_length=self.target_max_length,
            truncation=True,
            pad_to_max_length=True,
            padding="max_length", return_tensors="pt")
        target_ids = targets["input_ids"].squeeze()
        target_mask = targets["attention_mask"].squeeze()
        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_mask": target_mask.to(dtype=torch.long),
        }

    def fit(self, data_files, val_files=None,
            max_epochs=10, max_hours=None, n_gpus=None, training_steps=10000,
            solver='adamw',
            learning_rate=3e-4, adam_epsilon=1e-8, weight_decay=0.0,
            early_stopping=False, checkpoint_path=None,
            output_path=None, random_seed=42, streaming=False):
        set_seed(random_seed)  # 乱数を初期化
        data = T5DataModule(data_files, val_files,
                            transform=self.transform,
                            batch_size=self.step_batch_size,
                            num_of_workers=self.num_of_workers,
                            streaming=streaming)
        model = T5FineTuner(
            self.model_path,
            solver=solver,
            learning_rate=learning_rate,
            adam_epsilon=adam_epsilon,
            weight_decay=weight_decay,
            training_steps=training_steps)
        if self.step_batch_size < 1:
            tuner = pl.Trainer(
                enable_progress_bar=isatty(),
                gpus=(1 if USE_GPU else 0) if n_gpus is None else n_gpus,
                auto_scale_batch_size="binsearch",
            )
            tuner.tune(model, data)
            debug_print('GPU: batch_size', data.batch_size)
            self.step_batch_size = data.batch_size
        accumulate_grad_batches = max(
            self.batch_size // self.step_batch_size, 1)
        # EarlyStopping
        callbacks = []
        if early_stopping:
            early_stop_callback = EarlyStopping(
                monitor="val_loss", patience=3,
                verbose=True,
                mode="min"
            )
            callbacks.append(early_stop_callback)
        if checkpoint_path:
            # https://blog.shikoan.com/pytorch-lightning-max-time/
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                monitor="val_loss",
                dirpath=checkpoint_path,
                filename="epoch{epoch:02d}-{val_loss:.5f}",
                save_top_k=3,
                mode="max"
            )
            callbacks.append(checkpoint_callback)
            # resume_ckpt = find_latest_checkpoints(checkpoint_path)
        max_time = None
        if max_hours is not None:
            hours = int(max_hours)
            days = hours // 24
            hours = hours % 24
            mins = int(60 * (max_hours - hours))
            max_time = {'days': days, 'hours': hours, 'minutes': mins}
            debug_print('[max_time]', max_time)
        trainer = pl.Trainer(
            enable_progress_bar=isatty(),
            fast_dev_run=self.debug,
            gpus=(1 if USE_GPU else 0) if n_gpus is None else n_gpus,
            max_epochs=max_epochs,
            max_time=max_time,
            # gradient_clip_val=hparams.max_grad_norm,
            # k バッチ毎に勾配を蓄積する batch_size * k になる
            accumulate_grad_batches=accumulate_grad_batches,
            callbacks=callbacks,
            # precision=hparams.precision,
            # #        amp_level='O2' if hparams.precision == 16 else 'O0'
        )
        if max_epochs > 0:
            trainer.fit(model, data)
        # 最終エポックのモデルを保存 output_path に保存します
        if output_path is None:
            output_path = 'model'
        self.tokenizer.save_pretrained(output_path)
        model.model.save_pretrained(output_path)
        self.model_path = output_path

    def predict(self, test_files, split='train', output_file='result.jsonl', streaming=False):
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
        data = T5DataModule(test_files,
                            transform=self.transform, test_split=split,
                            batch_size=self.step_batch_size,
                            num_of_workers=self.num_of_workers,
                            streaming=streaming)
        data.setup('test')
        results = {}
        dataloader = data.test_dataloader()
        for batch in dataloader:
            outputs = model.generate(
                input_ids=batch['source_ids'],
                attention_mask=batch['source_mask'],
                max_length=self.target_max_length,
                return_dict_in_generate=True,
                output_scores=True)
            preds = [self.tokenizer.decode(ids, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False) for ids in outputs.sequences]
            for key in batch.keys():
                if isinstance(batch[key], list) and isinstance(batch[key][0], (int, str)):
                    if key in results:
                        results[key].extend(batch[key])
                    else:
                        results[key] = batch[key]
            if 'pred' in results:
                results['pred'].extend(preds)
            else:
                results['pred'] = preds
        if output_file:
            with open(output_file, 'w') as w:
                keys = list(results.keys())
                for idx in range(len(results['pred'])):
                    d = {key: results[key][idx] for key in keys}
                    print(json.dumps(d, ensure_ascii=False), file=w)
        return results


def setup_hyperparameters():
    import argparse
    # ハイパーパラメータの読み込み  何も書かなければ、デフォルト値 default
    # python3 finetune.py --batch_size 64
    parser = argparse.ArgumentParser(description='train script')
    parser.add_argument('files', type=str, nargs='+', help='jsonl files')
    parser.add_argument('--model_path', default='google/mt5-small')
    parser.add_argument('--tokenizer_path', default=None)
    parser.add_argument('--checkpoint_path', default=None)
    parser.add_argument('--output_path', default='model')
    parser.add_argument('--tested_file', default='tested.jsonl')
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--source_max_length', type=int, default=None)
    parser.add_argument('--target_max_length', type=int, default=None)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--max_time', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)  # 自動
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--warmup_steps', type=int, default=1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--precision', type=int, default=32)
    parser.add_argument('--n_gpus', type=int, default=1 if USE_GPU else 0)
    # https://note.nkmk.me/python-argparse-bool/
    parser.add_argument('--auto_batch_size',
                        action='store_true', default=False)
    parser.add_argument('--early_stopping', action='store_true', default=False)
    parser.add_argument('--progress_bar', action='store_true', default=False)
    parser.add_argument('--fast_dev_run', action='store_true', default=False)

    hparams = parser.parse_args()  # hparams になる
    # デフォルトがNoneのときは
    if hparams.tokenizer_path is None:
        hparams.tokenizer_path = hparams.model_path
    if hparams.source_max_length is None:
        hparams.source_max_length = hparams.max_length
    if hparams.target_max_length is None:
        hparams.target_max_length = hparams.max_length
    hparams.test = sum(1 for file in hparams.files if '_test.' in file) > 0

    # 訓練パラメータの設定
    # https://torch.classcat.com/2021/02/22/pytorch-lightning-1-1-notebooks-05-trainer-flags-overview-2/
    train_params = dict(
        enable_progress_bar=hparams.progress_bar,
        fast_dev_run=hparams.fast_dev_run,
        gpus=hparams.n_gpus,
        max_epochs=hparams.max_epochs,
        max_time=hparams.max_time,  # "00:00:15:00"
        gradient_clip_val=hparams.max_grad_norm,
        # k バッチ毎に勾配を蓄積する batch_size * k になる
        accumulate_grad_batches=hparams.gradient_accumulation_steps,
        # batch_size の自動調整,  hparams.batch_size が上書きされる
        auto_scale_batch_size="binsearch" if hparams.auto_batch_size else None,
        precision=hparams.precision,
        #        amp_level='O2' if hparams.precision == 16 else 'O0'
    )
    return hparams, train_params


def main():
    model = T5Model('kkuramitsu/mt5np_small8k',
                    batch_size=256, num_of_workers=4)
    model.fit('music/music_train.jsonl',
              'music/music_valid.jsonl', solver='adafactor')
    print(model.predict('music/music_test.jsonl'))


if __name__ == '__main__':
    main()
