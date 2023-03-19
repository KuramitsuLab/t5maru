# import dill as pickle
# import glob
import time
from transformers import default_data_collator
import os
import urllib.request
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
from transformers.optimization import Adafactor, AdafactorSchedule

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import datasets
from datasets import load_dataset, load_from_disk, disable_caching
from datasets.utils import disable_progress_bar
datasets.utils.logging.set_verbosity_warning()

try:
    from .metrics import calc_score, get_filename
except:
    def get_filename(s):
        return s

    def calc_score(*args, **kwargs):
        pass

# https://www.kaggle.com/code/noriyukipy/text-classification-dataloader-from-datasets


def set_seed(seed):  # 乱数シードの設定
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def isatty():
    return sys.stdout.isatty()


DUP = set()


def debug_print(*args, **kwargs):
    if len(DUP) < 512:
        sep = kwargs.get('sep', ' ')
        text = sep.join(str(a) for a in args)
        if text in DUP:
            return
        print('😱', text)
        DUP.add(text)


LOGFILE = None


def set_logfile(output_path):
    global LOGFILE
    os.makedirs(output_path, exist_ok=True)
    LOGFILE = f'{output_path}/train_log.txt'


def print_log(*args, **kwargs):
    if LOGFILE:
        sep = kwargs.get('sep', ' ')
        text = sep.join(str(a) for a in args)
        try:
            with open(LOGFILE, 'a') as w:
                print(text, file=w)
        except:
            pass
        finally:
            print('💭', text)


USE_GPU = torch.cuda.is_available()


def transform_nop(x):
    return x


#


def url_is_alive(url: str):
    request = urllib.request.Request(url)
    request.get_method = lambda: 'HEAD'
    try:
        urllib.request.urlopen(request)
        return True
    except urllib.request.HTTPError:
        return False


def check_saved_path(filename):
    if '/' in filename:
        _, _, filename = filename.rpartition('/')
    saved = filename.replace('.gz', '').replace('.jsonl', '_saved')
    return saved, os.path.isdir(saved)


def check_saved_path(filename):
    if '/' in filename:
        _, _, filename = filename.rpartition('/')
    saved = filename.replace('.gz', '').replace('.jsonl', '_saved')
    return saved, os.path.isdir(saved)


class T5DataModule(pl.LightningDataModule):
    def __init__(self, data_source: str,
                 transform=transform_nop,
                 split='train', valid_split=None,  # 'validation',
                 batch_size=32, num_of_workers=0, streaming=False):
        super().__init__()
        self.data_source = data_source
        self.split = split
        self.valid_split = valid_split
        self.transform = transform
        self.streaming = streaming
        self.batch_size = batch_size
        self.num_of_workers = num_of_workers

    def file_type(self, data_source):
        if isinstance(data_source, (list, tuple)):
            return self.file_type(data_source[0])
        if isinstance(data_source, dict):
            return self.file_type(data_source['train'])
        assert isinstance(data_source, str)
        if '.json' in data_source:
            return 'json', {}
        if '.csv' in data_source:
            return 'csv', {}
        if '.tsv' in data_source:
            return 'csv', dict(delimiter='\t')
        return 'text', {}

    def valid_files(self, data_source):
        if isinstance(data_source, (list, tuple)):
            files = []
            for file in data_source:
                files.extend(self.valid_files(file))
            return files
        assert isinstance(data_source, str)
        if '_train.' in data_source:
            file = data_source.replace('_train.', '_valid.')
            if self.exists(file):
                return [file]
        return []

    def load(self, data_source, split='train', transform=transform_nop):
        # if isinstance(data_source, str) and split == 'train':
        #     # 保存済みの前処理を活用する
        #     saved_path, has_saved = check_saved_path(data_source)
        #     if has_saved:
        #         return load_from_disk(saved_path)
        #     ds = ds.map(
        #         transform, num_proc=4).with_format('torch')
        #     ds.save_to_disk(saved_path)
        #     return ds
        if isinstance(data_source, (str, list, tuple, dict)):
            file_type, kwargs = self.file_type(data_source)
            ds = load_dataset(file_type,
                              data_files=data_source, split=split,
                              streaming=self.streaming, **kwargs)
            ds = ds.map(
                transform, num_proc=self.num_of_workers).with_format('torch')
            return ds
        return data_source.map(transform, num_proc=self.num_of_workers).with_format('torch')

    def exists(self, file_path):
        if file_path.startswith('https://') or file_path.startswith('http://'):
            return url_is_alive(file_path)
        return os.path.isfile(file_path)

    def dropper(self, data: dict):
        if 'in' in data:
            data['in'] = ''.join(c for c in data['in']
                                 if random.random() > 0.1 or ord(c) < 128)
        return self.transform(data)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.ds_train = self.load(self.data_source,
                                      split=self.split, transform=self.transform)
            if self.valid_split:
                self.ds_valid = self.load(
                    self.data_source, split=self.valid_split, transform=self.transform)
                return
            valid_files = self.valid_files(self.data_source)
            if len(valid_files) > 0:
                self.ds_valid = self.load(
                    valid_files, split='train', transform=self.transform)
            else:
                try:
                    self.ds_valid = self.load(
                        self.data_source, split='train[:256]', transform=self.dropper)
                except ValueError as e:
                    self.ds_valid = self.load(
                        self.data_source, split='train[:30%]', transform=self.dropper)

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.ds_test = self.load(
                self.data_source, split=self.split, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.ds_train,
                          collate_fn=default_data_collator,
                          num_workers=self.num_of_workers,
                          drop_last=True, shuffle=True,
                          batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.ds_valid,
                          collate_fn=default_data_collator,
                          num_workers=self.num_of_workers,
                          batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.ds_test,
                          # collate_fn=default_data_collator,
                          num_workers=self.num_of_workers,
                          batch_size=self.batch_size)


# def find_latest_checkpoints(checkpoint_dir):
#     ckpts = sorted(glob.glob(checkpoint_dir+"/*.ckpt"))
#     if len(ckpts) == 0:
#         return None
#     else:
#         return ckpts[-1]


class T5FineTuner(pl.LightningModule):
    def __init__(self, model_path, solver='adamw', output_path=None,
                 learning_rate=3e-4, adam_epsilon=1e-8, weight_decay=0.0,
                 training_steps=100000):
        super(T5FineTuner, self).__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        debug_print('model', self.model.config)
        self.output_path = output_path
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
        if self.output_path:
            self.model.save_pretrained(self.output_path)
        print_log(
            f'train epoch={self.current_epoch+1} loss={avg_loss:.5f} PPL={ppl:.5f}')
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
        print_log(
            f'val epoch={self.current_epoch+1} loss={avg_loss:.5f} PPL={ppl:.5f}')
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
        optimizer = Adafactor(
            self.grouped_parameters(),
            scale_parameter=True,
            relative_step=True,
            warmup_init=True,
            lr=None)
        scheduler = AdafactorSchedule(optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def configure_ConstantAdafactor(self):
        optimizer = Adafactor(
            self.model.parameters(),
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            lr=1e-3)
        debug_print('[optimizer]', optimizer)
        return optimizer


# Note - you must have torchvision installed for this example


class T5ModelTrainer(object):
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

    def fit(self, data_files,
            split='train', valid_split=None,
            max_epochs=10, max_hours=None,
            accelerator=None, devices=1, precision=32,
            solver='adamw', training_steps=10000,
            learning_rate=3e-4, adam_epsilon=1e-8, weight_decay=0.0,
            early_stopping=False, checkpoint_path=None,
            output_path='model', random_seed=42, streaming=False):
        set_seed(random_seed)  # 乱数を初期化
        if accelerator is None:
            accelerator = 'gpu' if USE_GPU else 'cpu'
        data = T5DataModule(data_files, split=split, valid_split=valid_split,
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
            output_path=output_path,
            training_steps=training_steps)
        if self.step_batch_size < 1:
            tuner = pl.Trainer(
                max_epochs=3,
                enable_progress_bar=isatty(),
                accelerator=accelerator, devices=devices,
                precision=precision,
                auto_scale_batch_size="power",
            )
            data.batch_size = self.batch_size // 4
            tuner.tune(model, data)
            print_log('[auto_batch_size]', data.batch_size)
            self.step_batch_size = data.batch_size
        accumulate_grad_batches = max(
            self.batch_size // self.step_batch_size, 1)
        print_log('[batch_size]', self.batch_size)
        print_log('[accumulate_grad_batches]', accumulate_grad_batches)
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
            print_log('[max_time]', max_time)
        trainer = pl.Trainer(
            enable_progress_bar=isatty(),
            fast_dev_run=self.debug,
            accelerator=accelerator, devices=devices,
            precision=precision,
            max_epochs=max_epochs,
            max_time=max_time,
            # gradient_clip_val=hparams.max_grad_norm,
            # k バッチ毎に勾配を蓄積する batch_size * k になる
            accumulate_grad_batches=accumulate_grad_batches,
            callbacks=callbacks,
        )
        if max_epochs > 0:
            trainer.fit(model, data)
        # 最終エポックのモデルを保存 output_path に保存します
        self.tokenizer.save_pretrained(output_path)
        model.model.save_pretrained(output_path)
        self.model_path = output_path

    def extract_filename(self, file):
        if '/' in file:
            _, _, file = file.rpartition('/')
        file = file.replace('.gz', '')
        if not file.endswith('.jsonl'):
            file = file+'.jsonl'
        return file

    def predict(self, test_file, split='train', streaming=False):
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
        if '[' in test_file:
            test_file, _, split = test_file.partition('[')
            split = f'train[{split}'
        batch_size = 1 if self.step_batch_size < 1 else self.step_batch_size
        data = T5DataModule(test_file,
                            transform=self.transform, split=split,
                            batch_size=batch_size,
                            num_of_workers=self.num_of_workers,
                            streaming=streaming)
        if os.path.isdir(self.model_path):
            output_file = f'{self.model_path}/pred_{self.extract_filename(test_file)}'
        else:
            output_file = f'pred_{self.extract_filename(test_file)}'
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
            debug_print('writing', output_file)
            with open(output_file, 'w') as w:
                keys = list(results.keys())
                for idx in range(len(results['pred'])):
                    d = {key: results[key][idx] for key in keys}
                    print(json.dumps(d, ensure_ascii=False), file=w)
        return results


def setup():
    import argparse
    # ハイパーパラメータの読み込み  何も書かなければ、デフォルト値 default
    # python3 finetune.py --batch_size 64
    parser = argparse.ArgumentParser(description='t5train script')
    parser.add_argument('files', type=str, nargs='+', help='jsonl files')
    parser.add_argument('--model_path', default='kkuramitsu/mt5np_mini12L')
    parser.add_argument('--tokenizer_path', default=None)
    parser.add_argument('--output_path', default='model')
    parser.add_argument('--checkpoint_path', default=None)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--source_max_length', type=int, default=None)
    parser.add_argument('--target_max_length', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=256)  # 自動
    parser.add_argument('--step_batch_size', type=int, default=0)  # 自動
    parser.add_argument('--solver', type=str, default='adamw')
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--max_hours', type=float, default=None)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--warmup_steps', type=int, default=1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--float32_matmul_precision', type=str, default=None)
    parser.add_argument('--accelerator', type=str, default=None)
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--precision', default=None)
    parser.add_argument('--early_stopping', action='store_true', default=False)
    parser.add_argument('--fast_dev_run', action='store_true', default=False)
    parser.add_argument('--pretrain', action='store_true', default=False)
    parser.add_argument('--score', type=str, default=None)
    parser.add_argument('--cache', action='store_true', default=False)

    print_log('[script]', ' '.join(sys.argv))

    hparams = parser.parse_args()  # hparams になる
    if hparams.pretrain == True:
        if hparams.step_batch_size > 0:
            # our batches contain roughly 2^16 = 65,536 tokens
            acc_steps = 2**16 // (hparams.max_length * hparams.step_batch_size)
            hparams.batch_size = acc_steps * hparams.step_batch_size
        hparams.solver = 'adafactor'
        # This setting is independent of the "precision" setting in the Trainer.
        torch.set_float32_matmul_precision('medium')
        datasets.config.IN_MEMORY_MAX_SIZE = 40*(1024*1024*1024)
        print_log('[in_memory]', datasets.config.IN_MEMORY_MAX_SIZE)
        if hparams.precision is None:
            hparams.precision = 'bf16'  # A100

    if not isatty():
        disable_progress_bar()

    if not hparams.cache:
        disable_caching()

    # デフォルトがNoneのときは
    if hparams.tokenizer_path is None:
        hparams.tokenizer_path = hparams.model_path
    if hparams.precision is None:
        hparams.precision = 32
    if hparams.source_max_length is None:
        hparams.source_max_length = hparams.max_length
    if hparams.target_max_length is None:
        hparams.target_max_length = hparams.max_length
    set_logfile(hparams.output_path)
    print_log('[hparams]', hparams)
    return hparams


def main():
    hparams = setup()
    train_files = [file for file in hparams.files if '_test.' not in file]
    test_files = [file for file in hparams.files if '_test.' in file]

    model = T5ModelTrainer(
        model_path=hparams.model_path,
        batch_size=hparams.batch_size,
        max_length=hparams.source_max_length,
        target_max_length=hparams.target_max_length,
        step_batch_size=hparams.step_batch_size,
        num_of_workers=hparams.num_workers)
    if len(train_files) > 0:
        t_start = time.time()
        print_log('[train]', train_files)
        model.fit(train_files,
                  random_seed=hparams.seed,
                  accelerator=hparams.accelerator,
                  devices=hparams.devices,
                  precision=hparams.precision,
                  max_epochs=hparams.max_epochs,
                  max_hours=hparams.max_hours,
                  early_stopping=hparams.early_stopping,
                  output_path=hparams.output_path,
                  solver=hparams.solver)
        t_time = (time.time() - t_start)
        t_min = (t_time % 3600) / 60
        print_log(
            f'[trained] {t_time//3600}[H] {t_min}[M] {t_time:.3f}[sec]')
    if len(test_files) > 0:
        print_log('[test]', test_files)
        for test_file in test_files:
            results = model.predict(test_file)
            if hparams.score and 'out' in results and 'pred' in results:
                outfile = get_filename(test_file).replace(
                    '.jsonl', '.csv').replace('.gz', '')
                outfile = f'{hparams.output_path}/{outfile}'
                calc_score(results['out'], results['pred'],
                           outfile, hparams.score, test_file,
                           hparams.model_path, print_fn=print_log)


def main_test():
    hparams = setup()
    model = T5ModelTrainer(
        model_path=hparams.model_path,
        batch_size=hparams.batch_size,
        max_length=hparams.source_max_length,
        target_max_length=hparams.target_max_length,
        step_batch_size=hparams.step_batch_size,
        num_of_workers=hparams.num_workers)
    for test_file in hparams.files:
        model.predict(test_file)
        results = model.predict(test_file)
        if hparams.score and 'out' in results and 'pred' in results:
            outfile = get_filename(test_file).replace('.jsonl', '.csv')
            outfile = f'{hparams.output_path}/{outfile}'
            calc_score(results['out'], results['pred'],
                       outfile, hparams.score, test_file, hparams.model_path)


def main2():
    model = T5ModelTrainer('kkuramitsu/mt5np_mini12L',
                           step_batch_size=32,
                           batch_size=256, num_of_workers=4)
    model.fit('music_train.jsonl.gz',
              max_epochs=1,
              solver='adafactor')
    print(model.predict('music_test.jsonl.gz'))


if __name__ == '__main__':
    main()
