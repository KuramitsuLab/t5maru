# import dill as pickle
# import glob
import os

import torch
# from torch.utils.data import DataLoader
# from torch.optim import AdamW

from transformers import (
    AutoTokenizer
)

import datasets
from datasets import load_dataset, load_from_disk, disable_caching
from datasets.utils import disable_progress_bar
disable_caching()
disable_progress_bar()
datasets.utils.logging.set_verbosity_warning()

tokenizer = None
max_length = 128
target_max_length = 128


def transform(jsonl: dict):
    inputs = tokenizer.batch_encode_plus(
        [jsonl['in']],
        max_length=max_length,
        truncation=True,
        pad_to_max_length=True,
        padding="max_length", return_tensors="pt")
    source_ids = inputs["input_ids"].squeeze()
    source_mask = inputs["attention_mask"].squeeze()
    targets = tokenizer.batch_encode_plus(
        [jsonl['out']],
        max_length=target_max_length,
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


def check_saved_path(filename):
    if '/' in filename:
        _, _, filename = filename.rpartition('/')
    saved = filename.replace('.gz', '').replace('.jsonl', '_saved')
    return saved, os.path.isdir(saved)


def dump(filename, split='train'):
    saved_path, has_saved = check_saved_path(filename)
    if has_saved:
        return load_from_disk(saved_path)
    else:
        ds = load_dataset('json',
                          data_files=filename, split=split)
        ds = ds.map(
            transform, num_proc=4).with_format('torch')
        ds.save_to_disk(saved_path)
        return ds


def setup():
    global tokenizer, max_length, target_max_length
    import argparse
    # ハイパーパラメータの読み込み  何も書かなければ、デフォルト値 default
    # python3 finetune.py --batch_size 64
    parser = argparse.ArgumentParser(description='t5dump script')
    parser.add_argument('files', type=str, nargs='+', help='jsonl files')
    parser.add_argument('--model_path', default='kkuramitsu/mt5np_mini12L')
    parser.add_argument('--tokenizer_path', default=None)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--target_max_length', type=int, default=None)

    hparams = parser.parse_args()  # hparams になる

    # デフォルトがNoneのときは
    if hparams.tokenizer_path is None:
        hparams.tokenizer_path = hparams.model_path
    tokenizer = AutoTokenizer.from_pretrained(
        hparams.tokenizer_path, use_fast=False)
    max_length = hparams.max_length
    if hparams.target_max_length is None:
        target_max_length = max_length
    return hparams


def main():
    hparams = setup()
    for filename in hparams.files:
        if filename.endswith('jsonl') or filename.endswith('jsonl.gz'):
            dump(filename)


if __name__ == '__main__':
    main()
