import time
import os
import sys
import random
import json

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from transformers import (
    # AutoTokenizer,
    # AutoModelForSeq2SeqLM,
    # get_linear_schedule_with_warmup,
    default_data_collator,
)

# from transformers.optimization import Adafactor, AdafactorSchedule

import pytorch_lightning as pl

# from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# from .metrics import eval_score, write_score_csv

import datasets
from datasets import load_dataset, load_from_disk, disable_caching, Dataset
from datasets.utils import disable_progress_bar

disable_caching()
datasets.utils.logging.set_verbosity_warning()

import pandas as pd

# https://www.kaggle.com/code/noriyukipy/text-classification-dataloader-from-datasets


def transform_nop(x):
    return x

# JSONLファイルの前処理

import gzip


def zopen(file):
    if file.endswith(".gz"):
        return gzip.open(file, "tr", encoding="utf-8", errors="ignore")
    return open(file, encoding="utf-8", errors="ignore")


def prepare_jsonl(
    jsonl_file,
    prompt="",
    source="in",
    target="out",
    lines=None,
    working_file="temp.jsonl",
    mode="w",
):
    if jsonl_file.startswith("__"):
        mods = jsonl_file.split("__")
        if len(mods) > 2:
            prompt = f"{mods[1]}: "
        if len(mods) > 3:
            source = mods[2]
        if len(mods) > 4:
            target = mods[3]
    if lines is not None:  # in-memory mode
        with zopen(jsonl_file) as f:
            for line in f.readlines():
                d = json.loads(line)
                lines.append((f"{prompt}{d[source]}", f"{d[target]}"))
        return
    with open(working_file, mode) as w:
        with zopen(jsonl_file) as f:
            for line in f.readlines():
                d = json.loads(line)
                line = json.dumps(
                    {"in": f"{prompt}{d[source]}", "out": f"{d[target]}"},
                    ensure_ascii=False,
                )
                print(line, file=w)


def prepare_DataFrame(
    df, source=0, target=1, lines=None, working_file="temp.jsonl", mode="w"
):
    if lines is not None:  # in-memory mode
        for i in range(len(df)):
            lines.append((f"{df.iloc[i][source]}", f"{df.iloc[i][target]}"))
        return
    with open(working_file, mode) as w:
        for i in range(len(df)):
            line = json.dumps(
                {"in": f"{df.iloc[i][source]}", "out": f"{df.iloc[i][target]}"},
                ensure_ascii=False,
            )
            print(line, file=w)


def shuffle_and_downsize(lines, shuffle=True, downsizing=None):
    if shuffle:
        random.shuffle(lines)
    if isinstance(downsizing, (int, float)):
        if downsizing < 1.0:
            downsizing = int(len(lines) * downsizing)
        lines = lines[:downsizing]
    return lines


def prepare_working_file(
    data_sources,
    lines=None,
    prefix="temp",
    split="train",
    shuffle=True,
    downsizing=None,
):
    working_file = f"{prefix}_{split}.jsonl"
    mode = "w"
    for data_source in data_sources:
        if isinstance(data_source, pd.DataFrame):
            prepare_DataFrame(
                data_source, lines=lines, working_file=working_file, mode=mode
            )
        elif isinstance(data_source, str):
            if data_source.endswith(".jsonl") or data_source.endswith(".jsonl.gz"):
                prepare_jsonl(
                    data_source, lines=lines, working_file=working_file, mode=mode
                )
    if lines is not None:
        lines = shuffle_and_downsize(lines, shuffle=shuffle, downsizing=downsizing)
        return lines, len(lines)
    with open(working_file) as f:
        lines = f.readlines()
    lines = shuffle_and_downsize(lines, shuffle=shuffle, downsizing=downsizing)
    with open(working_file, "w") as f:
        f.writelines(lines)
    return working_file, len(lines)


def prepare_train_file(
    data_sources, prefix=None, pretrain=False, shuffle=True, downsizing=None
):
    if not isinstance(data_sources, (list, tuple)):
        data_sources = [data_sources]
    lines = [] if prefix is None else None
    return prepare_working_file(
        data_sources, lines=lines, prefix=prefix, split="train", downsizing=downsizing
    )


def create_valid_file(train_data, pretrain):
    valid_file = None
    if isinstance(train_data, str):
        with open(train_data) as f:
            lines = f.readlines()
        valid_file = train_data.replace("_train.", "_valid.")
    else:
        lines = train_data
    if pretrain:
        lines = lines[:512]
        if valid_file:
            with open(valid_file, "w") as f:
                f.writelines(lines)
            return valid_file
        return lines
    valid_size = len(lines) // 10
    valid_lines = lines[:valid_size]
    lines = lines[valid_size:]
    if valid_file:
        with open(train_data, "w") as f:
            f.writelines(lines)
        with open(valid_file, "w") as f:
            f.writelines(valid_lines)
        return valid_file
    else:
        train_data.reset()
        train_data.extend(lines)
        lines = valid_lines
        return lines


def prepare_valid_file(
    data_sources, train_file, prefix=None, pretrain=False, downsizing=None
):
    if not isinstance(data_sources, (list, tuple)):
        data_sources = [data_sources]
    valids = []
    for data_source in data_sources:
        if isinstance(data_source, str):
            if "_train." in data_source:
                valid_file = data_source.replace("_train.", "_valid.")
                if os.path.isfile(valid_file):
                    valids.append(valid_file)
                    continue
                valid_file = data_source.replace("_train.", "_dev.")
                if os.path.isfile(valid_file):
                    valids.append(valid_file)
    # もしprefixがNoneならin-memory
    lines = [] if prefix is None else None
    if len(valids) > 0:
        valid_file, valid_len = prepare_working_file(
            valids,
            lines=lines,
            prefix=prefix,
            split="valid",
            shuffle=False,
            downsizing=downsizing,
        )
        return valid_file
    else:
        return create_valid_file(train_file, pretrain=pretrain)


def generator(data):
    def genetator_fn():
        for d in data:
            yield {"in": d[0], "out": d[1]}

    return genetator_fn


class T5TrainFileModule(pl.LightningDataModule):
    def __init__(
        self,
        data_sources,
        use_valid=True,
        prefix="temp",
        shuffle=True,
        downsizing=None,
        transform=transform_nop,
        batch_size=32,
        num_of_workers=4,
    ):
        super().__init__()
        self.train_file, self.train_size = prepare_train_file(
            data_sources, prefix=prefix, shuffle=shuffle, downsizing=downsizing
        )
        if use_valid:
            self.valid_file = prepare_valid_file(
                data_sources, self.train_file, prefix=prefix, downsizing=downsizing
            )
        else:
            self.valid_file = None
        self.ds_train = None
        self.ds_valid = None
        self.transform = transform
        self.batch_size = batch_size
        self.num_of_workers = num_of_workers

    def __enter__(self):
        return self

    def __exit__(self, ex_type, ex_value, trace):
        if self.ds_train:
            self.ds_train.cleanup_cache_files()
        if self.ds_valid:
            self.ds_valid.cleanup_cache_files()
        # if isinstance(self.train_file, str):
        #     os.remove(self.train_file)
        # if isinstance(self.valid_file, str):
        #     os.remove(self.valid_file)
        return False

    def load(self, data):
        if isinstance(data, list):
            ds = Dataset.from_generator(generator(data))
        else:
            ds = load_dataset("json", data_files=data, split="train")
        ds = ds.map(self.transform, num_proc=self.num_of_workers).with_format("torch")
        return ds

    def setup(self, stage: str):
        if stage == "fit":
            self.ds_train = self.load(self.train_file)
            if self.valid_file:
                self.ds_valid = self.load(self.valid_file)
        if stage == "test":
            self.ds_train = self.load(self.train_file)

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            collate_fn=default_data_collator,
            num_workers=self.num_of_workers,
            drop_last=True,
            shuffle=True,
            batch_size=self.batch_size,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_valid,
            collate_fn=default_data_collator,
            num_workers=self.num_of_workers,
            batch_size=self.batch_size,
        )

    def test_dataloader(self):
        return DataLoader(
            self.ds_train,
            # collate_fn=default_data_collator,
            num_workers=self.num_of_workers,
            batch_size=self.batch_size,
        )


def T5TestFileModule(
    data_sources,
    use_valid=False,
    prefix=None,
    shuffle=True,
    downsizing=None,
    transform=transform_nop,
    batch_size=32,
    num_of_workers=4,
):
    return T5TrainFileModule(
        data_sources,
        use_valid=False,
        prefix=None,
        shuffle=False,
        downsizing=downsizing,
        transform=transform,
        batch_size=batch_size,
        num_of_workers=num_of_workers,
    )


def main():
    with T5TrainFileModule("sample/music_train.jsonl.gz") as dm:
        dm.setup("fit")
        for batch in dm.train_dataloader():
            print(batch)

    with T5TestFileModule("sample/music_test.jsonl.gz", batch_size=1) as dm:
        dm.setup("test")
        for batch in dm.test_dataloader():
            print(batch)


if __name__ == "__main__":
    main()
