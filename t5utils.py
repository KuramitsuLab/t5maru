import os
import argparse
import json
import pandas as pd

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    MT5ForConditionalGeneration
)

from .t5log import set_logfile, print_log

def setup_cp():
    parser = argparse.ArgumentParser(description='t5cp - duplicate model')
    parser.add_argument('--model_path', default='kkuramitsu/t5jep')
    parser.add_argument('--tokenizer_path', default=None)
    parser.add_argument('--output_path', default='model')
    hparams = parser.parse_args()  # hparams になる

    if hparams.tokenizer_path is None:
        hparams.tokenizer_path = hparams.model_path
    return hparams

def main_cp():
    hparams = setup_cp()
    tokenizer = AutoTokenizer.from_pretrained(hparams.tokenizer_path, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(hparams.model_path)
    tokenizer.save_pretrained(self.output_path)
    model.save_pretrained(self.output_path)
    set_logfile(self.output_path)
    print_log('[source]', hparams.model_path, hparams.tokenizer_path)

def setup_new():
    parser = argparse.ArgumentParser(description='t5new - create a new vanilla model')
    parser.add_argument('--model_path', default='kkuramitsu/t5jep')
    parser.add_argument('--tokenizer_path', default=None)
    parser.add_argument('--output_path', default='model')
    parser.add_argument('--vocab_size', type=int, default=None)
    parser.add_argument('--d_model', type=int, default=None)
    parser.add_argument('--d_kv', type=int, default=None)
    parser.add_argument('--d_ff', type=int, default=None)
    parser.add_argument('--num_layers', type=int, default=None)
    parser.add_argument('--num_decoder_layers', type=int, default=None)
    parser.add_argument('--num_heads', type=int, default=None)
    parser.add_argument('--dropout_rate', type=float, default=None)

    hparams = parser.parse_args()  # hparams になる

    if hparams.tokenizer_path is None:
        hparams.tokenizer_path = hparams.model_path
    return hparams

def main_new():
    hparams = setup_new()
    tokenizer = AutoTokenizer.from_pretrained(hparams.tokenizer_path, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(hparams.model_path)
    config = model.config
    if hparams.vocab_size is not None:
        config.vocab_size = hparams.vocab_size
    if hparams.d_model is not None:
        config.d_model = hparams.d_model
    if hparams.d_kv is not None:
        config.d_kv = hparams.d_kv
    if hparams.d_ff is not None:
        config.d_ff = hparams.d_ff
    if hparams.num_layers is not None:
        config.num_layers = hparams.num_layers
    if hparams.num_decoder_layers is not None:
        config.num_decoder_layers = hparams.num_decoder_layers
    if hparams.num_heads is not None:
        config.num_heads = hparams.num_heads
    if hparams.dropout_rate is not None:
        config.dropout_rate = hparams.dropout_rate
    model = MT5ForConditionalGeneration(config)
    tokenizer.save_pretrained(self.output_path)
    model.save_pretrained(self.output_path)
    set_logfile(self.output_path)
    print_log('[model.config]', config)


##
## t5len
##

def setup_len():
    parser = argparse.ArgumentParser(description='t5len - check tokenizer length')
    parser.add_argument('files', type=str, nargs='+', help='jsonl files')
    parser.add_argument('--model_path', default='kkuramitsu/t5jep')
    parser.add_argument('--tokenizer_path', default=None)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--target_max_length', type=int, default=None)
    hparams = parser.parse_args()
    if hparams.tokenizer_path is None:
        hparams.tokenizer_path = hparams.model_path
    if hparams.target_max_length is None:
        hparams.target_max_length = hparams.max_length
    return hparams


def main_len():
    hparams = setup_len()
    tokenizer = AutoTokenizer.from_pretrained(
        hparams.tokenizer_path, use_fast=False)
    for file in hparams.files:
        in_lens = []
        out_lens = []
        with open(file) as f:
            for line in f.readlines():
                d = json.loads(line)
                ids = tokenizer.encode(d['in'])
                in_lens.append(len(ids))
                if len(ids) > hparams.max_length:
                    print(len(ids), d['in'], '\n', ids)
                ids = tokenizer.encode(d['out'])
                out_lens.append(len(ids))
                if len(ids) > hparams.target_max_length:
                    print(len(ids), d['out'], '\n', ids)
        print(file)
        df = pd.DataFrame({'in': in_lens, 'out': out_lens})
        print(df.describe())


if __name__ == '__main__':  # わかります
    main_len()
