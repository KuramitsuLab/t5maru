import os
import sys
import torch
import random

def extract_filename(file):
    if '/' in file:
        _, _, file = file.rpartition('/')
    file = file.replace('.gz', '')
    if not file.endswith('.jsonl'):
        file = file+'.jsonl'
    return file.replace('.jsonl', '_tested.jsonl')

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
