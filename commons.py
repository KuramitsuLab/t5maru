import os
import sys
import torch
import random
import time
from datetime import datetime
import json

def set_seed(seed):  # 乱数シードの設定
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def isatty():
    return sys.stdout.isatty()


## LOGGING

LOG = {}

def record(**kwargs):
    global LOG
    if len(LOG) == 0:
        LOG = dict(date=datetime.now().isoformat(timespec='seconds'), elapsed=time.time())
    LOG.update(kwargs)


def elapsed(t1):
    t2=time.time()
    t_time = int(t2 - t1)
    t_hour = t_time//3600
    t_min = (t_time % 3600) // 60
    t_sec = t_time % 60
    return f'{t_hour:02d}:{t_min:02d}:{t_sec:02d}'

def log_record(msg, output_file=None):
    global LOG
    LOG['elapsed'] = elapsed(LOG['elapsed'])
    line=json.dumps(LOG, ensure_ascii=False)
    line =f'{{"log": "{msg}", {line[1:]}'
    try:
        if output_file:
            if os.path.isdir(output_file):
                output_file = f'{output_file}/t5marulog.jsonl'
            with open(output_file, 'a') as w:
                print(line, file=w)
    finally:
        print(line)
    LOG={}


# def extract_filename(file):
#     if '/' in file:
#         _, _, file = file.rpartition('/')
#     file = file.replace('.gz', '')
#     if not file.endswith('.jsonl'):
#         file = file+'.jsonl'
#     return file.replace('.jsonl', '_tested.jsonl')


DUP = set()


def debug_print(*args, **kwargs):
    if len(DUP) < 512:
        sep = kwargs.get('sep', ' ')
        text = sep.join(str(a) for a in args)
        if text in DUP:
            return
        print('😱', text)
        DUP.add(text)


# LOGFILE = None


# def set_logfile(output_path):
#     global LOGFILE
#     os.makedirs(output_path, exist_ok=True)
#     LOGFILE = f'{output_path}/train_log.txt'


# def print_log(*args, **kwargs):
#     if LOGFILE:
#         sep = kwargs.get('sep', ' ')
#         text = sep.join(str(a) for a in args)
#         try:
#             with open(LOGFILE, 'a') as w:
#                 print(text, file=w)
#         except:
#             pass
#         finally:
#             print('💭', text)
