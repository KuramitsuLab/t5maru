LOGFILE = None


def set_logfile(output_path):
    global LOGFILE
    os.makedirs(output_path, exist_ok=True)
    LOGFILE = f'{output_path}/train_log.txt'


def t5print(*args, **kwargs):
    sep = kwargs.get('sep', ' ')
    text = sep.join(str(a) for a in args)
    try:
        if LOGFILE:
            with open(LOGFILE, 'a') as w:
                print(text, file=w)
    except:
        pass
    finally:
        print('💭', text)
