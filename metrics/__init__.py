from .score_py import eval_dump_py, get_filename


def dump_score(refs: list, preds: list, outputfile, lang='py', testfile='', model_id=''):
    eval_dump_py(refs, preds, outputfile, testfile, model_id)
