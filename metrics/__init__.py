from .score_py import score_py, get_filename


def dump_score(refs: list, preds: list, outputfile, lang='py', testfile='', model_id=''):
    score_py(refs, preds, outputfile, testfile, model_id)
