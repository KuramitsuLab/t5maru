from .score_py import score_py, get_filename


def calc_score(refs: list, preds: list, outputfile,
               lang='py', testfile='', model_id='', print_fn=print):
    score_py(refs, preds, outputfile, testfile, model_id, print_fn=print_fn)
