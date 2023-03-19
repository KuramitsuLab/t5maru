from .score import remove_prefix_tag, calc_hmean
from .score_reg import check_reg, eval_reg, eval_class
from .score_py import eval_py, eval_ja, eval_en


def eval_score(results: dict, refs: list, preds: list, score='auto', top_k=0, print_fn=print):
    refs = [remove_prefix_tag(s) for s in refs]
    preds = [remove_prefix_tag(s) for s in preds]
    if score == 'auto':
        if check_reg(refs):
            score = 'reg'
        score = 'py'
    results['Score'] = score
    if score == 'reg':
        eval_reg(results, refs, preds)
    elif score == 'class':
        eval_class(results, refs, preds, top_k=top_k)
    elif score == 'py':
        eval_py(results, refs, preds)
    elif score == 'ja':
        eval_ja(results, refs, preds)
    else:
        eval_en(results, refs, preds)
    calc_hmean(results, print_fn=print_fn)
    return results


def write_score_csv(results: dict, outputfile: str, print_fn=print):
    print_fn('[writing]', outputfile)
    with open(outputfile, 'a') as w:
        for key in results:
            print(f'{key},', end='', file=w)
        print(file=w)
        for key, value in results.items():
            if isinstance(value, float):
                print(f'{value:.3f},', end='', file=w)
            else:
                print(f'{value},', end='', file=w)
        print(file=w)
