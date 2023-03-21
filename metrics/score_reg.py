from collections import Counter
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .score import count_score, count_char, count_f1


def tofloat(s, default=0.0):
    try:
        return float(s), 1
    except:
        return default, 0


def check_reg(refs):
    cc = 0
    for ref in refs:
        if '.' in ref:
            cc += 1
        r, c = tofloat(ref)
        if c == 0:
            return False
    return cc > 0


def eval_reg(results, refs, preds):
    default = sum(float(s) for s in refs)/len(refs)
    frefs = []
    fpreds = []
    for ref, pred in zip(refs, preds):
        count_score(results, 'EM', 1 if ref == pred else 0)
        r, _ = tofloat(ref, default)
        p, c = tofloat(pred, default)
        frefs.append(r)
        fpreds.append(p)
        count_score(results, 'otherwise', c)
    y_test = np.array(frefs)
    y_pred = np.array(fpreds)
    results['MSE'] = mean_squared_error(y_test, y_pred)
    results['RMSE'] = mean_squared_error(y_test, y_pred, squared=False)
    results['MAE'] = mean_absolute_error(y_test, y_pred)
    results['R^2'] = r2_score(y_test, y_pred)
    r, p = pearsonr(y_test, y_pred)
    results['pearsonr'] = r
    results['pearsonr_p'] = p
    r, p = spearmanr(y_test, y_pred)
    results['spearmanr'] = r
    results['spearmanr_p'] = p


def ctokenize(ref, pred):
    trefs = list(ref)
    tpreds = list(pred)
    return trefs, tpreds


def head(s):
    if '=' in s:
        return s.split('=')[0]
    return s


def categories(ref):
    if ' ' in ref:
        return [head(s) for s in ref.split()]
    return [head(ref)]


def eval_class(results, refs, preds, top_k=10, tokenize=ctokenize):
    cats = Counter()
    for ref in refs:
        cats.update(categories(ref))

    for ref, pred in zip(refs, preds):
        count_char(results, refs, preds)
        trefs, tpreds = tokenize(ref, pred)
        count_f1(results, trefs, tpreds)

    if top_k > 1:
        for k, c in cats.most_common(top_k):
            results[f'{k}'] = []
            results[f'{k}_r'] = []
            results[f'{k}_p'] = []
            results[f'{k}_c'] = c
        for ref, pred in zip(refs, preds):
            cat_refs = categories(ref)
            cat_preds = categories(pred)
            for k, c in cats.most_common(top_k):
                if k in cat_refs:
                    count_score(results, f'{k}_r', 1 if k in cat_preds else 0)
            for cp in cat_preds:
                if cp in cats:
                    count_score(results, f'{k}_p', 1 if k in cat_refs else 0)
                    count_score(results, f'otherwise', 0)
                else:
                    count_score(results, f'otherwise', 1)
