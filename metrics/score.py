import json
from difflib import SequenceMatcher
import Levenshtein
from .bleu import sentence_bleu, SmoothingFunction


def remove_prefix_tag(s):
    s = s.replace('<nl>', '\n').replace('<tag>', '    ')
    if s.startswith('<') and '>' in s:
        _, _, s = ref.partition('<')
    return s


def count_score(results: dict, key: str, score: float):
    if key not in results:
        results[key] = []
    if score is not None:
        results[key].append(score)


def count_lcs(results, ref, pred):
    if len(ref) > 0 and len(pred) > 0:
        match = SequenceMatcher(None, ref, pred).find_longest_match(
            0, len(ref), 0, len(pred))
        # Recall: refの共通部分文字列を、どれだけ当てられたか
        count_score(results, 'LCS_r', match.size/len(pred))
        # Precision: predの共通部分文字列を、どれだけ含んでいるか？
        count_score(results, 'LCS_p', match.size/len(ref))


def count_f1(results, trefs, tpreds, base='F1', filter_fn=lambda x: True):
    trefs = set(t for t in trefs if filter_fn(t))
    tpreds = set(t for t in tpreds if filter_fn(t))
    count_score(results, f'{base}', None)
    recall = None
    precision = None
    if len(trefs) > 0:
        # Recall: refの単語を、どれだけ当てられたか
        recall = 0
        for ref in list(trefs):
            if ref in tpreds:
                recall += 1
        recall = recall/len(trefs)
        count_score(results, f'{base}_r', recall)
    else:
        count_score(results, f'{base}_r', None)
    if len(tpreds) > 0:
        # Precision: 生成した要約が、どれだけ人手の要約に含まれているか
        precision = 0
        for pred in list(tpreds):
            if pred in trefs:
                precision += 1
        precision = precision/len(tpreds)
        count_score(results, f'{base}_p', precision)
    else:
        count_score(results, f'{base}_p', None)


def count_char(results, ref, pred):
    count_score(results, 'ExactMatch', 1.0 if ref == pred else 0.0)
    count_score(results, 'EditSim', Levenshtein.ratio(ref, pred))
    count_lcs(results, ref, pred)
    count_f1(results, list(ref), list(pred), 'F1_Token')

# BLEU


def count_blue(results, trefs, tpreds):
    count_f1(results, trefs, tpreds, 'F1_Token')
    smoother = SmoothingFunction()
    trefs_list = [trefs]
    # score = corpus_bleu(trefs_list, tpreds)
    # count_score('BLEU', score, results)
    try:
        score = sentence_bleu(trefs_list, tpreds)
        count_score(results, 'BLEU', score)
    except ZeroDivisionError as e:
        print('ERR BLEU', e, trefs, tpreds)
    # try:
    #     b2 = sentence_bleu(trefs_list, tpreds,
    #                        smoothing_function=smoother.method2)
    #     count_score('BLEU2', b2, results)
    # except ZeroDivisionError as e:
    #     print('ERR BLEU2', e, trefs, tpreds)
    # try:
    #     b4 = sentence_bleu(trefs_list, tpreds,
    #                        smoothing_function=smoother.method4)
    #     count_score('BLEU4', b4, results)
    # except ZeroDivisionError as e:
    #     print('ERR BLEU4', e, trefs, tpreds)


def harmonic_mean(r, p, beta=1.0):
    return ((1+beta**2)*r*p)/(r+(beta**2)*p)


def calc_hmean(results, beta=1.0, print_fn=print):
    for key in results:
        value = results[key]
        if isinstance(value, list):
            if len(value) > 0:
                value = 100.0 * sum(value)/len(value)
            else:
                value = ''
            results[key] = value
    for key in results:
        if f'{key}_p' in results and f'{key}_r' in results:
            pval = results[f'{key}_p']
            rval = results[f'{key}_r']
            if isinstance(pval, float) and isinstance(rval, float):
                value = results[key] = harmonic_mean(pval, rval, 1.0)
    for key in results:
        if isinstance(value, float):
            print_fn(f'{key}: {value:.3f}')
        else:
            print_fn(f'{key}: {value}')


def read_jsonl(filename, ref='out', pred='pred'):
    refs = []
    preds = []
    with open(filename) as f:
        for line in f.readlines():
            data = json.loads(line)
            refs.append(data[ref])
            preds.append(data[pred])
    return refs, preds


def setup():
    import argparse
    # ハイパーパラメータの読み込み  何も書かなければ、デフォルト値 default
    # python3 finetune.py --batch_size 64
    parser = argparse.ArgumentParser(description='t5train script')
    parser.add_argument('files', type=str, nargs='+', help='jsonl files')
    parser.add_argument('--model_path', default='kkuramitsu/mt5np_mini12L')
    parser.add_argument('--output', default=None)
    parser.add_argument('--ref', default='out')
    parser.add_argument('--pred', default='pred')

    hparams = parser.parse_args()  # hparams になる
    return hparams


def main(filepath, outputfile=None):
    hparams = setup()
    for filename in hparams.files:
        refs, preds = read_jsonl(filepath, ref=hparams.ref, pred=hparams.pred)
        outputfile = filepath.replace('.jsonl', '.csv')
        score_reg(refs, preds, outputfile, filepath)
