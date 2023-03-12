import sys
import json
from difflib import SequenceMatcher
import black
import re
import Levenshtein
from .bleu import sentence_bleu, SmoothingFunction

# 前処理

BLACK_MODE = black.Mode()


def normalize(ref, pred, remove_tag=True):
    ref = ref.replace('<nl>', '\n').replace('<tab>', '    ')
    pred = pred.replace('<nl>', '\n').replace('<tab>', '    ')
    if remove_tag:
        # 先頭のタグを消す
        if ref.startswith('<') and '>' in ref:
            _, _, ref = ref.partition('<')
        if pred.startswith('<') and '>' in pred:
            _, _, pred = pred.partition('<')
    try:
        ref = black.format_str(ref, mode=BLACK_MODE)[:-1]
    except:
        pass
    try:
        pred = black.format_str(pred, mode=BLACK_MODE)[:-1]
        return ref, pred, 1
    except:
        return ref, pred, 0


# 字句処理

def substr(s, start, end):
    while end + 1 < len(s):
        if s[end] != ' ':
            break
        end += 1
    return s[start:end], end


PID = re.compile('[A-Za-z0-9_]+')
PNUM = re.compile(r'[0-9\.]+')
OP = '+*-/=<>!|&%@~'


def extract_identifier(tokens: list, s: str, start):
    end = re.match(PID, s[start:])
    if end:
        t, end = substr(s, start, start+end.end())
        tokens.append(t)
        return end
    # print(end, s[start], start, s)
    tokens.append(s[start:start+1])
    return start+1


def extract_num(tokens: list, s: str, start):
    end = re.match(PNUM, s[start:])
    if end:
        t, end = substr(s, start, start+end.end())
        tokens.append(t)
        return end
    # print(end, s)
    tokens.append(s[start:start+1])
    return start+1


def extract_op(tokens: list, s: str, start):
    end = start+1
    while end < len(s):
        if s[end] not in OP:
            break
        end += 1
    t, end = substr(s, start, end)
    tokens.append(t)
    return end


def extract_quote(tokens: list, s: str, q, start):
    shift = start+1
    while shift < len(s)+1:
        end = s.find(q, shift)
        if end == -1:
            tokens.append(s[start])
            return start+1
        if s[end-1] != '\\':
            break
        shift = end+1
    t, end = substr(s, start, end+1)
    tokens.append(t)
    return end


def extract_tquote(tokens: list, s: str, tq, start, remove_docstring=True):
    end = s.find(tq, start+3)
    if end == -1:
        tokens.append(s[start:start+3])
        return start+3
    t, end = substr(s, start, end+3)
    if remove_docstring and len(tokens) > 0:
        if tokens[-1].startswith('<nl>') or tokens[-1].startswith('<tab>'):
            return end
    tokens.append(t)
    return end


def extract_comment(tokens: list, s: str, start, remove_comment=True):
    end = s.find('\n', start+1)
    if end == -1:
        end = len(s)
    if not remove_comment:
        tokens.append(s[start:end])
    return end


def extract_indent(tokens: list, s: str, start):
    tokens.append('<nl>')
    shift = start+1
    ss = []
    while True:
        if s.startswith('    ', shift):
            ss.append('<tab>')
            shift += 4
            continue
        if s.startswith('\t', shift):
            ss.append('<tab>')
            shift += 1
            continue
        break
    tokens.append(''.join(ss))
    return shift


def tokenize(s: str, remove_space=True, remove_docstring=True, remove_comment=True):
    start = 0
    tokens = []
    while start < len(s):
        c = s[start]
        if c.isidentifier():
            start = extract_identifier(tokens, s, start)
        elif c.isdigit():
            start = extract_num(tokens, s, start)
        elif c in OP:
            start = extract_op(tokens, s, start)
        elif c == '\n':
            start = extract_indent(tokens, s, start)
        elif s.startswith('"""', start):
            start = extract_tquote(
                tokens, s, '"""', start, remove_docstring=remove_docstring)
        elif s.startswith("'''", start):
            start = extract_tquote(
                tokens, s, "'''", start, remove_docstring=remove_docstring)
        elif s.startswith('"', start):
            start = extract_quote(tokens, s, '"', start)
        elif s.startswith("'", start):
            start = extract_quote(tokens, s, "'", start)
        elif s.startswith("#", start):
            start = extract_comment(
                tokens, s, start, remove_comment=remove_comment)
        else:
            t, start = substr(s, start, start+1)
            tokens.append(t)
    if remove_space:
        return [t.strip() for t in tokens if len(t.strip()) > 0]
    return tokens


def record_score(key: str, score: float, results=None):
    if results:
        if key not in results:
            results[key] = []
        if score is not None:
            results[key].append(score)


def calc_LCS(ref, pred, results):
    match = SequenceMatcher(None, ref, pred).find_longest_match(
        0, len(ref), 0, len(pred))
    # Recall: refの共通部分文字列を、どれだけ当てられたか
    record_score('LCS_r', match.size/len(pred), results)
    # Precision: predの共通部分文字列を、どれだけ含んでいるか？
    record_score('LCS_p', match.size/len(ref), results)
    # 調和平均 F1
    record_score('LCS', match.size*2/(len(pred)+len(ref)), results)

# https://qiita.com/icoxfog417/items/65faecbbe27d3c53d212


def calc_rouge(base, ref_tokens, pred_tokens, filter_fn, results=None):
    # 識別子だけ取り出す
    ref_ids = set(t for t in ref_tokens if filter_fn(t))
    pred_ids = set(t for t in pred_tokens if filter_fn(t))
    recall = None
    precision = None
    if len(ref_ids) > 0:
        # Recall: refの単語を、どれだけ当てられたか
        recall = 0
        for ref_id in list(ref_ids):
            if ref_id in pred_ids:
                recall += 1
        recall = recall/len(ref_ids)
        record_score(f'{base}_r', recall, results)
    else:
        record_score(f'{base}_r', None, results)
    if len(pred_ids) > 0:
        # Precision: 生成した要約が、どれだけ人手の要約に含まれているか
        precision = 0
        for pred_id in list(pred_ids):
            if pred_id in ref_ids:
                precision += 1
        precision = precision/len(pred_ids)
        record_score(f'{base}_p', precision, results)
    else:
        record_score(f'{base}_p', None, results)
    record_score(f'{base}', None, results)


OP_SET = set(
    '+ - * ** / // % = != == < > <= >= & | << >> ^ ~ @ not and or += -= *= **= /= //= %= |= &= ->'.split())
PUNC_SET = set('( ) [ ] { } . , : ;'.split())


def is_isidentifier(t: str):
    if t not in OP_SET:
        return t[0].isidentifier()
    return False


def is_isnumber(t: str):
    return t.isnumeric()


def is_isstring(t: str):
    return t.startswith('"') or t.startswith("'")


def is_isoperator(t: str):
    return t in OP_SET


def calc_CodeROUGE(ref_tokens, pred_tokens, results=None):
    calc_rouge('ROUGE-1', ref_tokens, pred_tokens, lambda t: True, results)
    calc_rouge('ROUGE-I', ref_tokens, pred_tokens, is_isidentifier, results)
    calc_rouge('ROUGE-In', ref_tokens, pred_tokens, is_isnumber, results)
    calc_rouge('ROUGE-Is', ref_tokens, pred_tokens, is_isstring, results)
    calc_rouge('ROUGE-Io', ref_tokens, pred_tokens, is_isoperator, results)

# BLEU


def calc_blue(ref_tokens, pred_tokens, results=None):
    smoother = SmoothingFunction()
    ref_tokens_list = [ref_tokens]
    # score = corpus_bleu(ref_tokens_list, pred_tokens)
    # record_score('BLEU', score, results)
    try:
        score = sentence_bleu(ref_tokens_list, pred_tokens)
        record_score('BLEU', score, results)
    except ZeroDivisionError as e:
        print('ERR BLEU', e, ref_tokens, pred_tokens)
    try:
        b2 = sentence_bleu(ref_tokens_list, pred_tokens,
                           smoothing_function=smoother.method2)
        record_score('BLEU2', b2, results)
    except ZeroDivisionError as e:
        print('ERR BLEU2', e, ref_tokens, pred_tokens)
    try:
        b4 = sentence_bleu(ref_tokens_list, pred_tokens,
                           smoothing_function=smoother.method4)
        record_score('BLEU4', b4, results)
    except ZeroDivisionError as e:
        print('ERR BLEU4', e, ref_tokens, pred_tokens)


def calc_PythonCode(refs, preds, results=None):
    for ref, pred in zip(refs, preds):
        ref, pred, c = normalize(ref, pred)
        record_score('SyntaxPass', c, results)
        record_score('ExactMatch', 1.0 if ref == pred else 0.0, results)
        record_score('EditSim', Levenshtein.ratio(ref, pred), results)
        calc_LCS(ref, pred, results)
        ref_tokens = tokenize(ref)
        pred_tokens = tokenize(pred)
        calc_CodeROUGE(ref_tokens, pred_tokens, results)
        calc_blue(ref_tokens, pred_tokens, results)


def get_filename(filepath):
    if '/' in filepath:
        _, _, filename = filepath.rpartition('/')
        return filename
    return filepath


def score_py(refs, preds, outputfile, testfile='', model_id=''):
    results = {
        'Model': model_id,
        'Test': get_filename(testfile),
        'Total': len(refs),
    }
    calc_PythonCode(refs, preds, results)
    for key in results:
        value = results[key]
        if isinstance(value, list):
            if len(value) > 0:
                value = 100.0 * sum(value)/len(value)
            else:
                if f'{key}_p' in results:
                    value = results[key] = 0.5 * \
                        results[f'{key}_p'] + 0.5 * results[f'{key}_r']
                else:
                    value = ''
            results[key] = value
        if isinstance(value, float):
            print(f'{key}: {value:.3f}')
        else:
            print(f'{key}: {value}')
    if outputfile:
        print('writing', outputfile)
        with open(outputfile, 'w') as w:
            for key in results:
                print(f'{key},', end='', file=w)
            print(file=w)
            for key, value in results.items():
                if isinstance(value, float):
                    print(f'{value:.3f},', end='', file=w)
                else:
                    print(f'{value},', end='', file=w)
            print(file=w)


def read_jsonl(filename):
    refs = []
    preds = []
    with open(filename) as f:
        for line in f.readlines():
            data = json.loads(line)
            refs.append(data['out'])
            preds.append(data['pred'])
    return refs, preds


def main_calc(filepath, outputfile=None):
    refs, preds = read_jsonl(filepath)
    if outputfile is None:
        outputfile = filepath.replace('.jsonl', '.csv')
    score_py(refs, preds, outputfile, filepath)


if __name__ == '__main__':
    for file in sys.argv[1:]:
        if file.endswith('.jsonl'):
            main_calc(file)
