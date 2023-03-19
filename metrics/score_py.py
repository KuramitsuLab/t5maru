from sumeval.metrics.rouge import RougeCalculator
from janome.tokenizer import Tokenizer
from collections import Counter
import json
from difflib import SequenceMatcher
import black
import re
from .score import count_score, count_char, count_f1, count_blue, count_rouge

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
    end = start+1
    while end < len(s):
        if not s[end].isidentifier():
            break
        end += 1
    t, end = substr(s, start, end)
    tokens.append(t)
    return end


def extract_identifier0(tokens: list, s: str, start):
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


# https://qiita.com/icoxfog417/items/65faecbbe27d3c53d212


# CodeRouge


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


def count_crouge(results, trefs, tpreds):
    count_f1(results, trefs, tpreds, 'CROUGE-1', lambda t: True, )
    count_f1(results, trefs, tpreds, 'CROUGE-I', is_isidentifier)
    count_f1(results, trefs, tpreds, 'CROUGE-NUM', is_isnumber)
    count_f1(results,  trefs, tpreds, 'CROUGE-STR', is_isstring)
    count_f1(results, trefs, tpreds, 'CROUGE-OP', is_isoperator)


def eval_py(results, refs, preds):
    for ref, pred in zip(refs, preds):
        ref, pred, c = normalize(ref, pred)
        count_score(results, 'SyntaxPass', c)
        count_char(results, ref, pred)
        trefs = tokenize(ref)
        tpreds = tokenize(pred)
        count_blue(results, trefs, tpreds)
        count_rouge(results, trefs, tpreds, lang='py')
        count_crouge(results, trefs, tpreds)


def eval_en(results, refs, preds):
    for ref, pred in zip(refs, preds):
        count_char(results, ref, pred)
        trefs = ref.split()
        tpreds = pred.split()
        count_blue(results, trefs, tpreds)
        count_rouge(results, trefs, tpreds, lang='en')

# 日本語

# Python: 正規表現による簡易版形態素解析
# https://qiita.com/kinoshita_yuri/items/e15f143981f1616994ed


pJA = re.compile(r"/|[A-Z]+|[a-z]+|[ァ-ンー]+|[ぁ-ん-]+|[ァ-ヶ]+|[一-龍]+|[。、]|/")


def tokenize_ja(text):
    text_m = []
    m = pJA.findall(text)
    for row in m:
        if re.compile(r'^[あ-ん]+$').fullmatch(row):
            if row[0] in 'はがのにへともでを':
                prefix = row[0]
                token = row[1:]
                text_m.append(prefix)
                if (len(token) > 0):
                    text_m.append(token)
            elif row[-2:] in 'のでからまで':
                token = row[0:-2]
                suffix = row[-2:]
                text_m.append(token)
                text_m.append(suffix)
            elif row[-1:] in 'もはがでを':
                token = row[0:-1]
                suffix = row[-1:]
                text_m.append(token)
                text_m.append(suffix)
            else:
                text_m.append(row)
        else:
            text_m.append(row)
    return text_m


def eval_ja(results, refs, preds):
    for ref, pred in zip(refs, preds):
        count_char(results, ref, pred)
        trefs = tokenize_ja(ref)
        tpreds = tokenize_ja(pred)
        count_blue(results, trefs, tpreds)
        count_rouge(results, trefs, tpreds, lang='ja')
