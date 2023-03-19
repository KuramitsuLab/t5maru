import json
from .metrics import eval_score, write_score_csv


def setup():
    import argparse
    parser = argparse.ArgumentParser(description='t5score script')
    parser.add_argument('files', type=str, nargs='+', help='jsonl files')
    parser.add_argument('--pred', type=str, default='pred')
    parser.add_argument('--ref', type=str, default='out')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--score', type=str, default='auto')
    parser.add_argument('--desc', type=str, default='')
    parser.add_argument('--top_k', type=int, default=0)
    hparams = parser.parse_args()  # hparams になる
    return hparams


def read_jsonl(filename, key_ref='out', key_pred='pred'):
    refs = []
    preds = []
    with open(filename) as f:
        for line in f.readlines():
            data = json.loads(line)
            refs.append(data[key_ref])
            preds.append(data[key_pred])
    return refs, preds


def main():
    hparams = setup()
    for file in hparams.files:
        refs, preds = read_jsonl(file, hparams.ref, hparams.pred)
        results = {
            'Model': hparams.model_path,
            'Tested': file,
            'Desc': hparams.desc,
            'Count': len(refs),
        }
        eval_score(results, refs, preds, hparams.score, file)
        if hparams.output is None:
            hparams.output = file.replace('.jsonl', '.csv').replace('.gz', '')
        write_score_csv(results, hparams.output)


if __name__ == '__main__':
    main()
