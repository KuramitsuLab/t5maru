import json
from .metrics import calc_score


def setup():
    import argparse
    parser = argparse.ArgumentParser(description='t5score script')
    parser.add_argument('--score', type=str, default='auto')
    hparams = parser.parse_args()  # hparams になる
    return hparams


def read_jsonl(filename):
    refs = []
    preds = []
    with open(filename) as f:
        for line in f.readlines():
            data = json.loads(line)
            refs.append(data['out'])
            preds.append(data['pred'])
    return refs, preds


def main():
    hparams = setup()
    for file in hparams.files:
        refs, preds = read_jsonl(file)
        outfile = file.replace('.jsonl', '.csv')
        calc_score(refs, preds, outfile, hparams.score, file)


if __name__ == '__main__':
    main()
