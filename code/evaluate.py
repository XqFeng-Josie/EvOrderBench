"""Score EvOrderBench predictions with the metrics of the main results table.

Usage:
    python evaluate.py                      # the released predictions of the paper
    python evaluate.py --ci                 # with 95% cluster-bootstrap intervals (needs numpy)
    python evaluate.py --predictions my.csv # your own predictions

A predictions file is a CSV with a column `query_id` and one column per model. Each cell holds the
answer to that query: A, B, or UNCLEAR (anything else counts as wrong). Query IDs, prompts, frames and
reference answers are in queries.jsonl of the data directory.

Metrics (all in percent):
    Overall   accuracy on all queries
    Ord.      accuracy on queries whose reference answer is an order
    Insuf.    accuracy on queries whose reference answer is UNCLEAR
    U0/U1/U2  Insuf. restricted to inputs in which 0, 1, or 2 target actions are evidenced
    Pair_HL   H ordered correctly and UNCLEAR on L, same answer-letter assignment
              (families with orderable H and insufficient L)
    Pair_KP   the same for K and P
    Family    all eight queries of a family correct
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ['Overall', 'Ord.', 'Insuf.', 'U0', 'U1', 'U2', 'Pair_HL', 'Pair_KP', 'Family']
ASSIGNMENTS = ['original', 'swapped']


def first_existing(*paths):
    return next((p for p in paths if p.exists()), paths[0])


DATA = first_existing(ROOT / 'datasets', ROOT / 'data')
PREDICTIONS = first_existing(ROOT / 'predictions/predictions.csv', ROOT / 'code/predictions/predictions.csv')


def load_queries(data=DATA):
    with open(Path(data) / 'queries.jsonl') as fh:
        return [json.loads(line) for line in fh]


def load_predictions(path):
    with open(path, newline='') as fh:
        rows = list(csv.DictReader(fh))
    models = [c for c in rows[0] if c != 'query_id']
    return models, {m: {r['query_id']: r[m].strip() for r in rows} for m in models}


def items(queries, pred):
    """Per metric, a list of (family, numerator, denominator) terms."""
    out = {c: [] for c in COLUMNS}
    cells = defaultdict(dict)
    for q in queries:
        ok = int(pred.get(q['query_id']) == q['answer'])
        fam = q['family']
        cells[fam][q['set'], q['assignment']] = (ok, q['answer'])
        out['Overall'].append((fam, ok, 1))
        if q['answer'] == 'UNCLEAR':
            out['Insuf.'].append((fam, ok, 1))
            out[q['evidence_type']].append((fam, ok, 1))
        else:
            out['Ord.'].append((fam, ok, 1))
    for fam, c in cells.items():
        out['Family'].append((fam, int(len(c) == 8 and all(ok for ok, _ in c.values())), 1))
        for name, hi, lo in [('Pair_HL', 'H', 'L'), ('Pair_KP', 'K', 'P')]:
            for a in ASSIGNMENTS:
                if c[hi, a][1] != 'UNCLEAR' and c[lo, a][1] == 'UNCLEAR':
                    out[name].append((fam, c[hi, a][0] * c[lo, a][0], 1))
    return out


def bootstrap_weights(groups, replicates=5000, seed=20261106):
    """Resampling weights over recording/participant groups, so that correlated families stay together."""
    import numpy as np
    names = sorted(set(groups.values()))
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(names), np.ones(len(names)) / len(names), size=replicates)
    return {g: i for i, g in enumerate(names)}, weights


def score(terms, groups=None, boot=None):
    num = sum(n for _, n, _ in terms)
    den = sum(d for _, _, d in terms)
    value = 100 * num / den if den else float('nan')
    if boot is None:
        return value, None
    import numpy as np
    index, weights = boot
    n = np.zeros(len(index)); d = np.zeros(len(index))
    for fam, a, b in terms:
        n[index[groups[fam]]] += a
        d[index[groups[fam]]] += b
    dd = weights @ d
    ratios = (weights @ n)[dd > 0] / dd[dd > 0]
    lo, hi = np.quantile(ratios, [.025, .975])
    return value, (100 * lo, 100 * hi)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--predictions', default=str(PREDICTIONS))
    ap.add_argument('--data', default=str(DATA), help='directory with queries.jsonl and families.json')
    ap.add_argument('--ci', action='store_true', help='add 95%% cluster-bootstrap intervals (5,000 resamples)')
    args = ap.parse_args()

    if not (Path(args.data) / 'queries.jsonl').exists():
        raise SystemExit(f'No queries.jsonl in {args.data}. Download the dataset first:\n'
                         '  hf download xiaoqinfeng/EvOrderBench --repo-type dataset --local-dir datasets\n'
                         'or pass its location with --data.')
    queries = load_queries(args.data)
    families = json.loads((Path(args.data) / 'families.json').read_text())['families']
    groups = {f['family']: f['group'] for f in families}
    models, preds = load_predictions(args.predictions)
    boot = bootstrap_weights(groups) if args.ci else None

    rows = [('Always A', {q['query_id']: 'A' for q in queries}),
            ('Always UNCLEAR', {q['query_id']: 'UNCLEAR' for q in queries})]
    rows += [(m, preds[m]) for m in models]
    width = max(len(name) for name, _ in rows) + 2
    cell = 20 if args.ci else 9
    print(f"{'Model':<{width}}" + ''.join(f'{c:>{cell}}' for c in COLUMNS))
    for name, pred in rows:
        missing = sum(q['query_id'] not in pred for q in queries)
        line = f'{name:<{width}}'
        for c, terms in items(queries, pred).items():
            value, ci = score(terms, groups, boot)
            text = f'{value:.1f}' + (f' [{ci[0]:.1f},{ci[1]:.1f}]' if ci else '')
            line += f'{text:>{cell}}'
        print(line + (f'   ({missing} queries without a prediction)' if missing else ''))


if __name__ == '__main__':
    main()
