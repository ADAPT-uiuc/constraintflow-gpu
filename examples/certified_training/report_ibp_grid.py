"""Summarize run_ibp_grid.sh: IBP precision before/after, training time, and bound/gradient agreement.

    python report_ibp_grid.py OUT_DIR
"""
import glob
import json
import os
import sys

import torch


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def precision(path):
    result = read_json(path)
    return result['methods']['IBP']['precision'] if result else None


def max_rel_diffs(folder, name):
    """Largest relative difference of IBP lower bounds and of any parameter gradient, lirpa vs cf."""
    try:
        a = torch.load(os.path.join(folder, f'bounds_{name}_lirpa.pt'))['IBP']
        b = torch.load(os.path.join(folder, f'bounds_{name}_cf.pt'))['IBP']
    except (OSError, KeyError):
        return None, None
    rel = lambda x, y: ((x - y).abs().max() / (x.abs().max() + 1e-12)).item()
    return rel(a[0], b[0]), max(rel(a[1][k], b[1][k]) for k in a[1])


def timing(record):
    """setup + training epochs (the timed total), plus the per-epoch evaluation, compile and warmup."""
    train = sum(r['seconds'] for r in record['records'] if r['train'])
    test = sum(r['seconds'] for r in record['records'] if not r['train'])
    last = [r for r in record['records'] if r['train']][-1]['avg']
    return {'total': record.get('setup_seconds', 0) + train, 'setup': record.get('setup_seconds', 0),
            'train': train, 'eval': test, 'compile': record.get('compile_seconds'),
            'warmup': record.get('warmup_seconds'), 'loss': last.get('Loss'), 'verified_err': last.get('Verified_Err')}


def fmt(x, spec='.3f'):
    return '-' if x is None else format(x, spec)


def main():
    out = sys.argv[1]
    for d in sorted(glob.glob(os.path.join(out, '*', 'eps_*'))):
        net, eps = d.split(os.sep)[-2], d.split('eps_')[-1]
        print(f'\n== {net}  eps={float(eps):.5g}')
        for tool in ('lirpa', 'cf'):
            rec = read_json(os.path.join(d, tool, 'record.json'))
            t = timing(rec) if rec else {}
            print(f'  {tool:6s} precision before {fmt(precision(os.path.join(d, "before", f"precision_onnx_{tool}.json")))}'
                  f'  after {fmt(precision(os.path.join(d, tool, f"precision_ckpt_{tool}.json")))}'
                  f' | time {fmt(t.get("total"), ".1f")} s (setup {fmt(t.get("setup"), ".2f")}, train {fmt(t.get("train"), ".1f")};'
                  f' eval {fmt(t.get("eval"), ".1f")}, compile {fmt(t.get("compile"), ".0f")}, warmup {fmt(t.get("warmup"), ".1f")})'
                  f' | last epoch loss {fmt(t.get("loss"), ".4f")} verified err {fmt(t.get("verified_err"), ".4f")}')
        #FEAT: cross-evaluation (cross_eval_ibp_grid.sh): each tool's trained weights, precision by both tools
        for trained in ('lirpa', 'cf'):
            by = {tool: precision(os.path.join(d, trained, f'precision_ckpt_{tool}.json')) for tool in ('lirpa', 'cf')}
            print(f'  {trained}-trained weights: precision by lirpa {fmt(by["lirpa"])}, by cf {fmt(by["cf"])}')
        lirpa, cf = (read_json(os.path.join(d, tool, 'record.json')) for tool in ('lirpa', 'cf'))
        if lirpa and cf:
            print(f'  speedup (setup + training): {timing(lirpa)["total"] / timing(cf)["total"]:.2f}x')
        for label, folder, name in (('before', 'before', 'onnx'), ('after, lirpa-trained', 'after_lirpa', 'ckpt'),
                                    ('after, cf-trained', 'after_cf', 'ckpt')):
            lb, grad = max_rel_diffs(os.path.join(d, folder), name)
            print(f'  bounds/grads {label:21s} max rel diff: lb {fmt(lb, ".1e")}  grad {fmt(grad, ".1e")}')


if __name__ == '__main__':
    main()
