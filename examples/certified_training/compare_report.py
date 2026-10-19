"""Summarize run_compare.sh output: time, loss/metric similarity, and bound similarity.

    python compare_report.py OUT_DIR NET [NET ...]
"""
import json
import os
import sys

import torch


def load(out, net, tool):
    with open(os.path.join(out, net, tool, 'record.json')) as f:
        return json.load(f)


def train_ce_pairs(rec):
    """Per training batch: the robust CE (the loss) once eps > 0, else the regular CE."""
    losses = []
    for r in rec['records']:
        if r['train']:
            losses.extend(r['ce'])
    return losses


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-12)


def main():
    out, nets = sys.argv[1], sys.argv[2:]
    print('# auto_LiRPA vs cf-gpu, simple_training.py\n')
    rec0 = load(out, nets[0], 'lirpa')
    keep = ('bound_type', 'eps', 'num_epochs', 'batch_size', 'lr', 'scheduler_name', 'scheduler_opts', 'seed')
    print('Settings: ' + ', '.join(f'{k}={rec0["args"][k]}' for k in keep) + '\n')

    print('## Time (seconds): total = setup + train + eval; compile and warmup not included\n')
    print('| net | lirpa setup | cf setup | lirpa train | cf train | lirpa eval | cf eval | lirpa total | cf total | cf/lirpa | cf compile | lirpa warmup | cf warmup |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for net in nets:
        t = {}
        for tool in ('lirpa', 'cf'):
            rec = load(out, net, tool)
            setup = rec.get('setup_seconds') or 0.0
            tr = sum(r['seconds'] for r in rec['records'] if r['train'])
            ev = sum(r['seconds'] for r in rec['records'] if not r['train'])
            t[tool] = dict(setup=setup, train=tr, eval=ev, total=setup + tr + ev, warmup=rec['warmup_seconds'],
                           compile=rec.get('compile_seconds'))
        l, c = t['lirpa'], t['cf']
        compile_s = 'n/a' if c['compile'] is None else f"{c['compile']:.1f}"
        print(f"| {net} | {l['setup']:.2f} | {c['setup']:.2f} | {l['train']:.2f} | {c['train']:.2f} | {l['eval']:.2f} | {c['eval']:.2f} | "
              f"{l['total']:.2f} | {c['total']:.2f} | {c['total'] / l['total']:.3f} | {compile_s} | {l['warmup']:.2f} | {c['warmup']:.2f} |")

    print('\n## Per-epoch time (seconds, train + eval)\n')
    for net in nets:
        per = {tool: [r['seconds'] for r in load(out, net, tool)['records']] for tool in ('lirpa', 'cf')}
        n = len(per['lirpa']) // 2
        lt = [per['lirpa'][2 * i] + per['lirpa'][2 * i + 1] for i in range(n)]
        ct = [per['cf'][2 * i] + per['cf'][2 * i + 1] for i in range(n)]
        print(f'- {net}: lirpa ' + ' '.join(f'{x:.2f}' for x in lt) + ' | cf ' + ' '.join(f'{x:.2f}' for x in ct))

    print('\n## Loss trajectory (same seed and data order; per-batch CE values in training)\n')
    print('| net | batches | max rel diff, first 100 | median rel diff, all | max rel diff, all | final-epoch train loss lirpa / cf |')
    print('|---|---|---|---|---|---|')
    for net in nets:
        a, b = train_ce_pairs(load(out, net, 'cf')), train_ce_pairs(load(out, net, 'lirpa'))
        n = min(len(a), len(b))
        d = sorted(rel(a[i], b[i]) for i in range(n))
        first = max(rel(a[i], b[i]) for i in range(min(n, 100)))
        lt = [r for r in load(out, net, 'lirpa')['records'] if r['train']][-1]['avg']['Loss']
        ct = [r for r in load(out, net, 'cf')['records'] if r['train']][-1]['avg']['Loss']
        print(f'| {net} | {n}{"" if len(a) == len(b) else " (len differs!)"} | {first:.2e} | {d[n // 2]:.2e} | {d[-1]:.2e} | {lt:.4f} / {ct:.4f} |')

    print('\n## Final-epoch metrics (meter averages)\n')
    print('| net | split | tool | Loss | Robust_CE | Err | Verified_Err |')
    print('|---|---|---|---|---|---|---|')
    for net in nets:
        for split in ('train', 'eval'):
            for tool in ('lirpa', 'cf'):
                recs = [r for r in load(out, net, tool)['records'] if r['train'] == (split == 'train')]
                m = recs[-1]['avg']
                print(f'| {net} | {split} | {tool} | {m.get("Loss", float("nan")):.4f} | {m.get("Robust_CE", float("nan")):.4f} | '
                      f'{m.get("Err", float("nan")):.4f} | {m.get("Verified_Err", float("nan")):.4f} |')

    print('\n## Bounds on the same weights (256 test images, eps = training eps)\n')
    print('Lower bounds on the 9 margins and d(sum lb)/dθ, both tools on each final checkpoint.\n')
    print('| net | checkpoint | method | max abs lb diff | max rel lb diff | worst param grad rel diff |')
    print('|---|---|---|---|---|---|')
    for net in nets:
        for ckpt in ('lirpa', 'cf'):
            d = os.path.join(out, net, f'bounds_{ckpt}')
            try:
                lb_l = torch.load(os.path.join(d, 'bounds_ckpt_lirpa.pt'))
                lb_c = torch.load(os.path.join(d, 'bounds_ckpt_cf.pt'))
            except FileNotFoundError as e:
                print(f'| {net} | {ckpt} | missing: {os.path.basename(e.filename)} | | | |')
                continue
            for method in ('IBP', 'CROWN-IBP'):
                (l, gl), (c, gc) = lb_l[method], lb_c[method]
                g = max(((gc[k] - gl[k]).norm() / gl[k].norm().clamp_min(1e-12)).item() for k in gl)
                print(f'| {net} | {ckpt} | {method} | {(c - l).abs().max().item():.2e} | '
                      f'{((c - l).abs() / l.abs().clamp_min(1e-6)).max().item():.2e} | {g:.2e} |')


if __name__ == '__main__':
    main()
