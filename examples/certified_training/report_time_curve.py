"""Summarize run_time_curve.sh: accuracy over timed training time, auto_LiRPA vs cf-gpu.

    python report_time_curve.py OUT_DIR [BUDGET_S]   (BUDGET_S: cut both runs at that many timed seconds)

Writes OUT_DIR/time_curve.csv (one row per checkpoint: tool, epoch, timed seconds so far, clean and verified accuracy)
and OUT_DIR/time_curve.png. Timed seconds follow compare_driver.py: setup plus every train and test pass up to and
including that epoch (code generation and warmup are not timed).
"""
import csv
import json
import os
import sys

TOOLS = {'lirpa': 'auto_LiRPA', 'cf': 'cf-gpu'}


def timed_seconds(record):
    """{epoch: setup + all train/test passes through that epoch}."""
    total, by_epoch = record['setup_seconds'], {0: record['setup_seconds']}
    for r in record['records']:
        total += r['seconds']
        by_epoch[r['epoch']] = total
    return by_epoch


def main(out, cap=None):
    rows, budget = [], 0
    for tool, label in TOOLS.items():
        with open(os.path.join(out, tool, 'record.json')) as f:
            record = json.load(f)
        seconds = timed_seconds(record)
        budget = record.get('time_budget', budget)
        with open(os.path.join(out, tool, 'precision_curve_cf.json')) as f:
            curve = json.load(f)
        for c in sorted(curve, key=lambda c: c['epoch']):
            rows.append({'tool': label, 'epoch': c['epoch'], 'timed_s': round(seconds[c['epoch']], 2),
                         'clean': round(c['clean'], 4), **{f'verified_{m}': round(r['precision'], 4)
                                                          for m, r in c['methods'].items()}})
    #FEAT: a shorter budget read off the same runs (valid when the schedule doesn't depend on the run's length);
    #      its files get a _<budget>s suffix.
    suffix = ''
    if cap:
        budget, suffix = cap, f'_{cap:g}s'
        rows = [r for r in rows if r['timed_s'] <= cap]
    with open(os.path.join(out, f'time_curve{suffix}.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    final = {t: max((r for r in rows if r['tool'] == t), key=lambda r: r['epoch']) for t in TOOLS.values()}
    for t, r in final.items():
        print(f"{t:10} epoch {r['epoch']:3}  timed {r['timed_s'] / 60:6.2f} min  clean {r['clean']:.4f}  "
              + '  '.join(f"{k} {v:.4f}" for k, v in r.items() if k.startswith('verified_')))
    a, c = final['auto_LiRPA']['timed_s'], final['cf-gpu']['timed_s']
    print(f"speedup over the whole recipe: {a / c:.2f}x")
    #FEAT: --time_budget runs: each tool's model at the budget is its last checkpoint within it.
    if budget:
        print(f'model at the {budget:.0f} s budget (last checkpoint within it):')
        for t in TOOLS.values():
            r = max((r for r in rows if r['tool'] == t and r['timed_s'] <= budget), key=lambda r: r['epoch'])
            print(f"  {t:10} epoch {r['epoch']:3}  timed {r['timed_s']:6.1f} s  clean {r['clean']:.4f}  "
                  + '  '.join(f"{k} {v:.4f}" for k, v in r.items() if k.startswith('verified_')))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    #FEAT: accuracy per epoch; with --time_budget each tool's curve ends where its budget ran out.
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for t, color in zip(TOOLS.values(), ('tab:blue', 'tab:orange')):
        pts = [r for r in rows if r['tool'] == t and (not budget or r['timed_s'] <= budget)]
        for k in [k for k in pts[0] if k.startswith('verified_')]:
            ax.plot([r['epoch'] for r in pts], [r[k] for r in pts], '-', color=color, label=f"{t}, verified ({k[9:]})")
            ax.plot(pts[-1]['epoch'], pts[-1][k], 'o', color=color)
            ax.annotate(f"{pts[-1][k]:.1%} @ epoch {pts[-1]['epoch']}", (pts[-1]['epoch'], pts[-1][k]),
                        textcoords='offset points', xytext=(-5, 6), ha='right', fontsize=7, color=color)
        ax.plot([r['epoch'] for r in pts], [r['clean'] for r in pts], '--', color=color, alpha=0.6, label=f'{t}, clean')
    ax.set_xlabel('epoch')
    ax.set_ylabel('accuracy')
    ax.set_ylim(0, 1)
    if budget:
        ax.set_title(f'same {budget:.0f} s of timed training per tool', fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f'epoch_curve{suffix}.png'), dpi=150)

    fig, ax = plt.subplots(figsize=(5, 3.2))
    for t, color in zip(TOOLS.values(), ('tab:blue', 'tab:orange')):
        pts = [r for r in rows if r['tool'] == t]
        x = [r['timed_s'] for r in pts]
        for k in [k for k in pts[0] if k.startswith('verified_')]:
            ax.plot(x, [r[k] for r in pts], '-o', ms=3, color=color, label=f"{t}, verified ({k[9:]})")
        ax.plot(x, [r['clean'] for r in pts], '--', color=color, alpha=0.6, label=f'{t}, clean')
    if budget:
        ax.axvline(budget, color='gray', ls=':', label=f'budget {budget:.0f} s')
    ax.set_xlabel('timed training (s)')
    ax.set_ylabel('accuracy')
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f'time_curve{suffix}.png'), dpi=150)

    #FEAT: the gap over time: verified accuracy only, shaded between the tools, labeled at the budget.
    import numpy as np
    key = next(k for k in rows[0] if k.startswith('verified_'))
    end = budget or min(max(r['timed_s'] for r in rows if r['tool'] == t) for t in TOOLS.values())
    curves = {t: sorted(((r['timed_s'], r[key]) for r in rows if r['tool'] == t and r['timed_s'] <= end))
              for t in TOOLS.values()}
    grid = np.linspace(0, end, 400)
    interp = {t: np.interp(grid, [x for x, _ in c], [y for _, y in c]) for t, c in curves.items()}
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    colors = {'auto_LiRPA': 'tab:blue', 'cf-gpu': 'tab:orange'}
    ax.fill_between(grid, interp['auto_LiRPA'], interp['cf-gpu'], where=interp['cf-gpu'] >= interp['auto_LiRPA'],
                    color='tab:orange', alpha=0.15, lw=0)
    for t, cv in curves.items():
        ax.plot([x for x, _ in cv], [y for _, y in cv], '-', lw=2, color=colors[t], label=t)
        ax.plot(cv[-1][0], cv[-1][1], 'o', color=colors[t])
        ax.annotate(f'{cv[-1][1]:.1%}', cv[-1], textcoords='offset points', xytext=(-4, 6 if t == 'cf-gpu' else -12),
                    ha='right', fontsize=8, color=colors[t], fontweight='bold')
    ax.set_xlim(0, end * 1.02)
    ax.set_ylim(0, max(c[-1][1] for c in curves.values()) * 1.3)
    ax.set_xlabel('timed training (s)')
    ax.set_ylabel(f'verified accuracy ({key[9:]})')
    if budget:
        ax.set_title(f'same {budget:.0f} s of timed training per tool', fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(out, f'time_curve_gap{suffix}.png'), dpi=150)

    #FEAT: the two claims apart: accuracy per epoch (same model) and time per epoch (less time).
    fig, (left, right) = plt.subplots(1, 2, figsize=(8, 3.2))
    for t, color in zip(TOOLS.values(), ('tab:blue', 'tab:orange')):
        pts = [r for r in rows if r['tool'] == t]
        ep = [r['epoch'] for r in pts]
        for k in [k for k in pts[0] if k.startswith('verified_')]:
            left.plot(ep, [r[k] for r in pts], '-o', ms=3, color=color, label=f'{t}, verified ({k[9:]})')
        left.plot(ep, [r['clean'] for r in pts], '--', color=color, alpha=0.6, label=f'{t}, clean')
        right.plot(ep, [r['timed_s'] for r in pts], '-', color=color, label=f"{t}: {pts[-1]['timed_s']:.0f} s")
    left.set(xlabel='epoch', ylabel='accuracy', ylim=(0, 1), title='same model')
    right.set(xlabel='epoch', ylabel='timed training (s)', title=f'{a / c:.2f}x less time')
    for ax in (left, right):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f'time_curve_panels{suffix}.png'), dpi=150)

    #FEAT: time to reach each verified-accuracy level (linear between the two checkpoints around the crossing).
    key = next(k for k in rows[0] if k.startswith('verified_'))
    levels = [0.2, 0.3, 0.4]
    groups, times = [], {t: [] for t in TOOLS.values()}
    for level in levels:
        reached = {t: time_to(level, [r for r in rows if r['tool'] == t], key) for t in TOOLS.values()}
        if None not in reached.values():
            groups.append(f'{level:.0%} verified')
            for t in TOOLS.values():
                times[t].append(reached[t])
    groups.append('full recipe')
    for t in TOOLS.values():
        times[t].append(final[t]['timed_s'])
    fig, ax = plt.subplots(figsize=(5, 3.2))
    width = 0.38
    for i, (t, color) in enumerate(zip(TOOLS.values(), ('tab:blue', 'tab:orange'))):
        ax.bar([g + (i - 0.5) * width for g in range(len(groups))], times[t], width, color=color, label=t)
    for g in range(len(groups)):
        a_t, c_t = times['auto_LiRPA'][g], times['cf-gpu'][g]
        ax.text(g, max(a_t, c_t) * 1.02, f'{a_t / c_t:.2f}x', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(range(len(groups)), groups)
    ax.set_ylabel('timed training (s)')
    ax.set_ylim(0, max(times['auto_LiRPA'] + times['cf-gpu']) * 1.15)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f'time_to_accuracy{suffix}.png'), dpi=150)
    for g, name in enumerate(groups):
        print(f"{name:14} auto_LiRPA {times['auto_LiRPA'][g]:7.1f} s  cf-gpu {times['cf-gpu'][g]:7.1f} s  "
              f"{times['auto_LiRPA'][g] / times['cf-gpu'][g]:.2f}x")
    print('wrote', os.path.join(out, 'time_curve.csv'), ', time_curve.png, time_curve_panels.png and time_to_accuracy.png')


def time_to(level, pts, key):
    """Timed seconds at which verified accuracy first reaches level, or None."""
    pts = sorted(pts, key=lambda r: r['epoch'])
    for prev, cur in zip(pts, pts[1:]):
        if cur[key] >= level:
            if prev[key] >= level:
                return prev['timed_s']
            f = (level - prev[key]) / (cur[key] - prev[key])
            return prev['timed_s'] + f * (cur['timed_s'] - prev['timed_s'])
    return None


if __name__ == '__main__':
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else None)
