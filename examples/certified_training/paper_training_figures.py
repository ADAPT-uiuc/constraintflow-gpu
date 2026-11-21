"""Paper figures from two full run_time_curve.sh runs (IBP and CROWN-FAST), auto_LiRPA vs Scepter (cf-gpu),
in the paper's figure theme (paper/scripts/fig8_theme.py).

    python paper_training_figures.py IBP_DIR CROWN_FAST_DIR OUT_DIR [SHARES]

training_time.pdf/.png: training time Scepter saves over auto_LiRPA for all epochs, per certifier.
training_budget_<share%>.pdf/.png: verified accuracy over timed training, one panel per certifier, cut at each of SHARES
(comma-separated, default 0.25,0.3,0.5,0.75) of auto_LiRPA's own full training time; each tool's value is its last
checkpoint within it.
"""
import csv
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# the paper's theme: fonts, axis lines, tool colours
sys.path.insert(0, '/home/db50/cf-harness/paper/scripts')
import fig8_theme as F8  # noqa: E402

TOOLS = {'auto_LiRPA': 'auto_LiRPA', 'cf-gpu': 'Scepter'}  # run label -> paper name
COLOR = {t: F8.SYSTEM[name][0] for t, name in TOOLS.items()}
plt.rcParams.update({'font.size': 8})


def load(run):
    rows = list(csv.DictReader(open(os.path.join(run, 'time_curve.csv'))))
    key = next(k for k in rows[0] if k.startswith('verified_'))
    curves = {t: sorted((float(r['timed_s']), float(r[key]), int(r['epoch'])) for r in rows if r['tool'] == t)
              for t in TOOLS}
    return key[len('verified_'):], curves


def save(fig, out, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out, f'{name}.{ext}'), dpi=300, bbox_inches='tight', pad_inches=0.02)


def time_bars(runs, out):
    """One bar per certifier: the training time Scepter saves over auto_LiRPA for all epochs (and its share)."""
    fig, ax = plt.subplots(figsize=(3.0, 2.0))
    names = list(runs)
    saved = []
    for i, name in enumerate(names):
        ends = {t: runs[name][1][t][-1][0] for t in TOOLS}
        saved.append(ends['auto_LiRPA'] - ends['cf-gpu'])
        ax.bar(i, saved[-1], 0.5, color=COLOR['cf-gpu'], lw=0)
        ax.text(i, saved[-1], f"{saved[-1]:.0f} s ({100 * saved[-1] / ends['auto_LiRPA']:.0f}%)",
                ha='center', va='bottom', fontsize=7)
    ax.set_xticks(range(len(names)), names)
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.set_ylabel('training time saved (s)')
    ax.set_ylim(0, max(saved) * 1.18)
    F8.style(ax, bottom=True)
    save(fig, out, 'training_time')


def budget_panels(runs, out, share):
    fig, axes = plt.subplots(1, len(runs), figsize=(6.8, 2.3))
    for ax, (name, (method, curves)) in zip(np.atleast_1d(axes), runs.items()):
        budget = share * curves['auto_LiRPA'][-1][0]
        cut = {t: [p for p in c if p[0] <= budget] for t, c in curves.items()}
        for t, c in cut.items():
            ax.plot([p[0] for p in c], [100 * p[1] for p in c], '-', lw=1.0, color=COLOR[t], label=TOOLS[t])
            ax.plot(c[-1][0], 100 * c[-1][1], 'o', ms=3, color=COLOR[t])
            early = c[-1][0] < 0.95 * budget  # finished all epochs before the budget: label above the dot
            ax.annotate(f'{100 * c[-1][1]:.1f}%', (c[-1][0], 100 * c[-1][1]), textcoords='offset points',
                        xytext=(0, 4) if early else (4, 0), ha='center' if early else 'left',
                        va='bottom' if early else 'center', fontsize=7, color=COLOR[t])
        ax.set_xlim(0, budget * 1.15)
        ax.set_ylim(0, 100 * max(c[-1][1] for c in cut.values()) * 1.25)
        ax.set_title(f'{name} ({budget:.0f} s)', fontsize=8)
        ax.set_xlabel('training time (s)')
        ax.set_ylabel(f'verified accuracy, {method} (%)')
        F8.style(ax, bottom=True)
        ax.legend(frameon=False, fontsize=7, loc='upper left')
    fig.tight_layout(w_pad=2)
    save(fig, out, f'training_budget_{round(100 * share)}')


def at_budget(curve, budget):
    """Verified accuracy of the last checkpoint within budget (0 before the first one)."""
    return max((p for p in curve if p[0] <= budget), default=(0, 0.0, 0))[1]


def gap_curve(runs, out):
    """#FEAT: Scepter minus auto_LiRPA verified accuracy in the time auto_LiRPA takes for each epoch."""
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    for (name, (method, curves)), ls in zip(runs.items(), ('-', '--')):
        epochs = [e for _, _, e in curves['auto_LiRPA']]
        gap = [100 * (at_budget(curves['cf-gpu'], t) - v) for t, v, _ in curves['auto_LiRPA']]
        ax.plot(epochs, gap, ls, lw=1.0, color='black', label=name)
        i = int(np.argmax(gap))
        print(f'{name}: peak gap {gap[i]:.1f} points at auto_LiRPA epoch {epochs[i]}, min {min(gap):.1f}, '
              f'mean {np.mean(gap):.1f}')
        # same epoch, both tools: the recipe's models should match
        by_ep = {t: {e: v for _, v, e in c} for t, c in curves.items()}
        d = [100 * (by_ep['cf-gpu'][e] - by_ep['auto_LiRPA'][e]) for e in by_ep['auto_LiRPA'] if e in by_ep['cf-gpu']]
        print(f'  same-epoch difference: mean {np.mean(d):+.2f}, max |.| {max(map(abs, d)):.2f} points')
    ax.axhline(0, color='0.6', lw=0.6)
    ax.set_xlim(0, max(e for _, c in runs.values() for _, _, e in c['auto_LiRPA']))
    ax.set_xlabel('epochs trained by auto_LiRPA')
    ax.set_ylabel('verified accuracy gain (points)')
    F8.style(ax, bottom=True)
    ax.legend(frameon=False, fontsize=7, loc='upper right')
    save(fig, out, 'training_gap')


def main(ibp, crown_fast, out, shares='0.25,0.3,0.5,0.75'):
    os.makedirs(out, exist_ok=True)
    runs = {'IBP': load(ibp), 'CROWN-FAST': load(crown_fast)}
    time_bars(runs, out)
    gap_curve(runs, out)
    for share in map(float, shares.split(',')):
        budget_panels(runs, out, share)
        print('wrote', os.path.join(out, f'training_budget_{round(100 * share)}.pdf'), '(+ .png)')


if __name__ == '__main__':
    main(*sys.argv[1:5])
