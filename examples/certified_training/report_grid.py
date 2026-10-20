"""Summarize run_grid.sh into OUT_DIR/report.md, results.csv, curves.csv and plots/.

    python report_grid.py OUT_DIR
"""
import csv
import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

CERTIFIERS = ['IBP', 'CROWN-FAST', 'zid', 'CROWN']
TOOLS = ['lirpa', 'cf']
EVALUATORS = ['IBP', 'CROWN-IBP', 'CROWN']
COLORS = {'IBP': '#2a78d6', 'CROWN-FAST': '#eb6834', 'zid': '#1baf7a', 'CROWN': '#eda100'}  # dataviz reference palette, slots 1-4
MARKERS = {'IBP': 'o', 'CROWN-FAST': 's', 'zid': '^', 'CROWN': 'D'}
STYLES = {'cf': '-', 'lirpa': '--'}
TOOL_NAMES = {'cf': 'cf-gpu', 'lirpa': 'auto_LiRPA'}


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def fmt(x, spec='.3f'):
    return '-' if x is None else format(x, spec)


def scores(result):
    """Clean accuracy and verified precision per evaluator of one precision result."""
    if not result:
        return {}
    return {'clean': result.get('clean'), **{m: r['precision'] for m, r in result['methods'].items()}}


def timing(record):
    train = [r for r in record['records'] if r['train']]
    seconds = sum(r['seconds'] for r in train)
    return {'train': seconds, 'setup': record.get('setup_seconds', 0), 'total': record.get('setup_seconds', 0) + seconds,
            'per_epoch': seconds / len(train), 'epochs': len(train), 'compile': record.get('compile_seconds'),
            'eval': sum(r['seconds'] for r in record['records'] if not r['train'])}


def max_rel_diffs(folder, name, method):
    """Largest relative difference of the lower bounds and of any parameter gradient, auto_LiRPA vs cf-gpu."""
    try:
        a = torch.load(os.path.join(folder, f'bounds_{name}_lirpa.pt'))[method]
        b = torch.load(os.path.join(folder, f'bounds_{name}_cf.pt'))[method]
    except (OSError, KeyError):
        return None, None
    rel = lambda x, y: ((x - y).abs().max() / (x.abs().max() + 1e-12)).item()
    return rel(a[0], b[0]), max(rel(a[1][k], b[1][k]) for k in a[1])


def load_net(d):
    """Everything run_grid.sh wrote for one net."""
    net = {'before': {t: scores(read_json(os.path.join(d, 'before', f'precision_onnx_{t}.json'))) for t in TOOLS}, 'runs': {}}
    for cert in CERTIFIERS:
        for tool in TOOLS:
            run = os.path.join(d, cert, tool)
            record = read_json(os.path.join(run, 'record.json'))
            if not record:
                continue
            curve = read_json(os.path.join(run, 'precision_curve_cf.json')) or []
            net['runs'][cert, tool] = {
                'timing': timing(record),
                'curve': [(0, net['before']['cf'])] + [(r['epoch'], scores(r)) for r in curve],
                'test_loss': [(r['epoch'], r['avg'].get('Loss')) for r in record['records'] if not r['train']],
                'final_by': {t: scores(read_json(os.path.join(run, f'precision_ckpt_{t}.json'))) for t in TOOLS},
            }
    return net


def after(run):
    """Final weights judged by cf-gpu (the last checkpoint of the curve)."""
    return run['curve'][-1][1] if len(run['curve']) > 1 else {}


def net_section(name, d, net):
    lines = [f'## {name}', '']
    lines += ['Every model judged by the same evaluators (cf-gpu, full test set, final eps): '
              'clean accuracy and verified precision, before → after training.', '',
              '| Certifier | Trained by | Clean | IBP-verified | CROWN-IBP-verified | CROWN-verified '
              '| Training time (s) | Setup + training (s) | s / epoch | Speedup (training) |',
              '|---|---|---|---|---|---|---|---|---|---|']
    before = net['before']['cf']
    for cert in CERTIFIERS:
        for tool in TOOLS:
            run = net['runs'].get((cert, tool))
            if not run:
                continue
            a, t = after(run), run['timing']
            other = net['runs'].get((cert, 'lirpa'))
            speedup = f"{other['timing']['train'] / t['train']:.2f}×" if tool == 'cf' and other else ''
            cells = [f"{fmt(before.get(k))} → {fmt(a.get(k))}" for k in ['clean'] + EVALUATORS]
            lines.append(f"| {cert} | {TOOL_NAMES[tool]} | " + ' | '.join(cells)
                         + f" | {t['train']:.1f} | {t['total']:.1f} | {t['per_epoch']:.2f} | {speedup} |")
    lines += ['', 'Bounds and gradients, auto_LiRPA vs cf-gpu (largest relative difference), and the verified '
              'precision each tool gives the same final weights:', '',
              '| Weights | IBP lb | IBP grad | CROWN-IBP lb | CROWN-IBP grad | CROWN lb | CROWN grad '
              '| Precision by auto_LiRPA / cf-gpu (IBP, CROWN-IBP, CROWN) |',
              '|---|---|---|---|---|---|---|---|']
    rows = [('ONNX (before)', os.path.join(d, 'before'), 'onnx', None)]
    rows += [(f'{cert}, trained by {TOOL_NAMES[t]}', os.path.join(d, cert, f'bounds_{t}'), 'ckpt', net['runs'].get((cert, t)))
             for cert in ('IBP', 'CROWN-FAST', 'CROWN') for t in TOOLS]
    for label, folder, weights, run in rows:
        diffs = [fmt(x, '.1e') for m in ('IBP', 'CROWN-IBP', 'CROWN') for x in max_rel_diffs(folder, weights, m)]
        if run:
            by = run['final_by']
            judged = ', '.join(f"{fmt(by['lirpa'].get(m))} / {fmt(by['cf'].get(m))}" for m in EVALUATORS)
        else:
            judged = ', '.join(f"{fmt(net['before']['lirpa'].get(m))} / {fmt(net['before']['cf'].get(m))}" for m in EVALUATORS)
        lines.append(f'| {label} | ' + ' | '.join(diffs) + f' | {judged} |')
    lines += ['', f'![verified precision over training](plots/{name}_verified.png)', '',
              f'![test loss over training](plots/{name}_loss.png)', '']
    return lines


def plot_net(name, net, out):
    os.makedirs(os.path.join(out, 'plots'), exist_ok=True)
    panels = ['clean'] + EVALUATORS
    titles = {'clean': 'Clean accuracy', 'IBP': 'IBP-verified', 'CROWN-IBP': 'CROWN-IBP-verified', 'CROWN': 'CROWN-verified'}
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4.0), sharey=True)
    for ax, key in zip(axes, panels):
        for (cert, tool), run in net['runs'].items():
            pts = [(e, s.get(key)) for e, s in run['curve'] if s.get(key) is not None]
            if pts:
                ax.plot(*zip(*pts), STYLES[tool], color=COLORS[cert], marker=MARKERS[cert], markersize=5, linewidth=2,
                        label=f'{cert} ({TOOL_NAMES[tool]})')
        ax.set_title(titles[key], fontsize=10)
        ax.set_xlabel('epoch')
        ax.grid(alpha=0.25)
    axes[0].set_ylabel('fraction of test set')
    axes[0].set_ylim(0, 1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), fontsize=8, frameon=False)
    fig.suptitle(f'{name}: every checkpoint judged by the same evaluators (cf-gpu)', fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(out, 'plots', f'{name}_verified.png'), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for (cert, tool), run in net['runs'].items():
        pts = [(e, v) for e, v in run['test_loss'] if v is not None]
        if pts:
            ax.plot(*zip(*pts), STYLES[tool], color=COLORS[cert], linewidth=2, label=f'{cert} ({TOOL_NAMES[tool]})')
    ax.set_yscale('log')
    ax.set_xlabel('epoch')
    ax.set_ylabel('robust test loss')
    ax.set_title(f"{name}: test loss (own certifier, scheduled eps)", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'plots', f'{name}_loss.png'), dpi=130)
    plt.close(fig)


def main():
    out = sys.argv[1]
    nets = {os.path.basename(d): d for d in sorted(glob.glob(os.path.join(out, '*'))) if os.path.isdir(os.path.join(d, 'before'))}
    lines = ['# Certified training: certifiers and tools', '',
             'Training time = the training epochs only; setup + training adds TrainingBounds / BoundedModule construction. '
             'Not timed: cf-gpu code generation, warmup, per-epoch test evaluation. '
             'CROWN-FAST = pure CROWN-IBP lower bounds every epoch; CROWN = CROWN lower bounds with CROWN intermediate bounds. '
             'zid has no auto_LiRPA counterpart.', '']
    summary, curves = [], []
    for name, d in nets.items():
        net = load_net(d)
        lines += net_section(name, d, net)
        plot_net(name, net, out)
        for (cert, tool), run in net['runs'].items():
            a, t = after(run), run['timing']
            summary.append({'net': name, 'certifier': cert, 'tool': tool, **{f'before_{k}': net['before']['cf'].get(k) for k in ['clean'] + EVALUATORS},
                            **{f'after_{k}': a.get(k) for k in ['clean'] + EVALUATORS},
                            'train_s': t['train'], 'setup_plus_train_s': t['total'], 's_per_epoch': t['per_epoch'],
                            'compile_s': t['compile'], 'eval_s': t['eval']})
            loss = dict(run['test_loss'])
            for epoch, s in run['curve']:
                curves.append({'net': name, 'certifier': cert, 'tool': tool, 'epoch': epoch, 'test_loss': loss.get(epoch),
                               **{k: s.get(k) for k in ['clean'] + EVALUATORS}})
    with open(os.path.join(out, 'report.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    for path, rows in (('results.csv', summary), ('curves.csv', curves)):
        if rows:
            with open(os.path.join(out, path), 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
