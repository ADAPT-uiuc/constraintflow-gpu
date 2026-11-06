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
from matplotlib.colors import LinearSegmentedColormap
import torch

CORE = ['IBP', 'CROWN-FAST', 'zid', 'CROWN']  # trained in both tools (zid: cf-gpu only)
TOOLS = ['lirpa', 'cf']
EVALUATORS = ['IBP', 'CROWN-IBP', 'CROWN', 'zid']  # curve evaluators; final weights are judged by every certifier
COLORS = {'IBP': '#2a78d6', 'CROWN-FAST': '#eb6834', 'zid': '#1baf7a', 'CROWN': '#eda100', 'zono': '#e87ba4'}  # dataviz slots 1-5
MARKERS = {'IBP': 'o', 'CROWN-FAST': 's', 'zid': '^', 'CROWN': 'D', 'zono': 'v'}
RAMP = ['#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']  # blue 250-700
# custom stops get their own figure, so all 8 categorical slots, each with its own marker (separable without colour)
STOP_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
STOP_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
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


def is_stop(cert):
    return cert.startswith('DP-stop')


def load_net(d):
    """Everything run_grid.sh wrote for one net."""
    before_all = read_json(os.path.join(d, 'before', 'precision_all_cf.json'))
    stops = sorted(os.path.basename(p) for p in glob.glob(os.path.join(d, 'DP-stop*')))
    net = {'before': {t: scores(read_json(os.path.join(d, 'before', f'precision_onnx_{t}.json'))) for t in TOOLS},
           'before_all': scores(before_all), 'judges': list(before_all['methods']) if before_all else EVALUATORS,
           'certs': CORE + ['zono'] + stops, 'runs': {}}
    net['before_all'] = net['before_all'] or net['before']['cf']  # the core evaluation until the cross product exists
    for cert in net['certs']:
        for tool in TOOLS:
            run = os.path.join(d, cert, tool)
            record = read_json(os.path.join(run, 'record.json'))
            if not record:
                continue
            # every checkpoint judged by every certifier when available, else the core curve
            curve = read_json(os.path.join(run, 'precision_curve_all_cf.json')) or read_json(os.path.join(run, 'precision_curve_cf.json')) or []
            final = scores(read_json(os.path.join(run, 'precision_all_cf.json')))
            net['runs'][cert, tool] = {
                'timing': timing(record),
                'curve': [(0, net['before_all'])] + [(r['epoch'], scores(r)) for r in curve],  # epoch 0: ONNX weights, every judge
                'test_loss': [(r['epoch'], r['avg'].get('Loss')) for r in record['records'] if not r['train']],
                'final_by': {t: scores(read_json(os.path.join(run, f'precision_ckpt_{t}.json'))) for t in TOOLS},
                # the cross product; the last curve point until it exists
                'final': final or (scores(curve[-1]) if curve else {}),
            }
    return net


def net_section(name, d, net):
    lines = [f'## {name}', '', 'Training cost and clean accuracy (before → after):', '',
             '| Certifier | Trained by | Clean | Training time (s) | Setup + training (s) | s / epoch | Speedup (training) |',
             '|---|---|---|---|---|---|---|']
    before = net['before']['cf']
    for (cert, tool), run in net['runs'].items():
        t, other = run['timing'], net['runs'].get((cert, 'lirpa'))
        speedup = f"{other['timing']['train'] / t['train']:.2f}×" if tool == 'cf' and other else ''
        lines.append(f"| {cert} | {TOOL_NAMES[tool]} | {fmt(before.get('clean'))} → {fmt(run['final'].get('clean'))} "
                     f"| {t['train']:.1f} | {t['total']:.1f} | {t['per_epoch']:.2f} | {speedup} |")

    judges = net['judges']
    lines += ['', 'Verified precision of the final weights: trained with (rows) × verified with (columns); '
              'cf-gpu, full test set, final eps. **Bold** = best verifier for that model.', '',
              '| Trained with | Trained by | ' + ' | '.join(judges) + ' | Highest | Verified by |',
              '|---|---|' + '---|' * len(judges) + '---|---|']
    rows = [('ONNX (before)', '', net['before_all'])] + [(c, TOOL_NAMES[t], r['final']) for (c, t), r in net['runs'].items()]
    for cert, tool, s in rows:
        best = max((s[j] for j in judges if s.get(j) is not None), default=None)
        cells = [(f'**{fmt(s[j])}**' if s.get(j) is not None and s[j] == best and best > 0 else fmt(s.get(j))) for j in judges]
        #FEAT: the model's highest verified precision over all verifiers, and which verifiers reach it
        by = ', '.join(j for j in judges if s.get(j) is not None and s[j] == best) if best else ''
        lines.append(f'| {cert} | {tool} | ' + ' | '.join(cells) + f' | **{fmt(best)}** | {by} |')

    lines += ['', 'Bounds and gradients, auto_LiRPA vs cf-gpu (largest relative difference), and the verified '
              'precision each tool gives the same final weights:', '',
              '| Weights | IBP lb | IBP grad | CROWN-IBP lb | CROWN-IBP grad | CROWN lb | CROWN grad '
              f"| Precision by auto_LiRPA / cf-gpu ({', '.join(EVALUATORS)}) |",
              '|---|---|---|---|---|---|---|---|']
    agreement = [('ONNX (before)', os.path.join(d, 'before'), 'onnx', None)]
    agreement += [(f'{cert}, trained by {TOOL_NAMES[t]}', os.path.join(d, cert, f'bounds_{t}'), 'ckpt', net['runs'].get((cert, t)))
                  for cert in ('IBP', 'CROWN-FAST', 'CROWN') for t in TOOLS]
    for label, folder, weights, run in agreement:
        diffs = [fmt(x, '.1e') for m in ('IBP', 'CROWN-IBP', 'CROWN') for x in max_rel_diffs(folder, weights, m)]
        by = run['final_by'] if run else net['before']
        judged = ', '.join(f"{fmt(by['lirpa'].get(m))} / {fmt(by['cf'].get(m))}" for m in EVALUATORS)
        lines.append(f'| {label} | ' + ' | '.join(diffs) + f' | {judged} |')
    lines += ['', f'![cross product](plots/{name}_matrix.png)', '',
              f'![verified precision over training](plots/{name}_verified.png)', '']
    if any(is_stop(c) for c, _ in net['runs']):
        lines += [f'![custom stops over training](plots/{name}_stops.png)', '']
    lines += [f'![test loss over training](plots/{name}_loss.png)', '']
    return lines


def plot_matrix(name, net, out):
    judges = net['judges']
    rows = [('ONNX (before)', net['before_all'])] + [(f'{c} ({TOOL_NAMES[t]})', r['final']) for (c, t), r in net['runs'].items()]
    data = [[s.get(j, float('nan')) if s.get(j) is not None else float('nan') for j in judges] for _, s in rows]
    # last column: each model's highest verified precision over all verifiers, after a blank gap column
    data = [row + [float('nan'), max((v for v in row if v == v), default=float('nan'))] for row in data]
    columns = judges + ['', 'HIGHEST']
    fig, ax = plt.subplots(figsize=(max(6.0, 2.5 + 0.75 * len(columns)), 1.4 + 0.38 * len(rows)))
    cmap = LinearSegmentedColormap.from_list('blue', ['#f4f8fe'] + RAMP)
    ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect='auto')
    for i, row in enumerate(data):
        for j, v in enumerate(row):
            if v == v:
                r, g, b, _ = cmap(v)  # text ink by the cell's luminance
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7,
                        color='white' if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.6 else '#0b0b0b')
    ax.set_xticks(range(len(columns)), columns, rotation=45, ha='right', fontsize=8)
    ax.get_xticklabels()[-1].set_fontweight('bold')
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8)
    ax.set_xlabel('verified with (cf-gpu, final weights, full test set)')
    ax.set_ylabel('trained with')
    ax.set_title(name, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'plots', f'{name}_matrix.png'), dpi=130)
    plt.close(fig)


REFERENCE = '#52514e'  # neutral ink for core certifiers drawn as references in the stops figure


def style(cert, stops, reference=False):
    """Core certifiers and custom stops each use the categorical palette, in separate figures; in the stops
    figure the core certifiers are gray dotted references with hollow markers."""
    if reference:
        return dict(color=REFERENCE, linestyle=':', marker=MARKERS[cert], markersize=5, markerfacecolor='white', linewidth=1.5)
    if is_stop(cert):
        i = stops.index(cert) % len(STOP_COLORS)
        return dict(color=STOP_COLORS[i], marker=STOP_MARKERS[i], markersize=5, linewidth=2)
    return dict(color=COLORS[cert], marker=MARKERS[cert], markersize=5, linewidth=2)


def curve_figure(path, title, runs, panels, stops, references=None):
    """One panel per key ('clean', 'own' = the run's own certifier, or a certifier); clean on its own axis."""
    nrows = -(-len(panels) // 5)  # at most 5 panels per row, rows balanced
    cols = -(-len(panels) // nrows)
    fig, axes = plt.subplots(nrows, cols, figsize=(4 * cols, 3.6 * nrows + 0.8), squeeze=False)
    axes = axes.flatten()
    for ax, key in zip(axes, panels):
        for (cert, tool), run in runs.items():
            k = cert if key == 'own' else key
            pts = [(e, s.get(k)) for e, s in run['curve'] if s.get(k) is not None]
            if pts:
                ax.plot(*zip(*pts), STYLES[tool], label=f'{cert} ({TOOL_NAMES[tool]})', **style(cert, stops))
        ends = []
        for (cert, tool), run in (references or {}).items():
            k = cert if key == 'own' else key
            pts = [(e, s.get(k)) for e, s in run['curve'] if s.get(k) is not None]
            if pts:
                ax.plot(*zip(*pts), label=f'{cert} (reference)', **style(cert, stops, reference=True))
                ends.append([pts[-1][1], pts[-1][0], cert])
        if ends:  # end-of-line labels, spread apart vertically so they don't collide
            low, high = ax.get_ylim()
            gap = 0.045 * (high - low)
            ends.sort()
            for i in range(1, len(ends)):
                ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
            ends[-1][0] = min(ends[-1][0], high - gap / 2)  # keep the stack inside the axis
            for i in range(len(ends) - 2, -1, -1):
                ends[i][0] = min(ends[i][0], ends[i + 1][0] - gap)
            for y, x, cert in ends:
                ax.annotate(cert, (x, y), xytext=(5, 0), textcoords='offset points', fontsize=6, color=REFERENCE,
                            va='center', annotation_clip=False)
        ax.set_title({'clean': 'Clean accuracy', 'own': 'Verified by its own certifier'}.get(key, f'{key}-verified'), fontsize=10)
        ax.set_xlabel('epoch')
        ax.grid(alpha=0.25)
        if references:
            ax.set_xlim(right=ax.get_xlim()[1] + 12)
    for ax in axes[len(panels):]:
        ax.axis('off')
    axes[0].set_ylabel('fraction of test set')
    # clean accuracy on its own zoomed axis (MNIST moves within ~0.97-0.99); verified panels share 0-1
    for ax in axes[1:len(panels)]:
        ax.set_ylim(0, 1)
        ax.sharey(axes[1])
    handles, labels = axes[0].get_legend_handles_labels()
    ncol = min(len(labels), 2 * cols, 8)  # legend wraps to the figure width
    fig.legend(handles, labels, loc='lower center', ncol=ncol, fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.04 * -(-len(labels) // ncol) / nrows + 0.02, 1, 1))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_curves(name, net, out):
    stops = sorted(c for c, _ in net['runs'] if is_stop(c))
    core = {k: r for k, r in net['runs'].items() if not is_stop(k[0])}
    stop_runs = {k: r for k, r in net['runs'].items() if is_stop(k[0])}
    # all judges once the cross-product curves exist, else the core evaluators
    full = all(any(set(net['judges']) <= set(s) for _, s in r['curve'][1:]) for r in net['runs'].values())
    judges = net['judges'] if full else EVALUATORS
    by = 'every certifier' if full else 'the core evaluators'
    curve_figure(os.path.join(out, 'plots', f'{name}_verified.png'),
                 f'{name}: trained with IBP, CROWN-FAST, zid, CROWN, zono; every checkpoint judged by {by} (cf-gpu)',
                 core, ['clean'] + judges, stops)
    if stop_runs:
        curve_figure(os.path.join(out, 'plots', f'{name}_stops.png'),
                     f'{name}: trained with DeepPoly stopped KK layers back; every checkpoint judged by {by} (cf-gpu)',
                     stop_runs, ['clean', 'own'] + judges, stops,
                     references={k: r for k, r in core.items() if k[1] == 'cf'})

    fig, axes = plt.subplots(1, 2 if stop_runs else 1, figsize=(14 if stop_runs else 7, 4), squeeze=False, sharey=True)
    for ax, runs, title in zip(axes[0], (core, stop_runs), ('core certifiers', 'custom stops')):
        for (cert, tool), run in runs.items():
            pts = [(e, v) for e, v in run['test_loss'] if v is not None]
            if pts:
                ax.plot(*zip(*pts), STYLES[tool], label=f'{cert} ({TOOL_NAMES[tool]})', color=style(cert, stops)['color'], linewidth=2)
        ax.set_yscale('log')
        ax.set_xlabel('epoch')
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, frameon=False)
    axes[0][0].set_ylabel('robust test loss')
    fig.suptitle(f"{name}: test loss (own certifier, scheduled eps)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'plots', f'{name}_loss.png'), dpi=130)
    plt.close(fig)


def write_csv(path, rows):
    if rows:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def main():
    out = sys.argv[1]
    nets = {os.path.basename(d): d for d in sorted(glob.glob(os.path.join(out, '*'))) if os.path.isdir(os.path.join(d, 'before'))}
    lines = ['# Certified training: certifiers and tools', '',
             'Training time = the training epochs only; setup + training adds TrainingBounds / BoundedModule construction. '
             'Not timed: cf-gpu code generation, warmup, per-epoch test evaluation. '
             'CROWN-FAST = pure CROWN-IBP lower bounds every epoch; CROWN = CROWN lower bounds with CROWN intermediate bounds; '
             'DP-stopKK = DeepPoly with back-substitution stopped KK layers back (only the stops that differ on a net). '
             'zid, zono and DP-stopKK have no auto_LiRPA counterpart.', '']
    summary, curves = [], []
    os.makedirs(os.path.join(out, 'plots'), exist_ok=True)
    for name, d in nets.items():
        net = load_net(d)
        lines += net_section(name, d, net)
        plot_matrix(name, net, out)
        plot_curves(name, net, out)
        for (cert, tool), run in net['runs'].items():
            t = run['timing']
            summary.append({'net': name, 'certifier': cert, 'tool': tool,
                            **{f'before_{k}': net['before_all'].get(k, net['before']['cf'].get(k)) for k in ['clean'] + net['judges']},
                            **{f'after_{k}': run['final'].get(k) for k in ['clean'] + net['judges']},
                            'after_highest': max((run['final'][j] for j in net['judges'] if run['final'].get(j) is not None), default=None),
                            'train_s': t['train'], 'setup_plus_train_s': t['total'], 's_per_epoch': t['per_epoch'],
                            'compile_s': t['compile'], 'eval_s': t['eval']})
            loss = dict(run['test_loss'])
            for epoch, s in run['curve']:
                curves.append({'net': name, 'certifier': cert, 'tool': tool, 'epoch': epoch, 'test_loss': loss.get(epoch), **s})
    with open(os.path.join(out, 'report.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    write_csv(os.path.join(out, 'results.csv'), summary)
    write_csv(os.path.join(out, 'curves.csv'), curves)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
