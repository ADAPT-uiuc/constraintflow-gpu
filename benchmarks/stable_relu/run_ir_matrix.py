"""Build and compare representative stable-ReLU configurations on local datasets."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import onnx
import torch
from torchvision import datasets, transforms

from constraintflow.lib.spec import ImageDataset

CONFIGS = [
    ('N2', 'mnist/mnist_relu_3_100.onnx', 'mnist', .004, 1),
    ('N19_b3', 'cifar/cifar_relu_4_100.onnx', 'cifar', .00004, 3),
    ('N26', 'cifar/convSmallRELU__Point.onnx', 'cifar', .00004, 1),
    ('N32_b3', 'cifar/convBigRELU__DiffAI.onnx', 'cifar', .5/255, 3),
    ('N33', 'cifar/ResNetTiny_DiffAI.onnx', 'cifar', .001, 1),
    ('N42', 'TinyImageNet_resnet_medium.onnx', 'tinyimagenet', 1/255, 1),
    ('N36', 'cifar/ResNet18_PGD.onnx', 'cifar', .00004, 1),
]


def write_boxes(root, dataset, eps, batch, path):
    data_root = root / 'constraintflow-gpu'
    if dataset == 'mnist':
        data = datasets.MNIST(data_root, train=False, download=False, transform=transforms.ToTensor())
    elif dataset == 'cifar':
        data = datasets.CIFAR10(data_root, train=False, download=False, transform=transforms.ToTensor())
    else:
        from constraintflow.lib.tinyimagenet import TinyImageNetValidation
        data = TinyImageNetValidation(data_root / 'tinyimagenet/tiny-imagenet-200')
    # Five measured batches and one separate warmup batch, all deterministic.
    x = torch.stack([data[i][0] for i in range(6 * batch)])
    labels = np.array([data[i][1] for i in range(6 * batch)])
    mean, std = ImageDataset._normalization(x, dataset)
    if dataset == 'mnist':
        mean, std = mean.reshape(1, 1, 1, 1), std.reshape(1, 1, 1, 1)
    lower = ((x - eps).clamp(0, 1) - mean) / std
    upper = ((x + eps).clamp(0, 1) - mean) / std
    np.savez(path, down=lower.flatten(1).numpy(), up=upper.flatten(1).numpy(), labels=labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--harness', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--configs', nargs='*', choices=[c[0] for c in CONFIGS])
    parser.add_argument('--skip-dense-configs', nargs='*', default=[], choices=[c[0] for c in CONFIGS],
                        help='Explicitly omit an unavailable dense baseline for these configurations.')
    args = parser.parse_args()
    root = args.harness.resolve()
    worktree = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(worktree) + ':' + str(root), CF_DISABLE_TF32='1',
               OMP_NUM_THREADS='8', TORCHINDUCTOR_COMPILE_THREADS='8')
    report = []
    for name, relative, dataset, eps, batch in CONFIGS:
        if args.configs and name not in args.configs:
            continue
        print('START', name, flush=True)
        folder = output / name
        folder.mkdir(exist_ok=True)
        boxes = folder / 'boxes.npz'
        write_boxes(root, dataset, eps, batch, boxes)
        network = root / 'nets' / relative
        kernel = folder / 'program'
        model = onnx.load(network)
        shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim][1:]
        outputs = 200 if dataset == 'tinyimagenet' else 10
        # Capture needs only shapes. Actual labeled margin specifications and
        # normalized dataset boxes are supplied by compare_ir during execution.
        box = 'box-' + 'x'.join(map(str, shape)) + f'-{outputs}-{outputs-1}'
        command = [sys.executable, '-m', 'constraintflow.cli', 'jit',
                   str(worktree / 'examples/compiler_examples/deeppoly_priority_add.cf'),
                   '--network', str(network), '--dataset', box, '--device', 'gpu',
                   '--batch-size', str(batch), '--stable-relu', '--stable-relu-chunk', '128',
                   '--inductor', '--fuse-affine-subst', '--output-path', str(kernel)]
        started = time.perf_counter()
        with open(folder / 'compile.log', 'w') as log:
            result = subprocess.run(command, cwd=folder, env=env, stdout=log, stderr=subprocess.STDOUT)
        status = dict(name=name, compile_s=time.perf_counter()-started,
                      compile_rc=result.returncode, compile_command=command)
        if result.returncode == 0:
            command = [sys.executable, str(worktree / 'benchmarks/stable_relu/compare_ir.py'),
                       '--kernel', str(kernel), '--network', str(network), '--boxes', str(boxes),
                       '--dataset', dataset, '--batch-size', str(batch),
                       '--output', str(folder / 'result.json')]
            if name in args.skip_dense_configs:
                command.append('--skip-dense')
            with open(folder / 'benchmark.log', 'w') as log:
                result = subprocess.run(command, cwd=folder, env=env, stdout=log, stderr=subprocess.STDOUT)
            status.update(benchmark_rc=result.returncode, benchmark_command=command)
        report.append(status)
        (output / 'status.json').write_text(json.dumps(report, indent=2) + '\n')
        print('DONE', status, flush=True)
    if any(row.get('benchmark_rc', 1) != 0 for row in report):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
