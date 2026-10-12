"""Sequential GPU comparison using the existing CLI's full-call timings.

Run with a Python environment containing Torch/Inductor and CUDA-enabled JAX.
Completed cases are checkpointed, so an interrupted experiment can be resumed.
"""
import argparse
import ast
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
NETWORKS = [('N1', 'mnist_relu_3_50'), ('N6', 'mnist_relu_6_200'), ('N11', 'mnist_relu_4_1024')]
NETWORK_MAP = dict(NETWORKS + [('N12', 'convSmallRELU__Point'),
                             ('N18', 'skip__DiffAI'), ('N17', 'convSuperRELU__DiffAI')])


def execute(command, cwd, env, log):
    print('START', log.parent.name, log.name, flush=True)
    started = time.perf_counter()
    with log.open('w') as out:
        out.write('COMMAND ' + json.dumps(command) + '\n')
        out.flush()
        result = subprocess.run(command, cwd=cwd, env=env, stdout=out, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Command failed ({result.returncode}); see {log}')
    print('DONE', log.parent.name, log.name, f'{time.perf_counter() - started:.1f}s', flush=True)


def parse_result(log):
    source = log.read_text()
    warmup = [float(x) for x in re.findall(r'Warmup run \d+/5: ([\d.]+) s', source)]
    runs = [float(x) for x in re.findall(r'^Run \d+/5: ([\d.]+) s', source, re.M)]
    if len(warmup) != 5 or len(runs) != 5:
        raise ValueError(f'Expected exactly five warmups and five measured runs in {log}')
    bounds = {}
    for label in ('Lower', 'Upper'):
        match = re.search(label + r' bounds: tensor\((\[.*?\])\)', source, re.S)
        if not match:
            raise ValueError(f'Missing {label} bounds in {log}')
        bounds[label.lower()] = ast.literal_eval(match.group(1))
    return dict(warmup_seconds=warmup, run_seconds=runs,
                mean_seconds=statistics.mean(runs), median_seconds=statistics.median(runs),
                stdev_seconds=statistics.stdev(runs), bounds=bounds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--harness-root', type=Path, default=ROOT.parent)
    parser.add_argument('--data-root', type=Path, default=ROOT.parent / 'constraintflow-gpu')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--networks', nargs='+', choices=NETWORK_MAP, default=[n for n, _ in NETWORKS])
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--eps', type=float, default=0.004)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + str(args.harness_root),
               CUDA_VISIBLE_DEVICES='0', JAX_PLATFORMS='cuda',
               XLA_PYTHON_CLIENT_PREALLOCATE='false', CF_DISABLE_TF32='1',
               NVIDIA_TF32_OVERRIDE='0', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
               CF_CPU_THREADS='8', PYTHONUNBUFFERED='1')
    env['TORCHINDUCTOR_CACHE_DIR'] = str(output / 'inductor_cache')
    env['TRITON_CACHE_DIR'] = str(output / 'triton_cache')
    # Both libraries are verified in a separate process so the driver does not
    # retain a CUDA context or reserve GPU memory during the measurements.
    metadata_code = '''import json, torch, jax, jaxlib
import jax.numpy as jnp
assert torch.cuda.is_available()
assert all(d.platform == 'gpu' for d in jax.devices())
result = jax.jit(lambda x: x @ x)(jnp.ones((8, 8))).block_until_ready()
print(json.dumps(dict(torch=torch.__version__, jax=jax.__version__, jaxlib=jaxlib.__version__,
                     gpu=torch.cuda.get_device_name(0), jax_devices=[str(d) for d in jax.devices()])))
'''
    info = subprocess.check_output([sys.executable, '-c', metadata_code], env=env, text=True)
    metadata = json.loads(info.strip().splitlines()[-1])
    metadata.update(program='deeppoly_priority_add', batch_size=args.batch_size, epsilon=args.eps,
                    flow_segment_mb=0, networks=args.networks,
                    warmup_runs=5, measured_runs=5, precision='float32; TF32 disabled',
                    timing_scope='CLI full verification call, including network/input preparation and JAX bridge; GPU completion synchronized',
                    statistic='per-network arithmetic mean of 5 runs; geometric mean of Torch/JAX mean-time ratios',
                    python=sys.executable, environment={k: env[k] for k in ('CUDA_VISIBLE_DEVICES', 'JAX_PLATFORMS', 'XLA_PYTHON_CLIENT_PREALLOCATE', 'CF_DISABLE_TF32', 'NVIDIA_TF32_OVERRIDE', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'CF_CPU_THREADS')},
                    source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    snapshot = subprocess.check_output(['git', 'diff', '--binary'], cwd=ROOT)
    (output / 'tracked_source.patch').write_bytes(snapshot)
    metadata['source_hashes'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (ROOT / 'constraintflow').rglob('*.py')}
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    (output / 'nvidia-smi.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True))
    program = ROOT / 'examples/compiler_examples/deeppoly_priority_add.cf'
    records = {}
    import onnx
    for number, config in enumerate(args.networks):
        name = NETWORK_MAP[config]
        network = args.harness_root / 'nets/mnist' / (name + '.onnx')
        model = onnx.load(network)
        parameters = sum(math.prod(t.dims) for t in model.graph.initializer)
        records[config] = dict(network=name, parameters=parameters, backends={})
        for backend in (('torch', 'jax') if number % 2 == 0 else ('jax', 'torch')):
            case = output / (config + '_' + backend)
            case.mkdir(exist_ok=True)
            result_file = case / 'result.json'
            if result_file.exists():
                record = json.loads(result_file.read_text())
            else:
                base = [sys.executable, str(ROOT / 'constraintflow/cli.py')]
                common = [str(program), '--network', str(network), '--dataset', 'mnist',
                          '--batch-size', str(args.batch_size), '--eps', str(args.eps), '--device', 'gpu',
                          '--output-path', str(case / 'generated')]
                inductor = ['--inductor'] if backend == 'torch' else []
                if not (case / 'build.ok').exists():
                    execute(base + ['jit'] + common + ['--target', backend, '--flow-segment-mb', '0'] + inductor,
                            args.data_root, env, case / 'build.log')
                    (case / 'build.ok').touch()
                execute(base + ['run'] + common + inductor + ['--warmup', '5', '--repeat', '5'],
                        args.data_root, env, case / 'run.log')
                record = parse_result(case / 'run.log')
                result_file.write_text(json.dumps(record, indent=2))
            records[config]['backends'][backend] = record
            (output / 'partial_results.json').write_text(json.dumps(records, indent=2))
        torch_result, jax_result = (records[config]['backends'][b] for b in ('torch', 'jax'))
        import numpy as np
        diffs = []
        for label in ('lower', 'upper'):
            a, b = np.asarray(torch_result['bounds'][label]), np.asarray(jax_result['bounds'][label])
            if not (np.isfinite(a).all() and np.isfinite(b).all()):
                raise ValueError(f'Nonfinite bounds for {config}')
            np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4,
                                       err_msg=f'{config} {label} bounds differ')
            diffs.append(float(np.max(np.abs(a - b))))
        records[config]['max_absolute_bound_difference'] = max(diffs)
        records[config]['torch_over_jax'] = torch_result['mean_seconds'] / jax_result['mean_seconds']
        print('RESULT', config, json.dumps(records[config]), flush=True)
    geomean = math.exp(statistics.mean(math.log(r['torch_over_jax']) for r in records.values()))
    summary = dict(metadata=metadata, networks=records, geomean_torch_over_jax=geomean)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    with (output / 'summary.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['config', 'network', 'parameters', 'torch_mean_s', 'jax_mean_s', 'torch_over_jax', 'max_abs_bound_difference'])
        for config, row in records.items():
            writer.writerow([config, row['network'], row['parameters'], row['backends']['torch']['mean_seconds'],
                             row['backends']['jax']['mean_seconds'], row['torch_over_jax'], row['max_absolute_bound_difference']])
    print('GEOMEAN_TORCH_OVER_JAX', geomean, flush=True)


if __name__ == '__main__':
    main()
