"""Compare final-IR stable ReLU with the legacy rewrite and a dense baseline.

Compile the input --kernel using jit --stable-relu --inductor. All three paths
then use exactly the same final dense IR, weights, input boxes and chunk size.
Model loading is outside timing; CUDA work is synchronized. Warm every kernel
and every measured box before timing, and reject any compilation in timed runs.
"""
import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import torch

from constraintflow.compiler.optimizations.stable_relu import rewrite, warmup_stable_relu
from constraintflow.lib.globals import device_mode
from constraintflow.lib.spec import apply_spec_offset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kernel', required=True)
    p.add_argument('--network', required=True)
    p.add_argument('--boxes', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--chunk-size', type=int, default=128)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--num-batches', type=int, default=5)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--skip-dense', action='store_true',
                   help='Compare selected paths only when the dense baseline cannot execute.')
    a = p.parse_args()
    torch.set_num_threads(8)
    device_mode.set_mode('gpu')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(Path(a.kernel).resolve()))
    import main as program
    import transformers
    z = np.load(a.boxes)
    shape = tuple(d.dim_value for d in onnx.load(a.network).graph.input[0].type.tensor_type.shape.dim)[1:]
    if len(z['labels']) < (a.num_batches + 1) * a.batch_size:
        raise ValueError('need one extra batch for warmup')
    def build(i):
        sl = slice(i * a.batch_size, (i+1) * a.batch_size)
        l = torch.tensor(z['down'][sl], device='cuda')
        u = torch.tensor(z['up'][sl], device='cuda')
        y = torch.tensor(z['labels'][sl], device='cuda')
        x = ((l + u)/2).reshape(a.batch_size, *shape)
        net, ae = program.build(a.network, a.batch_size, 0, x, y, a.dataset,
                                False, False, False, input_box=(l, u))
        return net, program.explode_inputs(ae, a.batch_size)
    net, inputs = build(a.num_batches)
    dense_source = Path(a.kernel, 'transformers_dense.py').read_text()
    dense = dict(vars(transformers))
    exec(compile(dense_source, str(Path(a.kernel, 'transformers_dense.py')), 'exec'), dense)
    old_source = rewrite(dense_source, dense, inputs, net, a.chunk_size, compile_kernels=True)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output + '.legacy.py').write_text(old_source)
    old = dict(dense)
    exec(compile(old_source, a.output + '.legacy.py', 'exec'), old)
    flows = {}
    if not a.skip_dense:
        flows['dense'] = torch.compile(dense['flow'], fullgraph=True, dynamic=False,
                                       options={'layout_optimization': False})
    flows.update(old=old['flow'], ir=transformers.flow)
    from torch._dynamo.utils import counters
    records, warmups, errors = [], {}, []
    max_abs = [0., 0.]
    bitwise = True
    def invoke(kind, args):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = counters['stats']['unique_graphs']
        start = time.perf_counter()
        result = flows[kind](*args)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        return result, dict(kind=kind, seconds=elapsed,
            peak_bytes=torch.cuda.max_memory_allocated(),
            new_graphs=counters['stats']['unique_graphs'] - before)
    with torch.no_grad():
        for kind, flow in flows.items():
            print('warmup', kind, flush=True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            warmup_stable_relu(flow, *inputs)
            torch.cuda.synchronize()
            warmups[kind] = time.perf_counter() - start
            print('warmup done', kind, warmups[kind], flush=True)
        for i in range(a.num_batches):
            net, args = build(i)
            # Untimed correctness and per-box warmup before collecting timings.
            results = {kind: apply_spec_offset(net, flow(*args)) for kind, flow in flows.items()}
            for j, (got, ref) in enumerate(zip(results['ir'], results['old'])):
                max_abs[j] = max(max_abs[j], float((got-ref).abs().max()))
                bitwise &= torch.equal(got, ref)
                torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)
            ir_verified = (results['ir'][0] >= 0).flatten(1).all(1)
            old_verified = (results['old'][0] >= 0).flatten(1).all(1)
            assert torch.equal(ir_verified, old_verified)
            errors.append(dict(batch=i, max_abs_old=[float((x-y).abs().max())
                          for x, y in zip(results['ir'], results['old'])],
                          max_abs_dense=[float((x-y).abs().max())
                          for x, y in zip(results['ir'], results['dense'])] if 'dense' in results else None,
                          verified=ir_verified.tolist(),
                          lower=results['ir'][0].tolist(), upper=results['ir'][1].tolist()))
            del results
            for repeat in range(a.repeats):
                order = list(flows)
                rotate = (i + repeat) % len(order)
                for kind in order[rotate:] + order[:rotate]:
                    result, row = invoke(kind, args)
                    assert row['new_graphs'] == 0, row
                    row.update(batch=i, repeat=repeat)
                    records.append(row)
                    del result
            print('batch done', i, flush=True)
    medians = {kind: statistics.median(r['seconds'] for r in records if r['kind'] == kind)
               for kind in flows}
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    summary = dict(arguments=vars(a), gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        tf32=False, medians_s=medians, speedups={kind: medians['dense']/medians[kind] for kind in ('old', 'ir')} if 'dense' in medians else None,
        ir_over_old=medians['ir']/medians['old'], max_abs_old=max_abs, bitwise_equal=bitwise,
        warmup_s=warmups, measurements=records, bounds=errors,
        hashes={name: sha(path) for name, path in dict(network=a.network, boxes=a.boxes,
                ir=Path(a.kernel, 'transformers.py'), dense=Path(a.kernel, 'transformers_dense.py'),
                legacy=a.output + '.legacy.py').items()})
    Path(a.output).write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('bounds','measurements','hashes')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
