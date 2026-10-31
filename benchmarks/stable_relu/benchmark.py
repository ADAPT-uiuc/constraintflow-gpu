"""Compare selected refinement to an existing cf-gpu DeepPoly kernel on fixed boxes.

All timing includes CUDA synchronization. time_s includes model loading, input
transfer, and bounding; bound_s isolates bounding (GPUPoly's reported scope).
Cold compilation and warmup are recorded separately, never counted as steady state.
"""
import argparse
import ast
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import torch

from constraintflow.lib.globals import device_mode
from constraintflow.lib.spec import apply_spec_offset
from constraintflow.compiler.optimizations.stable_relu import rewrite, flatten_flow, warmup_stable_relu
from constraintflow.compiler.optimizations.row_chunks import _add_dels


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kernel', required=True)
    p.add_argument('--network', required=True)
    p.add_argument('--boxes', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--source-output', required=True)
    p.add_argument('--chunk-size', type=int, default=128)
    p.add_argument('--num-images', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--eager', action='store_true')
    p.add_argument('--compare', action='store_true')
    p.add_argument('--reuse-source', action='store_true')
    a = p.parse_args()
    device_mode.set_mode('gpu')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(Path(a.kernel).resolve()))
    import main as program
    import transformers
    z = np.load(a.boxes)
    if a.batch_size < 1 or a.num_images % a.batch_size or len(z['labels']) < a.num_images + a.batch_size:
        raise ValueError('need full timed batches and one additional warmup batch')
    shape = tuple(d.dim_value for d in onnx.load(a.network).graph.input[0].type.tensor_type.shape.dim)[1:]

    def build(i):
        l = torch.tensor(z['down'][i:i+a.batch_size], device='cuda')
        u = torch.tensor(z['up'][i:i+a.batch_size], device='cuda')
        y = torch.tensor(z['labels'][i:i+a.batch_size], device='cuda')
        x = ((l + u) / 2).reshape(a.batch_size, *shape)
        network, abs_elem = program.build(a.network, a.batch_size, 0, x, y, a.dataset, False, False, False, input_box=(l, u))
        return network, program.explode_inputs(abs_elem, a.batch_size)

    network, inputs = build(0)
    source = Path(a.kernel, 'transformers.py').read_text()
    if a.reuse_source:
        out = Path(a.source_output).read_text()
    else:
        out = rewrite(source, vars(transformers), inputs, network, chunk_size=a.chunk_size,
                      compile_kernels=not a.eager, report=lambda s: print(s, flush=True))
        Path(a.source_output).write_text(out)
    env = dict(vars(transformers))
    exec(compile(out, a.source_output, 'exec'), env)
    dense = None
    if a.compare:
        tree, fn = flatten_flow(source)
        fn.body = _add_dels(fn.body)
        dense = dict(vars(transformers))
        exec(compile(ast.fix_missing_locations(tree), '<dense>', 'exec'), dense)
    metadata = dict(arguments=vars(a), compile_cache_env={k: os.environ.get(k) for k in ('TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR')}, torch=torch.__version__, cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(), tf32=False,
                    kernel_sha256=hashlib.sha256(source.encode()).hexdigest(),
                    selected_source_sha256=hashlib.sha256(out.encode()).hexdigest(),
                    network_sha256=hashlib.sha256(Path(a.network).read_bytes()).hexdigest(),
                    boxes_sha256=hashlib.sha256(Path(a.boxes).read_bytes()).hexdigest())
    Path(a.output + '.metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    with open(a.output, 'w') as f, torch.no_grad():
        for i in [a.num_images, *range(0, a.num_images, a.batch_size)]:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            network, inputs = build(i)
            torch.cuda.synchronize()
            bound_start = time.perf_counter()
            from torch._dynamo.utils import counters
            graphs_before = counters['stats']['unique_graphs']
            if i == a.num_images:
                result = warmup_stable_relu(env['flow'], *inputs)
            else:
                result = env['flow'](*inputs)
            bounds = apply_spec_offset(network, result)
            torch.cuda.synchronize()
            end = time.perf_counter()
            new_graphs = counters['stats']['unique_graphs'] - graphs_before
            if i != a.num_images:
                assert new_graphs == 0, f'image {i} compiled {new_graphs} graphs after warmup'
            record = dict(new_graphs=new_graphs, kind='warmup' if i == a.num_images else 'batch', index=i,
                          images=[i, i+a.batch_size], time_s=end-start, bound_s=end-bound_start,
                          peak_bytes=torch.cuda.max_memory_allocated(),
                          lower=bounds[0].tolist(), upper=bounds[1].tolist(),
                          verified=bool((bounds[0] >= 0).all()),
                          verified_per_image=(bounds[0] >= 0).flatten(1).all(1).tolist())
            if dense is not None:
                reference = apply_spec_offset(network, dense['flow'](*inputs))
                torch.cuda.synchronize()
                record['max_abs_difference'] = [float((x-y).abs().max()) for x,y in zip(bounds, reference)]
                for got, expected in zip(bounds, reference):
                    torch.testing.assert_close(got, expected, atol=1e-3, rtol=1e-3)
                assert record['verified_per_image'] == (reference[0] >= 0).flatten(1).all(1).tolist()
            f.write(json.dumps(record) + '\n')
            f.flush()
            print({k:v for k,v in record.items() if k not in ('lower','upper')}, flush=True)


if __name__ == '__main__':
    main()
