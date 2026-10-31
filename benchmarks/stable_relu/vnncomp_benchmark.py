"""Apply stable-ReLU refinement to a cached unchunked VNN-COMP DeepPoly program.

Uses the harness's parsed box and output specification unchanged. The original
cached program is read-only; selected source is emitted into a separate package.
"""
import argparse
import importlib
import json
import hashlib
import signal
from pathlib import Path
import shutil
import sys
import time

import onnx
import torch

from constraintflow.lib.globals import device_mode
from constraintflow.lib.training import SpecBounds, load_program
from constraintflow.compiler.optimizations.stable_relu import rewrite, warmup_stable_relu


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--harness', required=True)
    p.add_argument('--kernel', required=True)
    p.add_argument('--network', required=True)
    p.add_argument('--spec', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--chunk-size', type=int, default=128)
    p.add_argument('--phase-limit', type=float, default=1200)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--reuse-source', action='store_true')
    a = p.parse_args()
    sys.path.insert(0, a.harness)
    from experiments.vnncomp.worker import load_spec, tf32_off
    spec = load_spec(a.spec)
    device_mode.set_mode('gpu')
    tf32_off()
    root = Path(a.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    program = root / 'selected_program'
    program.mkdir(exist_ok=True)
    original = Path(a.kernel).resolve()
    if not a.reuse_source:
        for name in ('__init__.py', 'main.py', 'jit_constants.py'):
            shutil.copyfile(original / name, program / name)
        shutil.copyfile(original / 'transformers_unchunked.py', program / 'transformers.py')
    started = time.perf_counter()
    inits = [i for i in onnx.load(a.network).graph.initializer if i.data_type == onnx.TensorProto.FLOAT]
    sb = SpecBounds.__new__(SpecBounds)
    sb.dataset = 'box-%s-%d-%d' % ('x'.join(map(str, spec['x_shape'])), spec['C'].shape[1], spec['C'].shape[0])
    sb.cudagraphs, sb.onnx_path = False, a.network
    sb.params = {i.name: torch.from_numpy(onnx.numpy_helper.to_array(i).copy()).cuda() for i in inits}
    module = load_program(str(program))
    net, ae, flow, explode = sb._build(module, torch.zeros(1, *spec['x_shape']), 1)
    torch.cuda.synchronize()
    load_s = time.perf_counter() - started
    print(json.dumps(dict(event='loaded', load_s=load_s)), flush=True)
    transformers = importlib.import_module(module.__name__.rsplit('.', 1)[0] + '.transformers')
    source = (program / 'transformers.py').read_text()
    if not a.reuse_source:
        started = time.perf_counter()
        source = rewrite(source, vars(transformers), explode(ae, 1), net,
                         chunk_size=a.chunk_size, report=lambda s: print(s, flush=True))
        (program / 'transformers.py').write_text(source)
        print(json.dumps(dict(event='transformed', transform_s=time.perf_counter()-started)), flush=True)
        if a.prepare_only:
            return
        env = dict(vars(transformers))
        exec(compile(source, str(program / 'transformers.py'), 'exec'), env)
        flow = env['flow']
    sb.programs = {('CROWN', 1): (net, ae, flow, explode)}
    def sha(path):
        digest = hashlib.sha256()
        with open(path, 'rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        return digest.hexdigest()
    metadata = dict(arguments=vars(a), torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                    tf32=False, network_sha256=sha(a.network), spec_sha256=sha(a.spec),
                    source_sha256=sha(program / 'transformers.py'),
                    optimizer_sha256=sha(Path(__file__).parents[2] / 'constraintflow/compiler/optimizations/stable_relu.py'))
    (root / 'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    def bound():
        lo = torch.from_numpy(spec['lo']).cuda().view(1, *spec['x_shape'])
        hi = torch.from_numpy(spec['hi']).cuda().view(1, *spec['x_shape'])
        C = torch.from_numpy(spec['C']).cuda().unsqueeze(0)
        d = torch.from_numpy(spec['d']).cuda()
        sb.begin_step()
        return sb.compute_spec_bounds('CROWN', lo, hi, C, d)[0].reshape(-1).float().cpu()
    def timeout(signum, frame):
        raise TimeoutError('benchmark phase exceeded its time limit')
    signal.signal(signal.SIGALRM, timeout)
    from torch._dynamo.utils import counters
    with open(root / 'results.jsonl', 'w') as out, torch.no_grad():
        for kind in ('warmup', 'timed'):
            print(json.dumps(dict(event=kind+'_start')), flush=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            resident = torch.cuda.memory_allocated()
            graphs = counters['stats']['unique_graphs']
            started = time.perf_counter()
            signal.setitimer(signal.ITIMER_REAL, 0 if kind == 'warmup' else max(0.001, a.phase_limit - load_s))
            lower = warmup_stable_relu(bound) if kind == 'warmup' else bound()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            signal.setitimer(signal.ITIMER_REAL, 0)
            new_graphs = counters['stats']['unique_graphs'] - graphs
            if kind == 'timed':
                assert new_graphs == 0, 'timed run compiled new graphs'
            record = dict(event=kind+'_end', load_s=load_s, verify_s=elapsed,
                          total_s=load_s+elapsed, lower=lower.tolist(),
                          peak_bytes=torch.cuda.max_memory_allocated()-resident,
                          peak_absolute_bytes=torch.cuda.max_memory_allocated(), new_graphs=new_graphs,
                          status='verified' if bool((lower > 0).all()) else 'unknown')
            out.write(json.dumps(record)+'\n'); out.flush()
            print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
