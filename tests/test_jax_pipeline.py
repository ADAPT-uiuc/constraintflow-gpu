"""Capture -> all shared passes -> Torch/JAX bounds, without dataset downloads."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def make_network(path, convolution=False, residual=False):
    import onnx
    from onnx import helper as h, numpy_helper as nh, TensorProto as T
    rng = np.random.default_rng(11)
    initializers = []

    def weight(name, shape):
        value = (rng.normal(size=shape) * 0.05).astype(np.float32)
        initializers.append(nh.from_array(value, name))
        return name

    if convolution:
        nodes = [h.make_node('Conv', ['input', weight('conv.weight', (2, 1, 3, 3)), weight('conv.bias', (2,))],
                             ['hidden'], dilations=[1, 1], group=1, kernel_shape=[3, 3], strides=[7, 7], pads=[1, 1, 1, 1]),
                 h.make_node('Relu', ['hidden'], ['relu'])]
        if residual:
            nodes.append(h.make_node('Conv', ['relu', weight('branch.weight', (2, 2, 3, 3)), weight('branch.bias', (2,))],
                                     ['branch'], dilations=[1, 1], group=1, kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1]))
            nodes.append(h.make_node('Add', ['hidden', 'branch'], ['added']))
        nodes.append(h.make_node('Flatten', ['added' if residual else 'relu'], ['flat'], axis=1))
        width = 32
    else:
        nodes = [h.make_node('Flatten', ['input'], ['flat_input'], axis=1),
                 h.make_node('Gemm', ['flat_input', weight('w1', (4, 784)), weight('b1', (4,))], ['hidden'], transB=1),
                 h.make_node('Relu', ['hidden'], ['flat'])]
        width = 4
    nodes.append(h.make_node('Gemm', ['flat', weight('w2', (10, width)), weight('b2', (10,))], ['output'], transB=1))
    graph = h.make_graph(nodes, 'jax-regression', [h.make_tensor_value_info('input', T.FLOAT, [1, 1, 28, 28])],
                        [h.make_tensor_value_info('output', T.FLOAT, [1, 10])], initializers)
    model = h.make_model(graph, opset_imports=[h.make_opsetid('', 13)], ir_version=8)
    onnx.checker.check_model(model)
    onnx.save(model, path)


def worker():
    import importlib
    import torch
    from constraintflow import cli
    target, program, network, output, segments = sys.argv[2:]
    torch.set_num_threads(2)
    torch.manual_seed(4)
    x, y = torch.rand(1, 1, 28, 28), torch.tensor([2])
    cli.get_dataset = lambda *args, **kwargs: (x, y)
    cli.app(args=['jit', str(ROOT / 'examples/compiler_examples' / (program + '.cf')),
                  '--network', network, '--output-path', output, '--target', target,
                  '--eps', '0.03', '--memory-budget-mb', '0', '--flow-segment-mb', segments],
            standalone_mode=False)
    for name in ('main', 'transformers', 'jit_constants', 'jax_flow'):
        sys.modules.pop(name, None)
    importlib.invalidate_caches()
    from main import run
    results = run(network, 1, 0.03, x, y, dataset='mnist', train=False,
                  print_intermediate_results=False, no_sparsity=False)
    np.savez(output + '.npz', *[v.detach().cpu().numpy() for v in results])


@unittest.skipUnless(importlib.util.find_spec('jax'), 'install constraintflow[jax]')
class JaxPipelineTest(unittest.TestCase):
    def test_torch_jax_and_segmented_bounds(self):
        env = dict(os.environ, JAX_PLATFORMS='cpu', OMP_NUM_THREADS='2',
                   PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get('PYTHONPATH', ''))
        with tempfile.TemporaryDirectory() as temp:
            for kind, program in [('dense', 'ibp'), ('conv', 'deeppoly'),
                                  ('residual', 'deeppoly_priority_add'), ('conv', 'zono')]:
                with self.subTest(kind=kind, program=program):
                    network = str(Path(temp) / (kind + '.onnx'))
                    make_network(network, convolution=kind != 'dense', residual=kind == 'residual')
                    outputs = []
                    for target, segment in [('torch', '0'), ('jax', '0'), ('jax', '0.002')]:
                        output = str(Path(temp) / (kind + program + target + segment))
                        result = subprocess.run([sys.executable, __file__, '--worker', target,
                                                 program, network, output, segment],
                                                cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
                        self.assertEqual(result.returncode, 0, result.stdout[-12000:] + result.stderr[-12000:])
                        with np.load(output + '.npz') as data:
                            outputs.append([data[k] for k in data.files])
                        if target == 'jax':
                            source = Path(output, 'jax_flow.py').read_text()
                            self.assertNotIn('import torch', source)
                            self.assertIn('@partial(jax.jit', source)
                            if segment != '0':
                                self.assertIn('def flow_seg_', source)
                    for actual in outputs[1:]:
                        for a, expected in zip(actual, outputs[0]):
                            np.testing.assert_allclose(a, expected, rtol=3e-5, atol=3e-5)


if __name__ == '__main__':
    if '--worker' in sys.argv:
        worker()
    else:
        unittest.main()
