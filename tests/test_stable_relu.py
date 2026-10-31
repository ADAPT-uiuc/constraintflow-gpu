import ast
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from constraintflow.compiler.optimizations.stable_relu import refine_stable, selected_patches_to_dense, flatten_flow, relu_pairs
from constraintflow.gbcsr.sparse_block import patches_to_dense


class StableReLUTest(unittest.TestCase):
    def test_selection_boundaries_padding_and_changing_masks(self):
        calls = []
        def kernel(indices):
            calls.append(indices.clone())
            return indices[None].float() - 10, indices[None].float() + 10
        lower = torch.tensor([[1., -2., -1., 0., -3., -4.]])
        upper = torch.tensor([[2., 3., 0., 0., 5., 6.]])
        lo, hi = refine_stable(lower, upper, kernel, (), 2)
        self.assertEqual([x.tolist() for x in calls], [[1, 4], [5, 5]])
        torch.testing.assert_close(lo, torch.tensor([[1., -9., -1., 0., -6., -5.]]))
        torch.testing.assert_close(hi, torch.tensor([[2., 11., 0., 0., 14., 15.]]))
        self.assertEqual(lower.tolist(), [[1., -2., -1., 0., -3., -4.]])
        calls.clear()
        refine_stable(torch.ones_like(lower), torch.ones_like(upper), kernel, (), 2)
        self.assertEqual(calls, [])
        refine_stable(-torch.ones_like(lower), torch.ones_like(upper), kernel, (), 2)
        self.assertEqual([x.tolist() for x in calls], [[0, 1], [2, 3], [4, 5]])

    def test_invalid_and_nonfinite_bounds_are_not_stability_proofs(self):
        called = []
        def kernel(indices):
            called.extend(indices.tolist())
            return torch.zeros(1, len(indices)), torch.ones(1, len(indices))
        refine_stable(torch.tensor([[2., float('nan'), float('inf')]]),
                      torch.tensor([[1., 2., float('inf')]]), kernel, (), 3)
        self.assertEqual(called, [0, 1, 2])

    def test_selected_patch_geometry_matches_dense(self):
        torch.manual_seed(9)
        for dims in [(1, 3, 4, 5, 2, 3, 2, 7, 8, 1, 2, 2, 1),
                     (1, 2, 3, 3, 2, 5, 5, 4, 4, 2, 2, 1, 1)]:
            b, oc, ox, oy, ic, kx, ky, *_ = dims
            pieces = torch.randn(b, oc * ox * oy, ic * kx * ky, dtype=torch.float64)
            indices = torch.tensor([oc * ox * oy - 1, 0, oy + 1, 0])
            got = selected_patches_to_dense(pieces.index_select(1, indices), indices, *dims)
            expected = patches_to_dense(pieces, *dims).index_select(1, indices)
            torch.testing.assert_close(got, expected, atol=0, rtol=0)

    def test_reused_regions_are_alpha_renamed(self):
        source = """
def flow_seg_0(x):
    temp = x + 1
    a = temp * 2
    b = a + 3
    return (a, b)
flow_seg_1 = flow_seg_0
def flow(x):
    a, b = flow_seg_0(x)
    c, d = flow_seg_1(a)
    return (c, d)
"""
        original, transformed = {}, {}
        exec(source, original)
        tree, fn = flatten_flow(source)
        exec(compile(ast.fix_missing_locations(tree), '<flat>', 'exec'), transformed)
        self.assertEqual(original['flow'](2), transformed['flow'](2))
        names = [s.targets[0].id for s in fn.body if isinstance(s, ast.Assign)]
        self.assertEqual(len(names), len(set(names)))

    def test_tensor_zero_guard_is_recognized(self):
        body = ast.parse("""
width = operator.sub(upper, lower)
a = operator.gt(lower, 0.0)
b = operator.le(upper, operator.mul(ones, 0.0))
""").body
        self.assertEqual(relu_pairs(body), [('lower', 'upper')])

    def test_jit_cli_and_numerical_equivalence_across_boxes(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            torch.manual_seed(3)
            model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(),
                                        torch.nn.Linear(8, 6), torch.nn.ReLU(), torch.nn.Linear(6, 2)).eval()
            onnx = str(Path(tmp, 'model.onnx'))
            torch.onnx.export(model, torch.zeros(1, 4), onnx, dynamo=False)
            folder = str(Path(tmp, 'program'))
            env = dict(os.environ, PYTHONPATH=str(root))
            result = subprocess.run([sys.executable, '-m', 'constraintflow.cli', 'jit',
                str(root / 'examples/compiler_examples/deeppoly_priority_add.cf'),
                '--network', onnx, '--dataset', 'box-4-2-2', '--device', 'cpu',
                '--stable-relu', '--stable-relu-chunk', '4', '--fuse-affine-subst',
                '--output-path', folder], env=env, cwd=tmp, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            check = """
import sys
from pathlib import Path
import torch
sys.path.insert(0, sys.argv[1])
import main, transformers
from constraintflow.lib.globals import device_mode
device_mode.set_mode('cpu')
env = dict(vars(transformers))
exec(Path(sys.argv[1], 'transformers_dense.py').read_text(), env)
for width in (0., 0.01, 0.5, 10.):
    x = torch.tensor([[.2, -.3, .4, .7]])
    net, ae = main.build(sys.argv[2], 1, 0, x, torch.zeros(1, dtype=torch.long),
                        'box-4-2-2', False, False, False, input_box=(x-width, x+width))
    net[-1].weight = torch.eye(2).unsqueeze(0)
    args = main.explode_inputs(ae, 1)
    for got, expected in zip(main.flow(*args), env['flow'](*args)):
        torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)
"""
            result = subprocess.run([sys.executable, '-c', check, folder, onnx],
                                    env=env, cwd=tmp, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_fullgraph_kernel_reuses_fixed_shape_across_masks(self):
        from torch._dynamo.testing import CompileCounterWithBackend
        counter = CompileCounterWithBackend('inductor')
        def body(indices, weight, lower, upper):
            w = weight.index_select(0, indices)
            return (lower @ w.clamp(min=0).T + upper @ w.clamp(max=0).T,
                    upper @ w.clamp(min=0).T + lower @ w.clamp(max=0).T)
        compiled = torch.compile(body, fullgraph=True, dynamic=False, backend=counter)
        weight = torch.randn(11, 7, device='cuda')
        args = weight, -torch.ones(1, 7, device='cuda'), torch.ones(1, 7, device='cuda')
        for unstable in (1, 7, 11, 0, 3):
            lower = torch.ones(1, 11, device='cuda')
            lower[:, :unstable] = -1
            upper = torch.ones_like(lower)
            got = refine_stable(lower, upper, compiled, args, 4)
            expected = refine_stable(lower, upper, body, args, 4)
            for a, b in zip(got, expected):
                torch.testing.assert_close(a, b)
        self.assertEqual(counter.frame_count, 1)


if __name__ == '__main__':
    unittest.main()
