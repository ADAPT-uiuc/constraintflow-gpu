import ast
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from constraintflow.compiler.optimizations.stable_relu import refine_stable, selected_patches_to_dense, flatten_flow, relu_pairs, warmup_stable_relu
from constraintflow.gbcsr.sparse_block import patches_to_dense


class StableReLUTest(unittest.TestCase):
    def test_expanded_and_repeated_weights_gather_before_materialization(self):
        from constraintflow.compiler.optimizations.stable_relu import _SelectedRows
        from constraintflow.compiler.optimizations.row_chunks import _Probe
        weights = torch.randn(1, 3, 5)
        indices = torch.tensor([11, 0, 5, 5])
        for method, shape in [('repeat', '[1, 1, 1, 4]'), ('expand', '[-1, -1, -1, 4]')]:
            expression = f'weights.unsqueeze(-1).{method}({shape}).permute(0, 1, 3, 2).reshape([1, 12, -1])'
            probe = _Probe({}, ['weights'], [weights])
            transform = _SelectedRows(probe, 12, set(), set())
            selected = transform._row_operand(ast.parse(expression, mode='eval').body, 3)
            self.assertFalse(transform.hoisted, 'must not hoist dense expanded weights')
            actual = eval(compile(ast.Expression(ast.fix_missing_locations(selected)), '<test>', 'eval'),
                          {'weights': weights, '_idx': indices})
            expected = eval(expression, {'weights': weights}).index_select(1, indices)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_batch_equal_to_row_count_keeps_batch_literal(self):
        # N18 at batch 10: a 10-neuron layer's [batch, rows, -1] reshape became [_n, _n, -1].
        from constraintflow.compiler.optimizations.stable_relu import _SelectedRows
        from constraintflow.compiler.optimizations.row_chunks import _Probe
        x = torch.randn(10, 10, 6)
        probe = _Probe({}, ['x'], [x])
        transform = _SelectedRows(probe, 10, {'x'}, set(), batch_size=10)
        for expression, expected in [('x.reshape([10, 10, -1])', 'x.reshape([10, _n, -1])'),
                                     ('x.reshape([100, 6])', 'x.reshape([10 * _n, 6])')]:
            rewritten = transform.visit(ast.parse(expression, mode='eval').body)
            self.assertEqual(ast.unparse(rewritten), expected)

    def test_driver_releases_tuple_outputs_at_last_use(self):
        import weakref
        from constraintflow.compiler.optimizations.stable_relu import _release_driver_tensors
        refs = []
        def make():
            x = torch.ones(3)
            refs.append(weakref.ref(x))
            return x, torch.ones(3)
        def consume(x, y):
            return x + y, y
        def check(x, y):
            self.assertIsNone(refs[0](), 'coordinator retained a dead region output')
            return x + y
        fn = ast.parse('def flow():\n a, b = make()\n c, d = consume(a, b)\n return check(c, d)').body[0]
        fn.body = _release_driver_tensors(fn.body)
        env = dict(make=make, consume=consume, check=check)
        exec(compile(ast.fix_missing_locations(ast.Module([fn], [])), '<test>', 'exec'), env)
        torch.testing.assert_close(env['flow'](), torch.full((3,), 3.))

    def test_batch_union_selects_neurons_unstable_in_any_example(self):
        lower = torch.tensor([[1., -1., 1., 1.], [1., 1., -2., 1.], [1., 1., 1., 1.]])
        upper = torch.full_like(lower, 3.)
        calls = []
        offsets = torch.arange(3.)[:, None] * 10
        def kernel(idx):
            calls.append(idx.tolist())
            return offsets + idx[None], offsets + idx[None] + 1
        got = refine_stable(lower, upper, kernel, (), 3)
        self.assertEqual(calls, [[1, 2, 2]])
        # Selected rows take the refined bounds for every example, stable or not.
        torch.testing.assert_close(got[0], torch.tensor([[1., 1., 2., 1.], [1., 11., 12., 1.], [1., 21., 22., 1.]]))
        torch.testing.assert_close(got[0][:, [0, 3]], lower[:, [0, 3]])
        calls.clear()
        refine_stable(torch.ones_like(lower), upper, kernel, (), 3)
        self.assertFalse(calls)
        warmup_stable_relu(refine_stable, torch.ones_like(lower), upper, kernel, (), 3)
        self.assertEqual(calls, [[0, 1, 2]])

    def test_several_chunk_sizes_pick_smallest_cover_and_cap_at_width(self):
        def kernel(idx):
            calls.append(idx.numel())
            return idx[None].float(), idx[None].float() + 1
        n = 300
        upper = torch.ones(1, n)
        for unstable, expected in [(1, [8]), (44, [128]), (300, [128, 128, 128]), (129, [128, 8])]:
            calls = []
            lower = torch.ones(1, n)
            lower[0, :unstable] = -1
            lo, hi = refine_stable(lower, upper, kernel, (), (8, 32, 128))
            self.assertEqual(calls, expected)
            torch.testing.assert_close(lo[0, :unstable], torch.arange(unstable).float())
            torch.testing.assert_close(lo[0, unstable:], lower[0, unstable:])
        # Sizes wider than the layer collapse to the width, and warmup compiles every size.
        calls = []
        warmup_stable_relu(refine_stable, torch.ones(1, 50), torch.ones(1, 50), kernel, (), (8, 32, 128, 512))
        self.assertEqual(calls, [8, 32, 50])

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

    def test_warmup_exercises_empty_selection_without_changing_bounds(self):
        lower = torch.tensor([[1., -2., 0.]])
        upper = torch.tensor([[2., -1., 0.]])
        calls = []
        def kernel(idx):
            calls.append(idx.clone())
            return torch.full((1, idx.numel()), float('nan')), torch.zeros(1, idx.numel())
        got = warmup_stable_relu(refine_stable, lower, upper, kernel, (), 4)
        self.assertEqual(calls[0].tolist(), [0, 1, 2, 0])
        for actual, expected in zip(got, (lower, upper)):
            torch.testing.assert_close(actual, expected)
        refine_stable(lower, upper, kernel, (), 4)
        self.assertEqual(len(calls), 1, 'normal all-stable calls must still skip the kernel')
        def fails(idx):
            raise RuntimeError('compile failed')
        with self.assertRaisesRegex(RuntimeError, 'compile failed'):
            warmup_stable_relu(refine_stable, lower, upper, fails, (), 4)
        refine_stable(lower, upper, fails, (), 4)  # warmup flag was reset

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
        for dims in [(3, 3, 4, 5, 2, 3, 2, 7, 8, 1, 2, 2, 1),
                     (1, 3, 4, 5, 2, 3, 2, 7, 8, 1, 2, 2, 1),
                     (1, 2, 3, 3, 2, 5, 5, 4, 4, 2, 2, 1, 1)]:
            b, oc, ox, oy, ic, kx, ky, *_ = dims
            pieces = torch.randn(b, oc * ox * oy, ic * kx * ky, dtype=torch.float64)
            indices = torch.tensor([oc * ox * oy - 1, 0, oy + 1, 0])
            got = selected_patches_to_dense(pieces.index_select(1, indices), indices, *dims)
            expected = patches_to_dense(pieces, *dims).index_select(1, indices)
            torch.testing.assert_close(got, expected, atol=0, rtol=0)

    def test_cli_warmup_protocols_exercise_skipped_kernel(self):
        # Execute each real CLI protocol with a tiny stand-in generated program.
        # All calls have stable bounds: only an explicit warmup may invoke kernel.
        from unittest.mock import patch
        from constraintflow import cli
        calls = []
        stable = torch.ones(1, 3)
        def kernel(idx):
            calls.append(idx.tolist())
            return stable, stable  # discarded dummy outputs
        def run(*args, **kwargs):
            return refine_stable(stable, stable, kernel, (), 4)
        def batch(index):
            return stable, torch.zeros(1, dtype=torch.long)
        with tempfile.TemporaryDirectory() as tmp:
            cli._run_image_batches(run, batch, 'unused', 1, 0., 2, 1,
                                   str(Path(tmp, 'images.jsonl')), False, False,
                                   'box-3-2-2', False, False, False, False)
            self.assertEqual(calls, [[0, 1, 2, 0]])
            calls.clear()
            # Exercise the repeat-mode command through Typer so defaults are
            # actual values, not the OptionInfo objects of a direct Python call.
            from typer.testing import CliRunner
            from types import SimpleNamespace
            with patch.dict(sys.modules, {'main': SimpleNamespace(run=run)}), \
                 patch.object(cli, 'get_dataset', return_value=(stable.repeat(2, 1), torch.zeros(2, dtype=torch.long))):
                result = CliRunner().invoke(cli.app, ['run', 'unused.cf',
                    '--network', 'unused.onnx', '--dataset', 'box-3-2-2',
                    '--device', 'cpu', '--warmup', '1', '--repeat', '2',
                    '--output-path', tmp])
            import gc
            gc.unfreeze()  # run froze this test process's heap after its warmup
            self.assertEqual(result.exit_code, 0, result.output + repr(result.exception))
            self.assertEqual(calls, [[0, 1, 2, 0]])

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
        torch.manual_seed(3)
        model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(),
                                    torch.nn.Linear(8, 6), torch.nn.ReLU(), torch.nn.Linear(6, 2)).eval()
        self._check_batched_jit(model, (4,), batch_size=1)
        self._check_batched_jit(model, (4,), batch_size=3)

    def test_batched_convolution_jit_matches_dense(self):
        torch.manual_seed(4)
        model = torch.nn.Sequential(torch.nn.Conv2d(1, 2, 3, padding=1), torch.nn.ReLU(),
                                    torch.nn.Conv2d(2, 2, 3, padding=1), torch.nn.ReLU(),
                                    torch.nn.Flatten(), torch.nn.Linear(40, 2)).eval()
        self._check_batched_jit(model, (1, 4, 5))

    def _check_batched_jit(self, model, input_shape, batch_size=3):
        dataset = 'box-' + 'x'.join(map(str, input_shape)) + '-2-2'
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            onnx = str(Path(tmp, 'model.onnx'))
            torch.onnx.export(model, torch.zeros(1, *input_shape), onnx, dynamo=False)
            folder = str(Path(tmp, 'program'))
            env = dict(os.environ, PYTHONPATH=str(root))
            result = subprocess.run([sys.executable, '-m', 'constraintflow.cli', 'jit',
                str(root / 'examples/compiler_examples/deeppoly_priority_add.cf'),
                '--network', onnx, '--dataset', dataset, '--device', 'cpu',
                '--batch-size', str(batch_size), '--stable-relu', '--stable-relu-chunk', '4', '--fuse-affine-subst',
                '--output-path', folder], env=env, cwd=tmp, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('[stable-relu-ir]', result.stdout)
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
def refine_all(lower, upper, kernel, args, chunk):
    pieces = [kernel(torch.arange(i, min(i+chunk, lower.shape[1])), *args)
              for i in range(0, lower.shape[1], chunk)]
    return tuple(torch.cat([part[j] for part in pieces], dim=1) for j in (0, 1))
ordinary_refine = transformers.refine_stable
batch = int(sys.argv[5])
for width in (0., 0.01, 0.5, 10.):
    shape = tuple(map(int, sys.argv[3].split(',')))
    import math
    x = torch.linspace(-2., 3., batch * math.prod(shape)).reshape(batch, *shape)
    net, ae = main.build(sys.argv[2], batch, 0, x, torch.zeros(batch, dtype=torch.long),
                        sys.argv[4], False, False, False, input_box=(x-width, x+width))
    net[-1].weight = torch.eye(2).unsqueeze(0)
    args = main.explode_inputs(ae, batch)
    # Compare all chunked rows with dense arithmetic, independent of whether
    # interval propagation proves a tighter stability sign than dense DeepPoly.
    transformers.refine_stable = refine_all
    for got, expected in zip(main.flow(*args), env['flow'](*args)):
        torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)
    transformers.refine_stable = ordinary_refine
    # An example's bounds may depend on its batch: rows are selected by the union
    # over the batch and refined for every example, as in auto_LiRPA --sparse.
"""
            result = subprocess.run([sys.executable, '-c', check, folder, onnx, ','.join(map(str, input_shape)), dataset, str(batch_size)],
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
        args = weight, -torch.ones(3, 7, device='cuda'), torch.ones(3, 7, device='cuda')
        # Reproduce N42: warmup is all-stable, a later image first needs this
        # kernel. Also ensure it compiles even if no subsequent image needs it.
        stable = torch.ones(3, 11, device='cuda')
        warmed = warmup_stable_relu(refine_stable, stable, stable, compiled, args, 4)
        self.assertEqual(counter.frame_count, 1)
        for actual in warmed:
            torch.testing.assert_close(actual, stable)
        for unstable in (0, 1, 7, 11, 0, 3):
            lower = torch.ones(3, 11, device='cuda')
            lower[0, :unstable] = -1
            lower[1, -unstable:] = -1 if unstable else 1
            upper = torch.ones_like(lower)
            got = refine_stable(lower, upper, compiled, args, 4)
            expected = refine_stable(lower, upper, body, args, 4)
            for a, b in zip(got, expected):
                torch.testing.assert_close(a, b)
        self.assertEqual(counter.frame_count, 1)


if __name__ == '__main__':
    unittest.main()
