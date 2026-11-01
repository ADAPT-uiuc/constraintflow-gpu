"""IR-only regression coverage; no generated-Python pattern matching."""
import operator
import unittest
from types import SimpleNamespace

import torch

from constraintflow.compiler.codeGen import CodeGen
from constraintflow.compiler.ir import (
    IrBlock, IrConst, IrSimpleBinary, IrTorchExpand, IrTorchPermute,
    IrTorchMatmul, IrTorchRepeat, IrTorchReshape, IrTorchSqueeze, IrTorchUnsqueeze, IrTransRetBasic,
)
from constraintflow.compiler.optimizations.stable_relu_ir import (
    SelectRows, assign, plan, relu_pairs, var,
)


class Probe:
    def __init__(self, env):
        self.env = dict(env, torch=torch, operator=operator)
        self.cg = CodeGen.__new__(CodeGen)
        self.cg.const_pool = {}

    def value(self, expr, **extra):
        return eval(self.cg.visit(expr), dict(self.env, **extra))

    def shape(self, expr):
        value = self.value(expr)
        return tuple(value.shape) if isinstance(value, torch.Tensor) else None


class StableReLUIRTest(unittest.TestCase):
    def test_guard_recognition_accepts_ir_and_native_scalar_zeros(self):
        for zero in (0., IrConst(0., 'Float')):
            block = IrBlock([])
            block.children = [
                assign('width', IrSimpleBinary(var('upper'), var('lower'), 'sub')),
                assign('positive', IrSimpleBinary(var('lower'), zero, 'gt')),
                assign('negative', IrSimpleBinary(var('upper'),
                    IrSimpleBinary(var('ones'), zero, 'mul'), 'le')),
            ]
            self.assertEqual(relu_pairs(block), [('lower', 'upper')])
            block.children.pop()
            self.assertEqual(relu_pairs(block), [])

    def test_channel_gather_precedes_spatial_materialization(self):
        weights = torch.randn(1, 3, 5)
        probe = Probe({'weights': weights})
        for repeat in (IrTorchRepeat(IrTorchUnsqueeze(var('weights'), -1), [1, 1, 1, 4]),
                       IrTorchExpand(IrTorchUnsqueeze(var('weights'), -1), [-1, -1, -1, 4])):
            expr = IrTorchReshape(IrTorchPermute(repeat, [0, 1, 3, 2]), [1, 12, -1])
            selector = SelectRows(probe, 12, set(), 1, '_hoist')
            selected = selector.operand(expr, 3)
            idx = torch.tensor([11, 0, 5, 5])
            torch.testing.assert_close(probe.value(selected, _idx=idx),
                                       probe.value(expr).index_select(1, idx))
            self.assertEqual(selector.hoisted, [])

    def test_batch_equal_to_neurons_and_folded_convolution_batch(self):
        x = torch.randn(10, 10, 6)
        probe = Probe({'x': x})
        selector = SelectRows(probe, 10, {'x'}, 10, '_hoist')
        idx = torch.tensor([9, 2, 2])
        for shape in ([10, 10, -1], [100, 6]):
            expr = IrTorchReshape(var('x'), shape)
            selected = selector.visit(expr)
            got = probe.value(selected, x=x.index_select(1, idx), _n=len(idx))
            expected = x.index_select(1, idx)
            if len(shape) == 2:
                expected = expected.reshape(30, 6)
            torch.testing.assert_close(got, expected)
            self.assertEqual(expr.shape, shape, 'the original dense IR must remain unchanged')

    def test_pointwise_operand_behind_views_is_gathered_in_kernel(self):
        # N33's last layer: W * slope behind squeeze/unsqueeze was computed for all rows before the kernel.
        weights, slope, other = torch.randn(1, 6, 5), torch.randn(1, 1, 5), torch.randn(5, 6)
        probe = Probe({'weights': weights, 'slope': slope, 'other': other})
        selector = SelectRows(probe, 6, set(), 1, '_hoist')
        expr = IrTorchSqueeze(IrTorchUnsqueeze(IrSimpleBinary(var('weights'), var('slope'), 'mul'), 3), 3)
        selected = selector.operand(expr, 3)
        idx = torch.tensor([5, 0, 3, 3])
        torch.testing.assert_close(probe.value(selected, _idx=idx), probe.value(expr).index_select(1, idx))
        self.assertEqual(selector.hoisted, [])
        # A matrix product is not pointwise: it is still computed once before the kernel.
        selector.operand(IrTorchMatmul(var('weights'), var('other')), 3)
        self.assertEqual(len(selector.hoisted), 1)

    def test_repeated_row_gather_preserves_noncontiguous_order(self):
        x = torch.randn(2, 3, 5)
        probe = Probe({'x': x})
        expr = IrTorchRepeat(var('x'), [1, 4, 1])
        selector = SelectRows(probe, 12, set(), 2, '_hoist')
        selected = selector.operand(expr, 3)
        idx = torch.tensor([11, 0, 7, 7])
        torch.testing.assert_close(probe.value(selected, _idx=idx),
                                   probe.value(expr).index_select(1, idx))
        self.assertEqual(selector.hoisted, [])

    def test_duplicate_ssa_is_rejected(self):
        block = IrBlock([])
        block.children = [assign('x', IrConst(1, 'Int')), assign('x', IrConst(2, 'Int')),
                          IrTransRetBasic([var('x')])]
        node = SimpleNamespace(flow_block=block, flow_functional=True)
        with self.assertRaisesRegex(ValueError, 'unique SSA'):
            plan(node, [], None)


if __name__ == '__main__':
    unittest.main()
