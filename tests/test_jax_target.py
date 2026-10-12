"""CPU numerical regression tests; run with the optional [jax] dependency."""
import importlib.util
import operator
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

HAS_JAX = importlib.util.find_spec('jax') is not None


@unittest.skipUnless(HAS_JAX, 'install constraintflow[jax]')
class JaxTargetTest(unittest.TestCase):
    def setUp(self):
        import jax
        from constraintflow.lib import jax_runtime
        self.jax = jax
        self.runtime = jax_runtime
        self.rng = np.random.default_rng(7)

    def array(self, shape):
        return self.rng.normal(size=shape).astype(np.float32)

    def compare(self, jax_fn, torch_fn, *args):
        expected = torch_fn(*[torch.from_numpy(x.copy()) for x in args]).numpy()
        for fn in (jax_fn, self.jax.jit(jax_fn)):
            actual = np.asarray(fn(*[self.jax.numpy.asarray(x) for x in args]))
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)

    def test_convolution_and_transpose_rectangular(self):
        for stride, padding, output_padding in [((1, 1), (0, 0), (0, 0)), ((2, 3), (1, 0), (1, 2))]:
            with self.subTest(stride=stride):
                x, w = self.array((2, 3, 7, 9)), self.array((4, 3, 3, 2))
                self.compare(lambda x, w: self.runtime.conv2d(x, w, stride, padding),
                             lambda x, w: F.conv2d(x, w, stride=stride, padding=padding), x, w)
                w = self.array((3, 4, 3, 2))
                self.compare(lambda x, w: self.runtime.conv_transpose2d(x, w, stride, padding, output_padding),
                             lambda x, w: F.conv_transpose2d(x, w, stride=stride, padding=padding, output_padding=output_padding), x, w)

    def test_unfold_and_fold_overlap(self):
        x = self.array((2, 3, 7, 9))
        for kernel, stride, padding in [((3, 2), (1, 1), (1, 0)), ((2, 3), (2, 2), (0, 1))]:
            with self.subTest(kernel=kernel):
                self.compare(lambda x: self.runtime.unfold(x, kernel, padding, stride),
                             lambda x: F.unfold(x, kernel, padding=padding, stride=stride), x)
                columns = F.unfold(torch.from_numpy(x), kernel, padding=padding, stride=stride).numpy()
                self.compare(lambda x: self.runtime.fold(x, (7, 9), kernel, stride, padding),
                             lambda x: F.fold(x, (7, 9), kernel, stride=stride, padding=padding), columns)

    def test_patches_to_dense(self):
        from constraintflow.gbcsr.sparse_block import patches_to_dense
        geometry = (2, 3, 3, 4, 2, 3, 2, 5, 8, 1, 0, 2, 2)
        pieces = self.array((1, 3, 12, 2, 6))
        self.compare(lambda x: self.runtime.patches_to_dense(x, *geometry),
                     lambda x: patches_to_dense(x, *geometry), pieces)

    def test_scatter_and_padding(self):
        index = np.array([[[0, 2], [1, 3], [0, 1]]], dtype=np.int32)
        source, dest = self.array((1, 3, 2)), self.array((1, 3, 4))
        self.compare(lambda x, s: self.runtime.scatter(x, -1, index, s),
                     lambda x, s: x.scatter(-1, torch.from_numpy(index).long(), s), dest, source)
        self.compare(lambda x: self.runtime.pad(x, (-1, 2, 1, 0)),
                     lambda x: F.pad(x, (-1, 2, 1, 0)), dest)

    def test_diagonal_and_torch_shape_semantics(self):
        self.compare(self.runtime.diag_embed, torch.diag_embed,
                     np.array([[1, np.inf, np.nan]], dtype=np.float32))
        self.compare(lambda x: self.runtime.squeeze(x, 1), lambda x: x.squeeze(1), self.array((1, 3)))
        self.compare(lambda x: self.runtime.expand(x, (2, -1, 3)),
                     lambda x: x.expand(2, -1, 3), self.array((1, 3)))

    def test_emitted_tensor_ir(self):
        import jax.numpy as jnp
        from constraintflow.compiler import ir
        from constraintflow.compiler.jaxCodeGen import JaxCodeGen
        x = ir.IrVar('x', [])
        w = ir.IrVar('w', [])
        # Shapes and views must be static inside jax.jit.
        expression = ir.IrTorchMatmul(ir.IrTensorClamp(x, 0, True), w)
        source = JaxCodeGen().visit(expression)
        namespace = {'jax': self.jax, 'jnp': jnp, 'jr': self.runtime, 'operator': operator}
        fn = eval('lambda x, w: ' + source, namespace)
        self.compare(fn, lambda x, w: x.clamp(min=0) @ w, self.array((2, 3)), self.array((3, 4)))
        with self.assertRaisesRegex(ValueError, 'cannot lower'):
            JaxCodeGen().visit(ir.IrTorchStride(x))

    def test_captured_scatter_geometry(self):
        import jax.numpy as jnp
        from constraintflow.compiler.jaxCodeGen import JaxCodeGen
        source = 'col2im_columns(2, 3, 2, 2, 4, 5, 1, device=device_mode.get_device()).view(1, 1, 6, 1, 4).expand(2, 3, 6, 2, 4)'
        translated = JaxCodeGen().raw(source)
        fn = eval('lambda: ' + translated, {'jr': self.runtime, 'jnp': jnp})
        from constraintflow.gbcsr.sparse_block import col2im_columns
        expected = col2im_columns(2, 3, 2, 2, 4, 5, 1).view(1, 1, 6, 1, 4).expand(2, 3, 6, 2, 4)
        np.testing.assert_array_equal(np.asarray(self.jax.jit(fn)()), expected.numpy())

    def test_product_grid_slices(self):
        import jax.numpy as jnp
        from constraintflow.compiler import ir
        from constraintflow.compiler.jaxCodeGen import JaxCodeGen
        expression = ir.IrTorchSlice(ir.IrVar('x', []), ['0:1', '1::2'])
        source = JaxCodeGen().visit(expression)
        fn = eval('lambda x: ' + source, {'jnp': jnp})
        self.compare(fn, lambda x: x[0:1, 1::2], self.array((3, 6)))

    def test_generated_flow_and_torch_bridge(self):
        from constraintflow.compiler import ir
        from constraintflow.compiler.codeGen import CodeGen
        from constraintflow.compiler.jaxCodeGen import emit_flow
        from constraintflow.lib.jax_bridge import run_flow
        x = ir.IrVar('x', [])
        node = SimpleNamespace(flow_functional=True, flow_params=[('x', "abs_elem.d['x']")],
                               flow_block=ir.IrBlock([ir.IrTransRetBasic([x, ir.IrTensorClamp(x, 0, True)])]))
        with tempfile.TemporaryDirectory() as folder:
            cg = CodeGen(folder, target='jax')
            cg._flow_metrics = {}
            cg.open(cg.transformers_file)
            emit_flow(cg, node, None, set())
            cg.file.close()
            namespace = {}
            with open(folder + '/jax_flow.py') as f:
                exec(compile(f.read(), 'jax_flow.py', 'exec'), namespace)
            value = torch.tensor([[-2.0, 3.0]])
            output = run_flow(namespace['flow'], value)
            torch.testing.assert_close(output[0], value)
            torch.testing.assert_close(output[1], value.clamp(min=0))


if __name__ == '__main__':
    unittest.main()
