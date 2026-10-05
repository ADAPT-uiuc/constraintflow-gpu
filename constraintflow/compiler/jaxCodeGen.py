"""Lower the final functional tensor IR to a standalone, jax.jit-able flow.

The capture pass and all shared optimization passes still use the Torch shape
probe. Only the final tensor emission changes backend.
"""
import ast
import io
import os
import re

import torch

from .codeGen import CodeGen
from .optimizations import flow_split, region_reuse
from constraintflow.lib.globals import flow_segment_mb


class JaxCodeGen(CodeGen):
    # Inherit only backend-independent renderers. Unknown residual aggregate or
    # mutation nodes must fail at compilation, never leak Torch into a JAX trace.
    _common = {'IrAssignment', 'IrDel', 'IrTransRetBasic', 'IrConst', 'IrVar',
               'IrSimpleBinary', 'IrTorchSlice'}

    def __init__(self):
        self.file = io.StringIO()
        self.indent = 1

    def visit(self, node):
        if isinstance(node, str):
            return self.raw(node)
        if isinstance(node, torch.Tensor):
            dtype = str(node.dtype).split('.')[-1]
            return f'jnp.asarray({node.tolist()!r}, dtype=jnp.{dtype})'
        if isinstance(node, (list, tuple)):
            return '(' + ''.join(self.visit(x) + ', ' for x in node) + ')'
        if node is None or isinstance(node, (bool, int, float)):
            return repr(node)
        name = type(node).__name__
        method = 'visit' + name
        if name not in self._common and method not in type(self).__dict__:
            raise ValueError(f'JAX target cannot lower {name}; requires functional tensor IR')
        return getattr(self, method)(node)

    def raw(self, source):
        """Translate the small expression language used by captured geometry.

        Captures contain shape/index expressions, not full programs. Parse them
        so method arguments and device keywords can be lowered structurally.
        """
        if not source:
            return source
        class Geometry(ast.NodeTransformer):
            def visit_Call(self, node):
                # device is irrelevant to pure JAX geometry; constants are
                # placed with the compiled computation, not with Torch.
                node.keywords = [k for k in node.keywords if k.arg != 'device']
                node = self.generic_visit(node)
                if isinstance(node.func, ast.Attribute):
                    obj = ast.unparse(node.func.value)
                    args = [ast.unparse(a) for a in node.args]
                    method = node.func.attr
                    if method in ('view', 'reshape', 'expand'):
                        shape = args[0] if len(args) == 1 and isinstance(node.args[0], (ast.List, ast.Tuple)) else '(' + ', '.join(args) + ',)'
                        fn = 'jr.expand' if method == 'expand' else 'jnp.reshape'
                        return ast.parse(f'{fn}({obj}, {shape})', mode='eval').body
                    if method == 'unsqueeze':
                        return ast.parse(f'jnp.expand_dims({obj}, {args[0]})', mode='eval').body
                    if obj not in ('jnp', 'jr'):
                        raise ValueError(f'JAX target: unsupported geometry method {method}')
                return node

            def visit_Attribute(self, node):
                if isinstance(node.value, ast.Name) and node.value.id == 'torch':
                    names = {'tensor': 'asarray', 'arange': 'arange', 'int64': 'int32',
                             'long': 'int32', 'int32': 'int32', 'float': 'float32',
                             'float32': 'float32', 'float64': 'float64', 'bool': 'bool_'}
                    if node.attr not in names:
                        raise ValueError(f'JAX target: unsupported geometry operation torch.{node.attr}')
                    return ast.Attribute(value=ast.Name(id='jnp', ctx=ast.Load()), attr=names[node.attr], ctx=node.ctx)
                return self.generic_visit(node)

            def visit_Name(self, node):
                if node.id == 'col2im_columns':
                    return ast.parse('jr.col2im_columns', mode='eval').body
                return node

        try:
            tree = Geometry().visit(ast.parse(source, mode='eval'))
            result = ast.unparse(ast.fix_missing_locations(tree))
        except SyntaxError as exc:
            raise ValueError(f'JAX target: invalid captured expression {source!r}') from exc
        if re.search(r'\b(torch|device_mode|SymExpSparse)\b', result):
            raise ValueError(f'JAX target: unsupported captured expression {source!r}')
        return result

    def render(self, statements, trailer=None):
        self.file = io.StringIO()
        self.indent = 1
        for stmt in statements:
            self.visit(stmt)
        if trailer:
            self.write(trailer)
        return self.file.getvalue()

    def call(self, name, node, *args):
        return name + '(' + ', '.join([self.visit(c) for c in node.children] + list(args)) + ')'

    def visitIrTorchDiagonal(self, n):
        return self.call('jnp.diagonal', n, f'axis1={n.dim1}', f'axis2={n.dim2}')

    def visitIrTorchPermute(self, n):
        return self.call('jnp.transpose', n, repr(tuple(n.permutation)))

    def visitIrTorchTranspose(self, n):
        return self.call('jnp.swapaxes', n, str(n.dim0), str(n.dim1))

    def visitIrTorchMatmul(self, n):
        return self.call('jnp.matmul', n, 'precision=jax.lax.Precision.HIGHEST')

    def visitIrTorchUnsqueeze(self, n):
        return self.call('jnp.expand_dims', n, str(n.index))

    def visitIrTorchSqueeze(self, n):
        return self.call('jr.squeeze', n, str(n.index))

    def visitIrTorchReshape(self, n):
        return self.call('jnp.reshape', n, self.visit(n.shape))

    visitIrTorchView = visitIrTorchReshape

    def visitIrTorchRepeat(self, n):
        return self.call('jnp.tile', n, self.visit(n.repeats))

    def visitIrTorchExpand(self, n):
        return self.call('jr.expand', n, self.visit(n.shape))

    def visitIrTorchPad(self, n):
        return self.call('jr.pad', n, repr(tuple(n.pad)))

    def visitIrTorchSum(self, n):
        dim = tuple(n.dim) if isinstance(n.dim, list) else n.dim
        return self.call('jnp.sum', n, f'axis={dim!r}')

    def visitIrTorchZeros(self, n):
        dtype = self.visit(n.dtype) if n.dtype is not None else 'jnp.float32'
        return f'jnp.zeros({self.visit(n.size)}, dtype={dtype})'

    def visitIrTorchEye(self, n):
        dtype = self.visit(n.dtype) if n.dtype is not None else 'jnp.float32'
        return f'jnp.eye({self.visit(n.size)}, dtype={dtype})'

    def visitIrTorchFloat(self, n):
        return self.call('jnp.asarray', n, 'dtype=jnp.float32')

    visitIrConvertBoolToFloat = visitIrTorchFloat

    def visitIrTorchDiagEmbed(self, n):
        return self.call('jr.diag_embed', n)

    def visitIrPatchesToDense(self, n):
        return self.call('jr.patches_to_dense', n, *map(str, n.geometry))

    def visitIrTensorScatter(self, n):
        return f'jr.scatter({self.visit(n.children[0])}, {n.dim}, {self.raw(n.index)}, {self.visit(n.children[1])})'

    def visitIrFConv2d(self, n):
        return self.call('jr.conv2d', n, 'stride=' + self.visit(n.stride), 'padding=' + self.visit(n.padding))

    def visitIrFConvTranspose2d(self, n):
        args = [key + '=' + self.visit(getattr(n, key)) for key in ('stride', 'padding', 'output_padding')
                if getattr(n, key) is not None]
        return self.call('jr.conv_transpose2d', n, *args)

    def visitIrFUnfold(self, n):
        return self.call('jr.unfold', n, 'kernel_size=' + self.visit(n.kernel_size),
                         'padding=' + self.visit(n.padding), 'stride=' + self.visit(n.stride))

    def visitIrFFold(self, n):
        return self.call('jr.fold', n, 'output_size=' + self.visit(n.output_size),
                         'kernel_size=' + self.visit(n.kernel_size), 'stride=' + self.visit(n.stride),
                         'padding=' + self.visit(n.padding if n.padding is not None else 0))

    def visitIrTorchEinsum(self, n):
        return 'jnp.einsum(' + repr(n.equation) + ', ' + ', '.join(self.visit(c) for c in n.children) + ', precision=jax.lax.Precision.HIGHEST)'

    def visitIrTorchCat(self, n):
        return 'jnp.concatenate([' + ', '.join(self.visit(c) for c in n.children) + f'], axis={n.dim})'

    def visitIrTorchWhere(self, n):
        return self.call('jnp.where', n)

    def visitIrSimpleUnary(self, n):
        ops = {'-': 'jnp.negative', 'not': 'jnp.logical_not', 'sigma': 'jax.nn.sigmoid'}
        if n.op not in ops:
            raise ValueError(f'JAX target: unsupported unary operator {n.op!r}')
        return self.call(ops[n.op], n)

    def visitIrTensorOnes(self, n):
        shape = self.raw(n.total_size) if isinstance(n.total_size, str) else repr(tuple(n.total_size.tolist()))
        return f'jnp.ones({shape}, dtype=jnp.float32)'

    def visitIrTensorRepeat(self, n):
        return self.call('jnp.tile', n, self.visit(n.repeat_dims))

    def visitIrTensorClamp(self, n):
        return self.call('jnp.maximum' if n.min_true else 'jnp.minimum', n, repr(n.const))

    def renderTraceIndex(self, index):
        parts = []
        for item in index:
            if isinstance(item, str) and ':' in item:
                # Product-grid extraction uses textual Python slices. Their
                # endpoints are expressions, but the slice itself is not one.
                fields = item.split(':')
                if len(fields) not in (2, 3):
                    raise ValueError(f'JAX target: invalid slice {item!r}')
                parts.append(':'.join(self.raw(x.strip()) if x.strip() else '' for x in fields))
            elif isinstance(item, list) and len(item) == 2:
                parts.append(':'.join('' if x is None else self.visit(x) for x in item))
            elif isinstance(item, int) and item == 0:
                parts.append(':')
            else:
                parts.append(self.visit(item))
        return ', '.join(parts)


def emit_flow(cg, node, sized, barriers):
    if not getattr(node, 'flow_functional', False):
        raise ValueError('JAX target requires functional SROA; residual aggregate/view mutations remain')
    if node.flow_block.inner_jump is not None or node.flow_block.jump is not None:
        raise ValueError('JAX target requires a straight-line specialized flow')
    emitter = JaxCodeGen()
    segments = None
    if flow_segment_mb.get_value() > 0:
        segments = flow_split.split(node.flow_block, sized,
                                    max_region_bytes=flow_segment_mb.get_value() * 1024 ** 2,
                                    barriers=barriers)
    params = list(node.flow_params)
    bodies = []
    for seg in segments or [None]:
        stmts = seg.stmts if seg else node.flow_block.children
        trailer = 'return (' + ''.join(n + ', ' for n in seg.live_out) + ')' if seg and seg is not segments[-1] else None
        body = emitter.render(stmts, trailer)
        names = list(seg.live_in) if seg else [name for name, _ in params]
        if re.search(r'\bbatch_size\b', body) and 'batch_size' not in names:
            names.append('batch_size')
        bodies.append((seg.name if seg else 'flow', names, body, seg))
    if any('batch_size' in names for _, names, _, _ in bodies) and not any(n == 'batch_size' for n, _ in params):
        params.append(('batch_size', 'batch_size'))
    source = ['import operator\nimport jax\nimport jax.numpy as jnp\nfrom functools import partial\n'
              'from constraintflow.lib import jax_runtime as jr\n\n']
    shapes = dict(getattr(sized, 'inputs', {}))
    shapes.update({v['name']: v for v in (sized or {}).values() if 'shape' in v})
    reused, distinct = {}, 0
    for name, names, body, seg in bodies:
        signature = region_reuse.key(body, names, shapes) if seg else None
        if signature is not None and signature in reused:
            source.append(name + ' = ' + reused[signature] + '\n\n')
            continue
        if signature is not None:
            reused[signature] = name
        distinct += 1
        static = "('batch_size',)" if 'batch_size' in names else '()'
        source.append(f'@partial(jax.jit, static_argnames={static})\ndef {name}(' + ', '.join(names) + '):\n' + body + '\n')
    if segments:
        source.append('def flow(' + ', '.join(n for n, _ in params) + '):\n')
        last_use = {n: i for i, (_, names, _, _) in enumerate(bodies) for n in names}
        for i, (name, names, _, seg) in enumerate(bodies):
            call = name + '(' + ', '.join(names) + ')'
            if seg is segments[-1]:
                source.append('\treturn ' + call + '\n')
            elif seg.live_out:
                source.append('\t' + ', '.join(seg.live_out) + ', = ' + call + '\n')
            else:
                source.append('\t' + call + '\n')
            if seg is not segments[-1]:
                dead = [n for n in names if last_use[n] == i and n not in seg.live_out]
                if dead:
                    source.append('\tdel ' + ', '.join(dead) + '\n')
    source = ''.join(source)
    # Detect residual backend names before publishing a runnable artifact.
    if re.search(r'\b(torch|F|SymExpSparse)\b', source):
        raise ValueError('JAX target: residual Torch expression in emitted flow')
    compile(source, 'jax_flow.py', 'exec')
    with open(os.path.join(cg.folder, 'jax_flow.py'), 'w') as f:
        f.write(source)
    cg._region_count = distinct
    cg._flow_metrics['host_offload'] = False
    print(f'[jax] {distinct} compiled regions; CUDA stream host offload is not used')
    cg._emit_explode_inputs(params)
    cg.write('from jax_flow import flow as jax_flow')
    cg.write('from constraintflow.lib.jax_bridge import run_flow')
    arguments = ', '.join(name for name, _ in params)
    cg.write('def flow(' + arguments + '):')
    cg.indent += 1
    cg.write('return run_flow(jax_flow, ' + arguments + ')')
    cg.indent -= 1
