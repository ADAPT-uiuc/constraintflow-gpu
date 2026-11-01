"""Selective DeepPoly refinement on final tensor SSA, before Python emission.

All recognition, dependency slicing, interval propagation and row selection use
IR nodes. Rendering is used only for FakeTensor shape inference and code emission,
never parsed or matched. The eager coordinator is shared with the legacy rewrite.
"""
import copy
import io
import operator
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from constraintflow.compiler.ir import (
    IrAssignment, IrConst, IrFConv2d, IrPatchesToDense, IrRowGather,
    IrSelectedPatchesToDense, IrSimpleBinary, IrSimpleUnary, IrTensorClamp,
    IrTorchExpand, IrTorchMatmul, IrTorchPad, IrTorchPermute, IrTorchRepeat, IrTorchReshape,
    IrTorchSlice, IrTorchSqueeze, IrTorchTranspose, IrTorchUnsqueeze, IrTorchView, IrTorchWhere,
    IrTransRetBasic, IrVar,
)
from constraintflow.compiler.optimizations.subexp_inlining import get_vars_expr_occurrences
from constraintflow.compiler.optimizations.sign_reductions import _op
from constraintflow.gbcsr.sparse_block import RUNTIME_HELPERS
from constraintflow.lib import globals
from constraintflow.lib.network import LayerType


def names(expr):
    return {v.name for v in get_vars_expr_occurrences(expr)}


def var(name):
    return IrVar(name, [])


def assign(name, expr):
    return IrAssignment(var(name), expr)


def walk(expr):
    yield expr
    for child in getattr(expr, 'children', ()):
        yield from walk(child)


def relu_pairs(block):
    """Recognize the standard DeepPoly bound guards in tensor IR."""
    def zero(expr):
        return (isinstance(expr, (int, float)) and expr == 0 or
                isinstance(expr, IrConst) and expr.const == 0 or
                _op(expr, 'mul') and any(zero(c) for c in expr.children))
    positive, negative = set(), set()
    for stmt in block.children:
        for expr in walk(stmt):
            if isinstance(expr, IrSimpleBinary):
                a, b = expr.children
                if isinstance(a, IrVar) and zero(b):
                    if _op(expr, 'gt') or _op(expr, 'ge'):
                        positive.add(a.name)
                    if _op(expr, 'lt') or _op(expr, 'le'):
                        negative.add(a.name)
    pairs = []
    for stmt in block.children:
        for expr in walk(stmt):
            if _op(expr, 'sub'):
                u, l = expr.children
                if (isinstance(l, IrVar) and isinstance(u, IrVar)
                        and l.name in positive and u.name in negative
                        and (l.name, u.name) not in pairs):
                    pairs.append((l.name, u.name))
    return pairs


class Shapes:
    """Evaluate IR expressions with fake tensors, as in flow_shapes.probe."""
    def __init__(self, cg, node):
        from torch._subclasses.fake_tensor import FakeTensorMode
        self.cg = cg
        self.mode = FakeTensorMode(allow_non_fake_inputs=True)
        self.env = dict(torch=torch, F=F, operator=operator,
                        device_mode=globals.device_mode, **RUNTIME_HELPERS)
        self.env['batch_size'] = node.flow_batch_size
        with self.mode:
            for name, (shape, dtype) in node.flow_param_meta.items():
                self.env[name] = (torch.empty(shape, device=globals.device_mode.get_device(),
                    dtype={'Bool': torch.bool, 'Int': torch.int64}.get(dtype, torch.float))
                    if shape is not None else 0.)
            for stmt in node.flow_block.children:
                if isinstance(stmt, IrAssignment):
                    self.env[stmt.children[0].name] = self.value(stmt.children[1])
        self.shapes = {n: tuple(v.shape) if isinstance(v, torch.Tensor) else None
                       for n, v in self.env.items()}

    def value(self, expr):
        text = self.cg.visit(expr)
        with self.mode:
            for source, name in self.cg.const_pool.items():
                if name not in self.env:
                    self.env[name] = eval(source, self.env)
            return eval(text, self.env)

    def shape(self, expr):
        value = self.value(expr)
        return tuple(value.shape) if isinstance(value, torch.Tensor) else None


_ELEMENTWISE = (IrSimpleBinary, IrSimpleUnary, IrTensorClamp, IrTorchWhere)
_VIEWS = (IrTorchReshape, IrTorchView, IrTorchExpand)
# Elementwise arithmetic over views: Inductor fuses a row gather into it, so it costs only the selected rows.
_POINTWISE = _ELEMENTWISE + _VIEWS + (IrVar, IrConst, IrTorchSqueeze, IrTorchUnsqueeze, IrTorchPad,
                                      IrTorchSlice, IrTorchPermute, IrTorchTranspose, IrTorchRepeat, int, float)


class SelectRows:
    def __init__(self, probe, n, rows, batch, prefix):
        self.probe, self.n, self.rows, self.batch = probe, n, rows, batch
        self.prefix, self.hoisted = prefix, []

    def is_row(self, expr):
        return bool(names(expr) & self.rows)

    def operand(self, expr, rank):
        shape = self.probe.shape(expr)
        axis = 1 - (rank - len(shape)) if shape is not None else -1
        if shape is None or axis < 0 or axis >= len(shape) or shape[axis] != self.n:
            return expr
        if len(shape) < 2 or shape[0] not in (1, self.batch) or shape[1] != self.n:
            raise ValueError(f'unsupported stable ReLU row operand: {shape}')
        if isinstance(expr, _ELEMENTWISE):
            result = copy.copy(expr)
            result.children = [self.operand(c, rank) for c in expr.children]
            return result
        # Channel weights expanded over spatial positions: gather channels first.
        if isinstance(expr, (IrTorchReshape, IrTorchView)):
            perm = expr.children[0]
            if isinstance(perm, IrTorchPermute) and list(perm.permutation) == [0, 1, 3, 2]:
                rep = perm.children[0]
                if isinstance(rep, (IrTorchRepeat, IrTorchExpand)):
                    dims = rep.repeats if isinstance(rep, IrTorchRepeat) else rep.shape
                    unsq = rep.children[0]
                    if (isinstance(dims, (list, tuple)) and len(dims) == 4
                            and list(dims[:3]) in ([1, 1, 1], [-1, -1, -1])
                            and isinstance(dims[3], int) and dims[3] > 0
                            and isinstance(unsq, IrTorchUnsqueeze) and unsq.index in (-1, 3)):
                        return IrRowGather(unsq.children[0], divisor=dims[3])
        if isinstance(expr, IrPatchesToDense):
            return IrSelectedPatchesToDense(self.operand(expr.children[0], 3), expr.geometry)
        if isinstance(expr, IrTorchRepeat):
            dims = expr.repeats
            if (isinstance(dims, (list, tuple)) and dims[0] == 1
                    and all(d == 1 for d in dims[2:])):
                return IrRowGather(expr.children[0], period=self.probe.shape(expr.children[0])[1])
        #FIX FOR: operands behind squeeze/pad views were computed for every row before the row kernel
        #         (N33's last layer: 452 MiB with 0 unstable rows); gather them inside the kernel instead.
        if all(isinstance(e, _POINTWISE) for e in walk(expr)):
            return IrRowGather(expr)
        name = f'{self.prefix}_{len(self.hoisted)}'
        self.hoisted.append(assign(name, expr))
        return IrRowGather(var(name))

    def visit(self, expr):
        if not self.is_row(expr):
            return expr
        result = copy.copy(expr)
        if isinstance(expr, _VIEWS) and self.is_row(expr.children[0]):
            dims = list(expr.shape)
            if (len(dims) >= 2 and dims[0] in (1, self.batch) and dims[1] == self.n):
                dims[1] = var('_n')
            elif isinstance(expr, (IrTorchReshape, IrTorchView)) and dims[0] == self.batch * self.n:
                dims[0] = IrSimpleBinary(IrConst(self.batch, 'Int'), var('_n'), 'mul')
            else:
                prefix = 1
                for i, dim in enumerate(dims):
                    if not isinstance(dim, int):
                        break
                    if prefix == 1 and dim == self.n:
                        dims[i] = var('_n')
                        break
                    prefix *= dim
            result.shape = dims
        if isinstance(expr, IrPatchesToDense):
            return IrSelectedPatchesToDense(self.visit(expr.children[0]), expr.geometry)
        rank = len(self.probe.shape(expr)) if isinstance(expr, _ELEMENTWISE) else None
        result.children = [self.visit(c) if self.is_row(c) else
                           self.operand(c, rank) if rank is not None else c
                           for c in expr.children]
        return result


def interval(network, target, known):
    body, cache = [], dict(known)
    def get(i):
        if i in cache:
            return cache[i]
        layer = network[i]
        if layer.type == LayerType.Input:
            return 'in_L_const_0', 'in_U_const_0'
        parents = [get(p) for p in layer.parents]
        l, u = map(var, parents[0])
        if layer.type == LayerType.ReLU:
            exprs = [IrTensorClamp(x, 0, True) for x in (l, u)]
        elif layer.type == LayerType.Add:
            exprs = [IrSimpleBinary(var(a), var(b), 'add') for a, b in zip(*parents)]
        elif layer.type in (LayerType.Linear, LayerType.Conv2D):
            w, b = var(f'net_{i}_weight'), var(f'net_{i}_bias')
            wp, wn = IrTensorClamp(w, 0, True), IrTensorClamp(w, 0, False)
            def op(x, weight):
                if layer.type == LayerType.Linear:
                    return IrTorchMatmul(x, IrTorchTranspose(weight, -1, -2))
                shape = [-1, *network[layer.parents[0]].shape[1:]]
                return IrTorchReshape(IrFConv2d(IrTorchReshape(x, shape), weight,
                    list(layer.stride), list(layer.padding)), [var('batch_size'), -1])
            exprs = [IrSimpleBinary(IrSimpleBinary(op(a, wp), op(z, wn), 'add'), b, 'add')
                     for a, z in ((l, u), (u, l))]
        else:
            raise ValueError(f'unsupported stable ReLU interval layer {i}: {layer.type}')
        outputs = (f'_sr_l_{i}', f'_sr_u_{i}')
        body.extend(assign(n, e) for n, e in zip(outputs, exprs))
        cache[i] = outputs
        return outputs
    outputs = get(target)
    return body, outputs


@dataclass
class Region:
    name: str
    body: list
    outputs: list
    params: list
    selected: bool = False


@dataclass
class Call:
    outputs: list
    function: str
    args: list
    # refinement passes a region function and a tuple of its inputs
    refinement: tuple | None = None


def plan(node, network, probe):
    block = node.flow_block
    stmts = block.children
    if (not getattr(node, 'flow_functional', False) or not stmts
            or not isinstance(stmts[-1], IrTransRetBasic)
            or any(not isinstance(s, IrAssignment) or not isinstance(s.children[0], IrVar)
                   for s in stmts[:-1])):
        raise ValueError('stable ReLU requires functional tensor SSA')
    defs = {s.children[0].name: i for i, s in enumerate(stmts[:-1])}
    if len(defs) != len(stmts) - 1:
        raise ValueError('stable ReLU requires unique SSA definitions')
    uses = {n: names(stmts[i].children[1]) & defs.keys() for n, i in defs.items()}
    pairs = relu_pairs(block)
    relus = [l for l in network if l.type == LayerType.ReLU]
    if len(pairs) != len(relus):
        raise ValueError(f'found {len(pairs)} bound pairs for {len(relus)} ReLUs')
    params = {n for n, _ in node.flow_params}
    provenance = {}
    for n, i in defs.items():
        provenance[n] = set().union(*(provenance.get(v, {v} if v in params else set())
                                     for v in names(stmts[i].children[1])))
    def ancestors(i):
        layer = network[i]
        result = set().union(*(ancestors(p) for p in layer.parents))
        if layer.type in (LayerType.Linear, LayerType.Conv2D):
            result |= {f'net_{i}_weight', f'net_{i}_bias'} & params
        return result
    for (l, u), relu in zip(pairs, relus):
        if l not in defs or u not in defs or {v for v in provenance[l] | provenance[u]
                                            if v.startswith('net_')} != ancestors(relu.parents[0]):
            raise ValueError(f'bound provenance does not match ReLU {relu.identifier}')
    emitted, regions, calls = set(), [], []
    known = {0: ('in_L_const_0', 'in_U_const_0')}
    def closure(outputs):
        needed, todo = set(), list(outputs)
        while todo:
            n = todo.pop()
            if n not in emitted and n not in needed and n in defs:
                needed.add(n)
                todo.extend(uses[n])
        return needed
    def region(name, body, outputs, selected=False):
        assigned = {s.children[0].name for s in body}
        read = set(outputs).union(*(names(s.children[1]) for s in body))
        # Traced shape attributes may contain the symbolic batch_size as text.
        args = sorted((read - assigned - {'_idx', '_n'}) | {'batch_size'})
        reg = Region(name, body, list(outputs), (['_idx'] if selected else []) + args, selected)
        regions.append(reg)
        return reg
    batch = node.flow_batch_size
    for k, ((lower, upper), relu) in enumerate(zip(pairs, relus)):
        target = relu.parents[0]
        n = network[target].size
        if probe.shapes[lower] != (batch, n) or probe.shapes[upper] != (batch, n):
            raise ValueError(f'bound shape does not match ReLU {relu.identifier}')
        needed = closure([lower, upper])
        phase = [stmts[defs[v]] for v in sorted(needed, key=defs.get)]
        rows = set()
        for s in phase:
            name = s.children[0].name
            shape = probe.shapes[name]
            if shape and len(shape) >= 3 and shape[0] in (1, batch) and shape[1] == n and shape[2] > 1:
                rows.add(name)
            elif shape and len(shape) >= 2 and shape[0] in (1, batch) and shape[1] == n and names(s.children[1]) & rows:
                rows.add(name)
        future = names(stmts[-1]).union(*(v for name, v in uses.items()
                                        if name not in needed and name not in emitted))
        if (rows & future) - {lower, upper}:
            raise ValueError(f'coefficient tensors escape ReLU {k}')
        can_select = n > 1 and lower in rows and upper in rows
        pre = [s for s in phase if not can_select or s.children[0].name not in rows]
        if can_select and any(names(s.children[1]) & rows for s in pre):
            raise ValueError(f'unsupported coefficient layout at ReLU {k}')
        exported = (needed & future) - rows if can_select else (needed & future) | {lower, upper}
        selected = []
        if can_select:
            coarse_body, coarse = interval(network, target, known)
            pre += coarse_body
            exported |= set(coarse)
            for s in phase:
                if s.children[0].name not in rows:
                    continue
                t = SelectRows(probe, n, rows, batch, f'_sr_hoist_{k}_{len(selected)}')
                expr = s.children[1]
                expr = t.visit(expr) if t.is_row(expr) else t.operand(expr, len(probe.shapes[s.children[0].name]))
                selected.append(assign(s.children[0].name, expr))
                pre += t.hoisted
            selected_region = region(f'_sr_rows_{k}', selected, [lower, upper], True)
            exported |= set(selected_region.params) & {s.children[0].name for s in pre}
        pre_region = region(f'_sr_pre_{k}', pre, sorted(exported))
        calls.append(Call(pre_region.outputs, pre_region.name, pre_region.params))
        if can_select:
            calls.append(Call([lower, upper], 'refine_stable', list(coarse),
                              (selected_region.name, selected_region.params[1:])))
        emitted |= needed
        known[target] = lower, upper
        print(f'[stable-relu-ir] ReLU {k}: {n} neurons, {len(selected)} selected statements')
    rest = [s for s in stmts[:-1] if s.children[0].name not in emitted]
    outputs = [f'_sr_result_{i}' for i in range(len(stmts[-1].children))]
    rest += [assign(n, e) for n, e in zip(outputs, stmts[-1].children)]
    final = region('_sr_final', rest, outputs)
    calls.append(Call([], final.name, final.params))
    return regions, calls


def emit(cg, node, chunk_size, compile_kernels):
    from constraintflow.compiler.optimizations.stable_relu import chunk_sizes
    chunk_sizes(chunk_size)
    if cg.target != 'torch':
        raise ValueError('stable ReLU requires torch')
    import onnx
    from constraintflow.lib.parse import parse_onnx_layers
    network = parse_onnx_layers(onnx.load(globals.network_path), torch.zeros(1, 1),
                               torch.zeros(1), no_sparsity=True)
    regions, calls = plan(node, network, Shapes(cg, node))
    # Build a dense diagnostic module from the same final IR, without source matching.
    dense = ('def flow(' + ', '.join(n for n, _ in node.flow_params) + ', batch_size):\n'
             + cg._render_stmts(node.flow_block.children))
    start = cg.file.tell()
    cg._emit_explode_inputs(list(node.flow_params) + [('batch_size', 'batch_size')])
    cg.indent = 0
    cg.write('from constraintflow.compiler.optimizations.stable_relu import refine_stable, selected_patches_to_dense')
    for reg in regions:
        if compile_kernels:
            cg.write("@torch.compile(fullgraph=True, dynamic=False, options={'layout_optimization': False})")
        cg.write('def ' + reg.name + '(' + ', '.join(reg.params) + '):')
        cg.indent = 1
        if reg.selected:
            cg.write('_n = _idx.numel()')
        last = {n: i for i, s in enumerate(reg.body) for n in names(s.children[1])}
        local = {s.children[0].name for s in reg.body}
        for i, stmt in enumerate(reg.body):
            cg.visit(stmt)
            dead = sorted(n for n in names(stmt.children[1]) & local - set(reg.outputs) if last[n] == i)
            if dead:
                cg.write('del ' + ', '.join(dead))
        cg.write('return (' + ', '.join(reg.outputs) + (',' if reg.outputs else '') + ')')
        cg.indent = 0
        cg.write('')
    cg.write('def flow(' + ', '.join(n for n, _ in node.flow_params) + ', batch_size):')
    cg.indent = 1
    reads = [set(c.args) | (set(c.refinement[1]) if c.refinement else set()) for c in calls]
    last = {n: i for i, ns in enumerate(reads) for n in ns}
    assigned = {n for c in calls for n in c.outputs}
    for i, call in enumerate(calls):
        args = list(call.args)
        if call.refinement:
            name, inputs = call.refinement
            args += [name, '(' + ', '.join(inputs) + (',' if inputs else '') + ')', repr(chunk_size)]
        expr = call.function + '(' + ', '.join(args) + ')'
        if i == len(calls) - 1:
            cg.write('return ' + expr)
        else:
            cg.write((', '.join(call.outputs) + ', = ' if call.outputs else '') + expr)
            dead = sorted(n for n in assigned & reads[i] if last[n] == i)
            if dead:
                cg.write('del ' + ', '.join(dead))
    cg.indent = 0
    cg.write('')
    end = cg.file.tell()
    # Include the matching input interface in the dense diagnostic artifact.
    real_file = cg.file
    cg.file = io.StringIO()
    cg._emit_explode_inputs(list(node.flow_params) + [('batch_size', 'batch_size')])
    dense = cg.file.getvalue() + dense
    cg.file = real_file
    cg._stable_dense_source = start, end, dense
    cg._region_count = len(regions)
