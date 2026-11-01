"""Selective DeepPoly refinement on final tensor SSA, before Python emission.

All recognition, dependency slicing, interval propagation and row selection use
IR nodes. Rendering is used only for FakeTensor shape inference and code emission,
never parsed or matched. The eager coordinator is shared with the legacy rewrite.
"""
import torch

from constraintflow.compiler.ir import (
    IrAssignment, IrConst, IrFConv2d, IrSimpleBinary, IrTensorClamp,
    IrTorchMatmul, IrTorchReshape, IrTorchTranspose, IrTransRetBasic, IrVar,
)
from constraintflow.compiler.optimizations.sign_reductions import _op
from constraintflow.lib import globals
from constraintflow.lib.network import LayerType
from constraintflow.compiler.optimizations.row_regions import (
    names, var, assign, walk, Shapes, SelectRows, Region, Call, emit_regions,
)


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
    return emit_regions(cg, node, regions, calls, chunk_size, compile_kernels)
