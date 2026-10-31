"""Selective DeepPoly refinement of pre-ReLU bounds (batch one).

The coordinator is eager: discovering a variable number of unstable neurons needs
one device synchronization per layer. Numerical regions use fullgraph torch.compile
and padded, fixed-size row chunks; masks are recomputed for every input box.
"""
import ast
import copy
from contextvars import ContextVar

import torch

from constraintflow.compiler.optimizations.row_chunks import _Probe, _ChunkBody, _names, _add_dels
from constraintflow.lib.network import LayerType


def flatten_flow(source):
    """Inline generated regions, alpha-renaming reused SSA bodies at each call."""
    tree = ast.parse(source)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Name) and stmt.value.id in functions:
            functions[stmt.targets[0].id] = functions[stmt.value.id]
    fn = functions['flow']
    body = []
    class Rename(ast.NodeTransformer):
        def __init__(self, names):
            self.names = names
        def visit_Name(self, node):
            return ast.copy_location(ast.Name(self.names.get(node.id, node.id), node.ctx), node)
    for serial, stmt in enumerate(fn.body):
        if isinstance(stmt, ast.Delete):
            continue
        call = stmt.value if isinstance(stmt, (ast.Assign, ast.Return)) else None
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id.startswith('flow_seg_'):
            seg = functions[call.func.id]
            if call.keywords or len(call.args) != len(seg.args.args) or not all(isinstance(a, ast.Name) for a in call.args):
                raise ValueError('stable ReLU requires positional SSA regions')
            names = {a.arg: b.id for a, b in zip(seg.args.args, call.args)}
            ret = seg.body[-1]
            if not isinstance(ret, ast.Return):
                raise ValueError('segment has no return')
            if isinstance(stmt, ast.Assign):
                if not isinstance(ret.value, ast.Tuple) or not all(isinstance(n, ast.Name) for n in ret.value.elts):
                    raise ValueError('region returns must be named SSA tensors')
                names.update({a.id: b.id for a, b in zip(ret.value.elts, stmt.targets[0].elts)})
            # Preserve original names unless this is an aliased/reused region.
            if call.func.id != seg.name or any(a.arg != b.id for a, b in zip(seg.args.args, call.args)):
                for sub in seg.body:
                    if isinstance(sub, ast.Assign):
                        for n in sub.targets:
                            if isinstance(n, ast.Name):
                                names.setdefault(n.id, f'_sr_inline_{serial}_{n.id}')
            rename = Rename(names)
            body.extend(rename.visit(copy.deepcopy(s)) for s in seg.body[:-1] if not isinstance(s, ast.Delete))
            if isinstance(stmt, ast.Return):
                body.append(rename.visit(copy.deepcopy(ret)))
        else:
            body.append(copy.deepcopy(stmt))
    fn.body, fn.decorator_list = body, []
    tree.body = [s for s in tree.body
                 if not (isinstance(s, ast.FunctionDef) and s.name.startswith('flow_seg_'))
                 and not (isinstance(s, ast.Assign) and any(isinstance(t, ast.Name) and t.id.startswith('flow_seg_') for t in s.targets))]
    return tree, fn


def relu_pairs(stmts):
    """Recognize the generated DeepPoly chord u/(u-l) and its sign guards."""
    def zero(node):
        if isinstance(node, ast.Constant):
            return node.value == 0
        return (isinstance(node, ast.Call) and ast.unparse(node.func) == 'operator.mul'
                and any(isinstance(a, ast.Constant) and a.value == 0 for a in node.args))
    positive, negative = set(), set()
    for stmt in stmts:
        for node in ast.walk(stmt):
            if not isinstance(node, ast.Call) or len(node.args) != 2:
                continue
            f, (a, b) = ast.unparse(node.func), node.args
            if isinstance(a, ast.Name) and zero(b):
                if f in ('operator.gt', 'operator.ge'):
                    positive.add(a.id)
                if f in ('operator.lt', 'operator.le'):
                    negative.add(a.id)
    pairs = []
    for stmt in stmts:
        if not isinstance(stmt, ast.Assign):
            continue
        node = stmt.value
        if isinstance(node, ast.Call) and ast.unparse(node.func) == 'operator.sub' and len(node.args) == 2:
            u, l = node.args
            if isinstance(u, ast.Name) and isinstance(l, ast.Name) and l.id in positive and u.id in negative:
                if (l.id, u.id) not in pairs:
                    pairs.append((l.id, u.id))
    return pairs


class _SelectedRows(_ChunkBody):
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == 'patches_to_dense' and self._is_row(node):
            node.args[0] = self.visit(node.args[0])
            node.func.id = 'selected_patches_to_dense'
            node.args.insert(1, ast.Name('_idx', ast.Load()))
            return node
        return super().visit_Call(node)

    def _row_operand(self, node, result_ndim):
        # Generated convolution weights repeat each channel over spatial positions.
        # Gather channels before the repeat so stable rows never get materialized.
        shape = self.probe.shape(node)
        if shape is not None and len(shape) >= 2 and shape[:2] == (1, self.n_rows):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'reshape':
                perm = node.func.value
                if isinstance(perm, ast.Call) and isinstance(perm.func, ast.Attribute) and perm.func.attr == 'permute' and [ast.literal_eval(a) for a in perm.args] == [0, 1, 3, 2]:
                    rep = perm.func.value
                    if isinstance(rep, ast.Call) and isinstance(rep.func, ast.Attribute) and rep.func.attr == 'repeat':
                        reps = ast.literal_eval(rep.args[0])
                        unsq = rep.func.value
                        if len(reps) == 4 and reps[:3] == [1, 1, 1] and isinstance(unsq, ast.Call) and isinstance(unsq.func, ast.Attribute) and unsq.func.attr == 'unsqueeze':
                            value = ast.unparse(unsq.func.value)
                            return ast.parse(f'({value}).index_select(1, _idx // {reps[3]})', mode='eval').body
        if shape is not None and len(shape) >= 2 and shape[:2] == (1, self.n_rows):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'patches_to_dense':
                node.args[0] = self._row_operand(node.args[0], 3)
                node.func.id = 'selected_patches_to_dense'
                node.args.insert(1, ast.Name('_idx', ast.Load()))
                return node
        return super()._row_operand(node, result_ndim)


class _UseIndices(ast.NodeTransformer):
    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            if node.func.id == '_cf_rows':
                return ast.Call(ast.Attribute(node.args[0], 'index_select', ast.Load()), [ast.Constant(1), ast.Name('_idx', ast.Load())], [])
            if node.func.id == '_cf_tiled_rows':
                src = node.args[0]
                return ast.parse(f'({ast.unparse(src)}).index_select(1, _idx % ({ast.unparse(src)}).shape[1])', mode='eval').body
        return node


def _function(name, body, outputs, globals_, extra=()):
    assigned = {s.targets[0].id for s in body if isinstance(s, ast.Assign) and isinstance(s.targets[0], ast.Name)}
    read = set().union(*(_names(s.value) for s in body if isinstance(s, ast.Assign))) if body else set()
    read |= set(outputs)
    params = list(extra) + sorted(read - assigned - globals_.keys() - set(extra))
    src = f'def {name}({", ".join(params)}):\n'
    src += ''.join('    ' + ast.unparse(ast.fix_missing_locations(s)) + '\n' for s in body)
    src += '    return (' + ', '.join(outputs) + ',)\n'
    fn = ast.parse(src).body[0]
    fn.body = _add_dels(fn.body)
    return ast.unparse(ast.fix_missing_locations(fn)) + '\n', params


def _interval_source(network, target, known):
    """IBP from previously refined preactivations, including residual branches."""
    lines, cache = [], dict(known)
    def get(i):
        if i in cache:
            return cache[i]
        layer = network[i]
        if layer.type == LayerType.Input:
            return ('in_L_const_0', 'in_U_const_0')
        parents = [get(p) for p in layer.parents]
        l, u = parents[0]
        lo, hi = f'_sr_l_{i}', f'_sr_u_{i}'
        if layer.type == LayerType.ReLU:
            exprs = [f'torch.clamp({v}, min=0)' for v in (l, u)]
        elif layer.type == LayerType.Add:
            exprs = [f'{a} + {b}' for a, b in zip(parents[0], parents[1])]
        elif layer.type in (LayerType.Conv2D, LayerType.Linear):
            w, b = f'net_{i}_weight', f'net_{i}_bias'
            wp, wn = f'torch.clamp({w}, min=0)', f'torch.clamp({w}, max=0)'
            if layer.type == LayerType.Conv2D:
                shape = tuple(network[layer.parents[0]].shape)
                def op(x, weight):
                    return f'F.conv2d({x}.reshape({shape!r}), {weight}, stride={tuple(layer.stride)!r}, padding={tuple(layer.padding)!r}).flatten(1)'
            else:
                def op(x, weight):
                    return f'torch.matmul({x}, {weight}.transpose(-1, -2))'
            exprs = [f'{op(l, wp)} + {op(u, wn)} + {b}', f'{op(u, wp)} + {op(l, wn)} + {b}']
        else:
            raise ValueError(f'unsupported interval layer {i}: {layer.type}')
        lines.extend([f'{lo} = {exprs[0]}', f'{hi} = {exprs[1]}'])
        cache[i] = (lo, hi)
        return lo, hi
    result = get(target)
    return ast.parse('\n'.join(lines)).body, result


def rewrite(source, module_globals, inputs, network, chunk_size=128, compile_kernels=True, report=print):
    """Return selective source; reject layouts whose row independence is unproven.

    Supports the emitted standard DeepPoly ReLU chord and batch-one tensor flows.
    Final specification bounds are always computed densely.
    """
    tree, fn = flatten_flow(source)
    params = [a.arg for a in fn.args.args]
    if not inputs or tuple(inputs[0].shape[:1]) != (1,):
        raise ValueError('stable ReLU currently requires batch size one')
    if chunk_size < 1:
        raise ValueError('chunk size must be positive')
    stmts = fn.body
    if not all(isinstance(s, ast.Assign) for s in stmts[:-1]) or not isinstance(stmts[-1], ast.Return):
        raise ValueError('stable ReLU requires a straight-line functional tensor flow')
    pairs = relu_pairs(stmts)
    relus = [l for l in network if l.type == LayerType.ReLU]
    if len(pairs) != len(relus):
        raise ValueError(f'found {len(pairs)} bound pairs for {len(relus)} ReLUs; unsupported program')
    defs = {s.targets[0].id: i for i, s in enumerate(stmts) if isinstance(s, ast.Assign) and isinstance(s.targets[0], ast.Name)}
    if len(defs) != sum(isinstance(s, ast.Assign) for s in stmts):
        raise ValueError('stable ReLU requires scalarized SSA assignments')
    probe = _Probe(module_globals, params, inputs)
    probe.run(stmts)
    uses = {n: _names(stmts[i].value) & defs.keys() for n, i in defs.items()}
    # Shape alone cannot associate equal-width layers with ReLUs. Check that
    # each bound pair depends on exactly that preactivation's affine ancestors.
    provenance = {}
    for stmt in stmts:
        if isinstance(stmt, ast.Assign):
            name = stmt.targets[0].id
            inputs_used = _names(stmt.value)
            provenance[name] = set().union(*(provenance.get(v, {v} if v in params else set()) for v in inputs_used))
    def ancestor_parameters(i):
        layer = network[i]
        result = set().union(*(ancestor_parameters(p) for p in layer.parents))
        if layer.type in (LayerType.Linear, LayerType.Conv2D):
            result |= {f'net_{i}_weight', f'net_{i}_bias'} & set(params)
        return result
    for (lower, upper), relu in zip(pairs, relus):
        actual = {v for v in provenance[lower] | provenance[upper] if v.startswith('net_')}
        if actual != ancestor_parameters(relu.parents[0]):
            raise ValueError(f'bound provenance does not match ReLU {relu.identifier}')
    emitted, driver, sources, known = set(), [], [], {0: ('in_L_const_0', 'in_U_const_0')}
    globals_ = dict(module_globals)
    globals_['_idx'] = None
    globals_['selected_patches_to_dense'] = selected_patches_to_dense
    def closure(names):
        needed, todo = set(), list(names)
        while todo:
            n = todo.pop()
            if n in emitted or n in needed or n not in defs:
                continue
            needed.add(n)
            todo.extend(uses[n])
        return needed
    def emit_region(body, outputs, name, extra=()):
        text, args = _function(name, body, outputs, globals_, extra)
        sources.append(text)
        if compile_kernels:
            sources.append(f"{name} = torch.compile({name}, fullgraph=True, dynamic=False, options={{'layout_optimization': False}})\n")
        return args
    for k, ((lower, upper), relu) in enumerate(zip(pairs, relus)):
        target = relu.parents[0]
        n = network[target].size
        if probe.shapes[lower] != (1, n) or probe.shapes[upper] != (1, n):
            raise ValueError(f'bound shape does not match ReLU {relu.identifier}')
        needed = closure([lower, upper])
        phase = [stmts[defs[v]] for v in sorted(needed, key=defs.get)]
        # Coefficient tensors carry target neurons on axis 1. Scalar ancestors
        # (previous ReLU relaxations, convolutions of intercepts) stay outside.
        rows = set()
        for s in phase:
            name = s.targets[0].id
            shape = probe.shapes[name]
            if shape and len(shape) >= 3 and shape[:2] == (1, n) and shape[2] > 1:
                rows.add(name)
            elif shape and len(shape) >= 2 and shape[:2] == (1, n) and _names(s.value) & rows:
                rows.add(name)
        future = set().union(*(v for name, v in uses.items() if name not in needed and name not in emitted))
        # Non-bound coefficient escape would require dense values downstream.
        if (rows & future) - {lower, upper}:
            raise ValueError(f'coefficient tensors escape ReLU {k}: {(rows & future) - {lower, upper}}')
        can_select = n > 1 and lower in rows and upper in rows
        pre = [s for s in phase if not can_select or s.targets[0].id not in rows]
        if can_select and any(_names(s.value) & rows for s in pre):
            raise ValueError(f'unsupported coefficient layout at ReLU {k}')
        exported = sorted((needed & future) - rows if can_select else needed & future)
        exported = sorted(set(exported) | ({lower, upper} if not can_select else set()))
        if can_select:
            interval, coarse = _interval_source(network, target, known)
            pre += interval
            exported += list(coarse)
        else:
            coarse = (lower, upper)
        # Include scalar ancestors needed by the selected kernel in pre outputs.
        selected, hoisted = [], []
        if can_select:
            for s in phase:
                if s.targets[0].id not in rows:
                    continue
                t = _SelectedRows(probe, n, rows, set())
                s = copy.deepcopy(s)
                s.value = t.visit(s.value) if t._is_row(s.value) else t._row_operand(s.value, len(probe.shapes[s.targets[0].id]))
                if t.failed:
                    raise ValueError(t.failed)
                for old, expr in t.hoisted:
                    new = f'_sr_hoist_{k}_{len(hoisted)}'
                    for node in ast.walk(s):
                        if isinstance(node, ast.Name) and node.id == old:
                            node.id = new
                    hoisted.append(ast.Assign([ast.Name(new, ast.Store())], expr))
                selected.append(_UseIndices().visit(s))
            pre += hoisted
            selected = ast.parse('_n = _idx.numel()').body + selected
            args = emit_region(selected, [lower, upper], f'_sr_rows_{k}', ('_idx',))
            pre_names = {s.targets[0].id for s in pre}
            exported = sorted(set(exported) | (set(args) & pre_names))
        pre_args = emit_region(pre, exported, f'_sr_pre_{k}')
        driver.append(f'{", ".join(exported)}, = _sr_pre_{k}({", ".join(pre_args)})')
        if can_select:
            driver.append(f'{lower}, {upper} = refine_stable({coarse[0]}, {coarse[1]}, _sr_rows_{k}, ({", ".join(args[1:])},), {chunk_size})')
        emitted |= needed
        known[target] = (lower, upper)
        report(f'[stable-relu] ReLU {k}: {n} neurons, {len(selected)} selected statements')
    # Remaining output computation is unchanged; segment at original source
    # boundaries later if its memory footprint requires it.
    rest = [s for s in stmts if isinstance(s, ast.Assign) and s.targets[0].id not in emitted]
    ret = stmts[-1]
    if not isinstance(ret, ast.Return) or not isinstance(ret.value, (ast.Tuple, ast.List)):
        raise ValueError('expected a pair of final bound expressions')
    out_names = []
    for i, expr in enumerate(ret.value.elts):
        name = f'_sr_result_{i}'
        rest.append(ast.Assign([ast.Name(name, ast.Store())], expr))
        out_names.append(name)
    final_args = emit_region(rest, out_names, '_sr_final')
    driver.append(f'return _sr_final({", ".join(final_args)})')
    fn.body = ast.parse('\n'.join(driver)).body
    tree.body = [s for s in tree.body if s is not fn]
    tree.body += ast.parse('from constraintflow.compiler.optimizations.stable_relu import refine_stable, selected_patches_to_dense\n' + '\n'.join(sources)).body + [fn]
    return ast.unparse(ast.fix_missing_locations(tree))


_warming_stable_relu = ContextVar('warming_stable_relu', default=False)


def warmup_stable_relu(call, *args, **kwargs):
    """Run a warmup that exercises even currently unused refinement kernels.

    Dummy results are discarded, so warmup returns the same bounds as a normal
    call. Use the same grad mode, dtype and device as subsequent measurements.
    No warmup state persists after the call, including if compilation raises.
    """
    token = _warming_stable_relu.set(True)
    try:
        return call(*args, **kwargs)
    finally:
        _warming_stable_relu.reset(token)


def refine_stable(lower, upper, kernel, args, chunk_size):
    """Refine uncertain rows only; pad the last chunk to avoid recompilation.

    NaN/invalid intervals are not a proof of stability. Zero-width zero intervals
    are stable. Use functional index_copy so inputs and compiled outputs aren't
    mutated. The dense final specification must never use this shortcut.
    """
    stable = ((lower >= 0) | (upper <= 0)) & (lower <= upper) & torch.isfinite(lower) & torch.isfinite(upper)
    indices = torch.nonzero(~stable.all(dim=0), as_tuple=False).flatten()
    if indices.numel() == 0 and _warming_stable_relu.get() and lower.shape[1] > 0:
        # Match the real call's int64 index shape, strides and device, including
        # layers with fewer neurons than a chunk. Never scatter these results.
        dummy = torch.arange(chunk_size, device=lower.device) % lower.shape[1]
        kernel(dummy, *args)
    out_l, out_u = lower, upper
    for offset in range(0, indices.numel(), chunk_size):
        idx = indices[offset:offset + chunk_size]
        count = idx.numel()
        if count < chunk_size:
            idx = torch.cat((idx, idx[-1:].expand(chunk_size - count)))
        lo, hi = kernel(idx, *args)
        out_l = out_l.index_copy(1, idx[:count], lo[:, :count])
        out_u = out_u.index_copy(1, idx[:count], hi[:, :count])
    return out_l, out_u


def selected_patches_to_dense(pieces, indices, batch, oc, ox, oy, ic, kx, ky, ix, iy, px, py, sx, sy):
    """Patch conversion retaining each selected neuron's original spatial origin."""
    pos = indices % (ox * oy)
    x = torch.arange(ix, device=pieces.device)[None, :] + px - (pos // oy)[:, None] * sx
    y = torch.arange(iy, device=pieces.device)[None, :] + py - (pos % oy)[:, None] * sy
    valid = ((x >= 0) & (x < kx))[:, :, None] & ((y >= 0) & (y < ky))[:, None, :]
    index = (x.clamp(0, kx - 1)[:, :, None] * ky + y.clamp(0, ky - 1)[:, None, :])
    count = indices.numel()
    source = pieces.reshape(batch, count, ic, kx * ky)
    index = index.reshape(1, count, 1, ix * iy).expand(batch, count, ic, ix * iy)
    values = source.gather(-1, index)
    return torch.where(valid.reshape(1, count, 1, ix * iy), values, 0).reshape(batch, count, ic * ix * iy)


def optimize_program(folder, network_file, x, y, dataset, eps=0.0, *, chunk_size=128, compile_kernels=True):
    """Apply the pass to a freshly emitted standalone (non-training) program.

    Load sibling modules in an isolated temporary namespace and restore the
    compiler's capture modules afterwards. Only replace source after validation.
    """
    import importlib.util
    from pathlib import Path
    import sys
    root = Path(folder).resolve()
    previous = {name: sys.modules.get(name) for name in ('jit_constants', 'transformers', 'main')}
    try:
        modules = {}
        for name in previous:
            spec = importlib.util.spec_from_file_location(name, root / f'{name}.py')
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            modules[name] = module
        main = modules['main']
        network, abs_elem = main.build(network_file, 1, eps, x, y, dataset, False, False, False)
        inputs = main.explode_inputs(abs_elem, 1)
        path = root / 'transformers.py'
        source = path.read_text()
        transformed = rewrite(source, vars(modules['transformers']), inputs, network,
                              chunk_size=chunk_size, compile_kernels=compile_kernels)
        (root / 'transformers_dense.py').write_text(source)
        path.write_text(transformed)
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
