#FEAT: row-chunked back-substitution for networks whose traversals do not fit in GPU memory (e.g. VGG16).
"""Run each over-budget back-substitution of an eager (no --inductor) flow() as a loop over row chunks.

A traversal bounds the N neurons of one layer. Its coefficient tensors carry those neurons on axis 1
([batch, N, columns]; [batch*N, C, P, P] inside conv_transpose2d), and every neuron's row is computed
independently and ends in a per-neuron sum. So the same statements run on rows [r0, r1) give rows
[r0, r1) of every result: the bounds are unchanged, only the peak memory drops.

The pass works on the emitted source (batch 1):
  1. fake-executes flow() for every variable's shape and every statement's allocation;
  2. from each statement over the budget, collects its traversal: statements whose 3-D+ results
     carry the same N rows on axis 1, connected through each other, plus the row-indexed statements
     that consume them (the per-neuron sums);
  3. cheap traversal tensors that use no other traversal tensor stay outside the loop and are sliced
     per chunk; the rest run in the loop with the row count in their reshapes replaced by the chunk
     size and row-aligned operands built from other layers (per-position patches tiled with .repeat,
     weights expanded over positions) taken row by row;
  4. row-indexed results used after the loop are concatenated; statements are re-ordered so each
     loop follows its inputs and precedes its consumers; `del` frees every variable after its last use.

    python -m constraintflow.compiler.optimizations.row_chunks <program dir> --network net.onnx \
        --dataset box-3x224x224-1000-1 --budget-gb 20
"""
import argparse
import ast
import copy
import heapq
import math
import os
import sys

import torch

ROWS_HELPER = '''
import os as _cf_os
import time as _cf_time
_CF_PROGRESS = _cf_os.environ.get('CF_ROW_CHUNKS_PROGRESS', '0') not in ('', '0')
_cf_last = [None]


def _cf_progress(k, n_rows):
    """CF_ROW_CHUNKS_PROGRESS=1: time since the previous chunked traversal finished."""
    if _CF_PROGRESS:
        torch.cuda.synchronize()
        now = _cf_time.perf_counter()
        print(f'[row-chunks] loop {k} ({n_rows} rows) done, +{now - (_cf_last[0] or now):.1f} s', flush=True)
        _cf_last[0] = now


def _cf_rows(x, r0, r1):
    """Rows [r0, r1) of axis 1 of a row-aligned tensor whose rows repeat with period x.shape[1]."""
    return x[:, r0:r1]


def _cf_tiled_rows(x, times, r0, r1):
    """Rows [r0, r1) of x.repeat([1, times, 1, ...]) without materializing the repeat."""
    period = x.shape[1]
    if r0 % period == 0 and (r1 - r0) % period == 0:
        return x.repeat([1, (r1 - r0) // period] + [1] * (x.dim() - 2))
    idx = torch.arange(r0, r1, device=x.device) % period
    return x.index_select(1, idx)
'''


class _Names(ast.NodeVisitor):
    def __init__(self):
        self.names = set()

    def visit_Name(self, node):
        self.names.add(node.id)


def _names(node):
    v = _Names()
    v.visit(node)
    return v.names


def _shape_of(value):
    return tuple(value.shape) if isinstance(value, torch.Tensor) else None


class _Probe:
    """Fake execution of flow(): shapes of variables and sub-expressions, bytes per statement."""

    def __init__(self, module_globals, params, inputs):
        from torch._subclasses.fake_tensor import FakeTensorMode
        from constraintflow.compiler.optimizations.flow_shapes import _Recorder
        self.mode = FakeTensorMode(allow_non_fake_inputs=True)
        self.recorder_cls = _Recorder
        self.env = dict(module_globals)
        with self.mode:
            for name, value in zip(params, inputs):
                self.env[name] = (torch.empty(value.shape, dtype=value.dtype, device=value.device)
                                  if isinstance(value, torch.Tensor) else value)
        self.shapes = {n: _shape_of(v) for n, v in self.env.items() if n in params}
        self.stmt_bytes = {}

    def run(self, stmts):
        for i, stmt in enumerate(stmts):
            if not isinstance(stmt, ast.Assign):
                continue
            rec = self.recorder_cls()
            rec.known_roots = set()
            with self.mode, rec:
                exec(compile(ast.Module([stmt], []), '<flow>', 'exec'), self.env)
            name = stmt.targets[0].id
            self.shapes[name] = _shape_of(self.env[name])
            self.stmt_bytes[i] = max(sum(rec.bytes), rec.temporary_peak)

    def shape(self, node):
        with self.mode:
            return _shape_of(eval(compile(ast.Expression(node), '<expr>', 'eval'), self.env))


def _per_row(shape, n_rows):
    """[1, N] or [1, N, 1, ...]: one value per neuron."""
    return shape is not None and len(shape) >= 2 and shape[:2] == (1, n_rows) and math.prod(shape[2:]) == 1


def _row_axis_size(shape, ndim, n_rows):
    """Does an operand of `shape`, broadcast against a [1, N, ...] result of rank ndim, carry the rows?"""
    if shape is None:
        return False
    offset = ndim - len(shape)
    return 1 - offset >= 0 and shape[1 - offset] == n_rows


def _rewrite_row_literals(node, n_rows):
    """In reshape/view/expand shape lists, the leading row entry (prefix product 1) N -> _n."""
    lst = node.args[0] if len(node.args) == 1 and isinstance(node.args[0], (ast.List, ast.Tuple)) else None
    elts = lst.elts if lst is not None else node.args
    prefix = 1
    for k, e in enumerate(elts):
        if not isinstance(e, ast.Constant) or not isinstance(e.value, int):
            return False
        if prefix == 1 and e.value == n_rows:
            elts[k] = ast.Name('_n', ast.Load())
            return True
        prefix *= max(e.value, 1)
        if prefix > 1:
            return False
    return False


ELEMENTWISE = {('operator', f) for f in ('add', 'sub', 'mul', 'truediv', 'neg', 'gt', 'lt', 'ge', 'le', 'eq', 'ne')} \
    | {('torch', f) for f in ('where', 'clamp', 'maximum', 'minimum', 'abs')} | {(None, 'safe_div')}


def _elementwise(node):
    if not isinstance(node, ast.Call):
        return False
    f = node.func
    if isinstance(f, ast.Name):
        return (None, f.id) in ELEMENTWISE
    return isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and (f.value.id, f.attr) in ELEMENTWISE


class _ChunkBody(ast.NodeTransformer):
    """Rewrite one loop statement so it computes rows [_r0, _r1)."""

    def __init__(self, probe, n_rows, row_vars, sliced_vars):
        self.probe, self.n_rows = probe, n_rows
        self.row_vars, self.sliced_vars = row_vars, sliced_vars
        self.hoisted = []          # (name, expression) computed once before the loop
        self.periods = []          # rows of each tiled source (positions per channel)
        self.failed = None

    def _is_row(self, node):
        return bool(_names(node) & self.row_vars)

    def _row_operand(self, node, result_ndim):
        """Replace a column-side operand that is nevertheless row-aligned by its chunk rows."""
        shape = self.probe.shape(node)
        if not _row_axis_size(shape, result_ndim, self.n_rows):
            return node
        if len(shape) < 2 or shape[0] != 1 or shape[1] != self.n_rows:
            self.failed = f'row-aligned operand of shape {shape} not in [1, N, ...] form'
            return node
        if _elementwise(node):
            # Broadcasting elementwise op: slice its row-aligned operands, keep the rest.
            node.args = [a if isinstance(a, (ast.Constant, ast.List, ast.Tuple)) else self._row_operand(a, result_ndim)
                         for a in node.args]
            return node
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'repeat'):
            reps = node.args[0].elts if len(node.args) == 1 and isinstance(node.args[0], ast.List) else node.args
            if all(isinstance(r, ast.Constant) for r in reps) and [r.value for r in reps][0] == 1 \
                    and all(r.value == 1 for r in reps[2:]):
                self.periods.append(self.probe.shape(node.func.value)[1])
                return ast.Call(ast.Name('_cf_tiled_rows', ast.Load()),
                                [node.func.value, reps[1], ast.Name('_r0', ast.Load()), ast.Name('_r1', ast.Load())], [])
        name = f'_cf_hoist_{len(self.hoisted)}_{id(node)}'
        self.hoisted.append((name, node))
        return ast.Call(ast.Name('_cf_rows', ast.Load()),
                        [ast.Name(name, ast.Load()), ast.Name('_r0', ast.Load()), ast.Name('_r1', ast.Load())], [])

    def visit_Name(self, node):
        if node.id in self.sliced_vars:
            return ast.Call(ast.Name('_cf_rows', ast.Load()),
                            [ast.Name(node.id, ast.Load()), ast.Name('_r0', ast.Load()), ast.Name('_r1', ast.Load())], [])
        return node

    def generic_visit_row(self, node):
        """A node that touches row variables: fix its column-side operands, then recurse."""
        #FIX FOR: a matmul's column-side operand of size N (fc 4096 -> 4096) was sliced like a row operand.
        # Only a broadcasting elementwise op aligns a column-side operand with the rows.
        elementwise = isinstance(node, ast.BinOp) or _elementwise(node)
        try:
            ndim = len(self.probe.shape(node)) if elementwise else None
        except Exception:
            ndim = None
        for field, value in ast.iter_fields(node):
            items = value if isinstance(value, list) else [value]
            new = []
            for item in items:
                if isinstance(item, ast.expr) and not isinstance(item, (ast.Constant, ast.List, ast.Tuple)) \
                        and not self._is_row(item) and ndim is not None and not isinstance(item, ast.Attribute):
                    item = self._row_operand(item, ndim)
                elif isinstance(item, ast.AST):
                    item = self.visit(item)
                new.append(item)
            setattr(node, field, new if isinstance(value, list) else new[0])
        return node

    def visit_Call(self, node):
        if not self._is_row(node):
            return node
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in ('reshape', 'view', 'expand') and self._is_row(func.value):
            _rewrite_row_literals(node, self.n_rows)
            node.func.value = self.visit(func.value)
            return node
        if isinstance(func, ast.Attribute) and self._is_row(func.value):
            node.func.value = self.visit(func.value)
            node.args = [self.visit(a) for a in node.args]
            return node
        return self.generic_visit_row(node)

    def visit_BinOp(self, node):
        return self.generic_visit_row(node) if self._is_row(node) else node

    def visit_Subscript(self, node):
        if not self._is_row(node.value):
            return node
        node.value = self.visit(node.value)
        dims = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        row = dims[1] if len(dims) > 1 else None
        if (isinstance(row, ast.Slice) and row.step is None and (row.lower is None or
                (isinstance(row.lower, ast.Constant) and row.lower.value == 0))
                and isinstance(row.upper, ast.Constant) and row.upper.value == self.n_rows):
            row.upper = ast.Name('_n', ast.Load())
        else:
            self.failed = 'subscript of a row tensor that does not span all rows'
        return node


def _components(stmts, probe, budget):
    """Traversals to chunk: (N, statement indices in the loop, cheap outside row statements)."""
    defs = {s.targets[0].id: i for i, s in enumerate(stmts) if isinstance(s, ast.Assign)}
    uses = {i: _names(s.value if isinstance(s, ast.Assign) else s) & defs.keys() for i, s in enumerate(stmts)}
    users = {i: [] for i in range(len(stmts))}
    for i, names in uses.items():
        for n in names:
            users[defs[n]].append(i)

    def rows(i):
        name = stmts[i].targets[0].id if isinstance(stmts[i], ast.Assign) else None
        shape = probe.shapes.get(name) if name else None
        return shape[1] if shape is not None and len(shape) >= 3 and shape[0] == 1 else None

    done, comps = set(), []
    for seed in sorted(probe.stmt_bytes, key=probe.stmt_bytes.get, reverse=True):
        if probe.stmt_bytes[seed] <= budget or seed in done or rows(seed) is None:
            continue
        n_rows = rows(seed)
        comp, todo = set(), [seed]
        while todo:
            i = todo.pop()
            if i in comp:
                continue
            comp.add(i)
            for j in [defs[n] for n in uses[i]] + users[i]:
                if j not in comp and rows(j) == n_rows:
                    todo.append(j)
        done |= comp
        # Statements of the traversal that use no other traversal tensor and are cheap: compute once, slice.
        names = {stmts[i].targets[0].id for i in comp}
        outside = {i for i in comp if not (uses[i] & names) and probe.stmt_bytes[i] <= budget / 8}
        loop = comp - outside
        # Row-indexed consumers of the coefficient tensors (the per-neuron sums) join the loop.
        for i in sorted(loop):
            if _per_row(probe.shapes[stmts[i].targets[0].id], n_rows):
                continue
            for j in users[i]:
                if j in loop or not isinstance(stmts[j], ast.Assign):
                    continue
                if _per_row(probe.shapes.get(stmts[j].targets[0].id), n_rows):
                    loop.add(j)
        comps.append((n_rows, loop, outside))
    return comps, defs, uses, users


def _chunk_body(stmts, probe, n_rows, loop, outside, users):
    """Loop statements rewritten for rows [_r0, _r1), hoisted expressions, exported results, source periods."""
    row_vars = {stmts[i].targets[0].id for i in loop}
    sliced = {stmts[i].targets[0].id for i in outside}
    body, hoisted, periods = [], [], []
    for i in sorted(loop):
        stmt = copy.deepcopy(stmts[i])
        t = _ChunkBody(probe, n_rows, row_vars | sliced, sliced)
        if t._is_row(stmt.value):
            stmt.value = t.visit(stmt.value)
        else:   # built only from other layers and weights, but row-aligned: take its rows
            stmt.value = t._row_operand(stmt.value, len(probe.shapes[stmts[i].targets[0].id]))
        if t.failed:
            raise ValueError(f'statement {i} ({stmts[i].targets[0].id}): {t.failed}')
        hoisted += t.hoisted
        periods += t.periods
        body.append(stmt)
    exported = sorted({stmts[i].targets[0].id for i in loop
                       if any(j not in loop for j in users[i]) or isinstance(stmts[-1], ast.Return)
                       and stmts[i].targets[0].id in _names(stmts[-1])})
    for name in exported:
        shape = probe.shapes[name]
        if not _per_row(shape, n_rows):
            raise ValueError(f'{name} {shape} is used after the loop but is not a per-neuron result')
    return [ast.Assign([ast.Name(n, ast.Store())], e) for n, e in hoisted], body, exported, periods


def _emit_loop(n_rows, chunk, pre, body, exported, k=0):
    """Eager ast.For over row chunks plus the statements before and after it."""
    pre = list(pre)
    pre += [ast.parse(f'_cf_out_{n} = []').body[0] for n in exported]
    header = ast.parse(f'for _r0 in range(0, {n_rows}, {chunk}):\n'
                       f'    _r1 = min(_r0 + {chunk}, {n_rows})\n    _n = _r1 - _r0\n').body[0]
    header.body += body + [ast.parse(f'_cf_out_{n}.append({n})').body[0] for n in exported]
    post = [ast.parse(f'{n} = torch.cat(_cf_out_{n}, dim=1)').body[0] for n in exported]
    post.append(ast.parse(f'_cf_progress({k}, {n_rows})').body[0])
    return pre, header, post


def _largest_divisor(n, at_most):
    for d in range(min(n, max(1, at_most)), 0, -1):
        if n % d == 0:
            return d
    return 1


def _fixed_chunk(n_rows, target, periods):
    """A chunk size dividing n_rows (constant shapes, one compile) and the mode of its tiled sources."""
    periods = set(periods)
    if not periods:
        return _largest_divisor(n_rows, target), 'plain'
    if len(periods) == 1 and target >= next(iter(periods)):
        period = next(iter(periods))
        return period * _largest_divisor(n_rows // period, target // period), 'whole'
    return _largest_divisor(math.gcd(*periods), target), 'part'


class _BodyArgs(ast.NodeTransformer):
    """Turn per-chunk slicing calls into parameters of a compiled body; collect what the loop passes."""

    def __init__(self, probe, chunk, mode, k):
        self.probe, self.chunk, self.mode, self.k = probe, chunk, mode, k
        self.args = []      # (parameter, expression evaluated in the loop)
        self.positions = {} # period -> parameter holding the chunk's positions (sub-channel chunks)

    def _param(self, expr):
        name = f'_cf_a{len(self.args)}'
        self.args.append((name, expr))
        return ast.Name(name, ast.Load())

    def visit_Call(self, node):
        self.generic_visit(node)
        f = node.func
        if isinstance(f, ast.Name) and f.id == '_cf_rows':
            return self._param(node)
        if isinstance(f, ast.Name) and f.id == '_cf_tiled_rows':
            # The per-position source stays inside the body (Inductor fuses it into its consumer);
            # nothing of it is kept across chunks.
            src = node.args[0]
            shape = self.probe.shape(src)
            if self.mode == 'whole':
                return _repeat(src, self.chunk // shape[1], len(shape))
            period = shape[1]
            if period not in self.positions:
                self.positions[period] = self._param(ast.parse(
                    f'torch.arange(_r0 % {period}, _r0 % {period} + {self.chunk}, device=device_mode.get_device())',
                    mode='eval').body)
            return ast.Call(ast.Attribute(src, 'index_select', ast.Load()),
                            [ast.Constant(1), ast.Name(self.positions[period].id, ast.Load())], [])
        return node

    def visit_Name(self, node):
        return ast.Constant(self.chunk) if node.id == '_n' else node


def _repeat(param, times, ndim):
    return ast.Call(ast.Attribute(param, 'repeat', ast.Load()),
                    [ast.List([ast.Constant(1), ast.Constant(times)] + [ast.Constant(1)] * (ndim - 2), ast.Load())], [])


INDUCTOR = ("dict(fullgraph=True, dynamic=False, options={'layout_optimization': False, "
            "'triton.autotune_pointwise': False, 'pre_grad_custom_pass': PRE_GRAD_CSE, "
            "'post_grad_custom_post_pass': SharedLhsMM()})")


def _emit_compiled(k, n_rows, chunk, mode, probe, pre, body, exported, module_globals):
    """A torch.compile'd body over one chunk, the eager loop calling it, and the statements around it."""
    t = _BodyArgs(probe, chunk, mode, k)
    body = [t.visit(copy.deepcopy(s)) for s in body]
    defined = {tgt.id for s in body for tgt in s.targets}
    loads = set().union(*(_names(s.value) for s in body)) - defined - {a for a, _ in t.args}
    free = sorted(n for n in loads if n not in module_globals)
    params = [a for a, _ in t.args] + free
    fn = ast.FunctionDef(f'_cf_body_{k}', ast.arguments([], [ast.arg(p) for p in params], None, [], [], None, []),
                         body + [ast.Return(ast.Tuple([ast.Name(n, ast.Load()) for n in exported], ast.Load()))],
                         [], None, None, [])
    compiled = ast.parse(f'_cf_body_{k} = torch.compile(_cf_body_{k}, **{INDUCTOR})').body
    pre = list(pre)
    pre += [ast.parse(f'_cf_out_{n} = []').body[0] for n in exported]
    call = ast.Call(ast.Name(f'_cf_body_{k}', ast.Load()),
                    [e for _, e in t.args] + [ast.Name(n, ast.Load()) for n in free], [])
    loop = ast.parse(f'for _r0 in range(0, {n_rows}, {chunk}):\n    _cf_res = None\n').body[0]
    loop.body = [ast.parse(f'_r1 = _r0 + {chunk}').body[0], ast.Assign([ast.Name('_cf_res', ast.Store())], call)]
    loop.body += [ast.parse(f'_cf_out_{n}.append(_cf_res[{j}])').body[0] for j, n in enumerate(exported)]
    post = [ast.parse(f'{n} = torch.cat(_cf_out_{n}, dim=1)').body[0] for n in exported]
    post.append(ast.parse(f'_cf_progress({k}, {n_rows})').body[0])
    return [fn] + compiled, pre, loop, post


def _add_dels(body, protected=()):
    """Insert `del` after each name's last use in a statement list (recursing into for-loops)."""
    # Only names assigned at this level; a loop body frees its own.
    defined_here = {t.id for s in body if isinstance(s, ast.Assign) for t in s.targets if isinstance(t, ast.Name)}
    last = {}
    for k, s in enumerate(body):
        for n in _names(s):
            last[n] = k
    out = []
    for k, s in enumerate(body):
        if isinstance(s, ast.For):
            s.body = _add_dels(s.body, protected=set(protected) | _loop_carried(s))
        out.append(s)
        dead = sorted(n for n, kk in last.items() if kk == k and n in defined_here and n not in protected
                      and not isinstance(s, ast.Return))
        if dead:
            out.append(ast.Delete([ast.Name(n, ast.Del()) for n in dead]))
    return out


def _loop_carried(loop):
    """Names a loop body must not delete: lists it appends to and the loop indices."""
    keep = {'_r0', '_r1', '_n'}
    for t in ast.walk(loop):
        if isinstance(t, ast.Attribute) and t.attr == 'append' and isinstance(t.value, ast.Name):
            keep.add(t.value.id)
    return keep


def rewrite(source, module_globals, inputs, budget, report=print, compile_bodies=False):
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'flow')
    fn.decorator_list = []
    params = [a.arg for a in fn.args.args]
    stmts = fn.body
    probe = _Probe(module_globals, params, inputs)
    probe.run(stmts)
    comps, defs, uses, users = _components(stmts, probe, budget)
    if not comps:
        report('[row-chunks] nothing over budget')
        return None
    # Each loop is one schedulable unit; everything else keeps its own statement.
    unit_of = {i: ('s', i) for i in range(len(stmts))}
    units, module_defs = {}, []
    for c, (n_rows, loop, outside) in enumerate(comps):
        per_row = max(probe.stmt_bytes.get(i, 0) for i in loop) / n_rows
        target = max(1, int(budget / (3 * per_row)))
        pre, body, exported, periods = _chunk_body(stmts, probe, n_rows, loop, outside, users)
        if compile_bodies:
            chunk, mode = _fixed_chunk(n_rows, target, periods)
            defs_c, pre, header, post = _emit_compiled(c, n_rows, chunk, mode, probe, pre, body, exported,
                                                       module_globals)
            module_defs += defs_c
        else:
            chunk = target
            pre, header, post = _emit_loop(n_rows, chunk, pre, body, exported, c)
        units[('l', c)] = pre + [header] + post
        for i in loop:
            unit_of[i] = ('l', c)
        report(f'[row-chunks] traversal of {n_rows} rows: {len(loop)} statements in the loop, '
               f'{len(outside)} sliced, {math.ceil(n_rows / chunk)} chunks of {chunk} rows, '
               f'largest statement {max(probe.stmt_bytes.get(i, 0) for i in loop) / 2**30:.1f} GiB -> '
               f'{chunk * per_row / 2**30:.1f} GiB, results {exported}')
    for i in range(len(stmts)):
        if unit_of[i][0] == 's':
            units[unit_of[i]] = [stmts[i]]
    # Stable topological order of units by first statement index.
    first = {}
    for i in range(len(stmts)):
        first.setdefault(unit_of[i], i)
    deps = {u: set() for u in units}
    for i in range(len(stmts)):
        for n in uses[i]:
            d = unit_of[defs[n]]
            if d != unit_of[i]:
                deps[unit_of[i]].add(d)
    # Hoisted expressions inside loops may name other statements' variables too.
    for u, block in units.items():
        for s in block:
            for n in _names(s) & defs.keys():
                d = unit_of[defs[n]]
                if d != u:
                    deps[u].add(d)
    indeg = {u: len(d) for u, d in deps.items()}
    rdeps = {u: [] for u in units}
    for u, d in deps.items():
        for v in d:
            rdeps[v].append(u)
    ready = [(first[u], u) for u in units if indeg[u] == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        _, u = heapq.heappop(ready)
        order.append(u)
        for v in rdeps[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(ready, (first[v], v))
    if len(order) != len(units):
        raise ValueError('row chunks: dependency cycle between loops')
    fn.body = _add_dels([s for u in order for s in units[u]], protected=set(params))
    helper = ast.parse(ROWS_HELPER).body + module_defs
    idx = tree.body.index(fn)
    tree.body[idx:idx] = helper
    return ast.unparse(ast.fix_missing_locations(tree))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('program')
    p.add_argument('--network', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--input-shape', required=True, help='e.g. 3x224x224')
    p.add_argument('--budget-gb', type=float, default=20)
    p.add_argument('--compile-bodies', action='store_true',
                   help='torch.compile (Inductor) each loop body; chunk sizes divide the rows so shapes stay fixed')
    args = p.parse_args()
    from constraintflow.lib.training import SpecBounds, load_program
    from constraintflow.lib.globals import device_mode
    import onnx
    device_mode.set_mode('gpu')
    src_path = os.path.join(args.program, 'transformers.py')
    orig_path = os.path.join(args.program, 'transformers_unchunked.py')
    if not os.path.exists(orig_path):
        os.replace(src_path, orig_path)
    with open(orig_path) as f:
        source = f.read()
    with open(src_path, 'w') as f:      # load the unchunked program to build the inputs
        f.write(source)
    main_mod = load_program(args.program)
    shape = tuple(map(int, args.input_shape.split('x')))
    inits = [i for i in onnx.load(args.network).graph.initializer if i.data_type == onnx.TensorProto.FLOAT]
    params = {i.name: torch.from_numpy(onnx.numpy_helper.to_array(i).copy()).cuda() for i in inits}
    x = torch.zeros(1, *shape, device='cuda')
    y = torch.zeros(1, dtype=torch.long, device='cuda')
    network, abs_elem = main_mod.build(args.network, 1, 0.0, x, y, args.dataset, False, False, False,
                                       initializers=params, input_box=(x, x))
    inputs = main_mod.explode_inputs(abs_elem, 1)
    transformers = sys.modules[main_mod.__name__.rsplit('.', 1)[0] + '.transformers']
    out = rewrite(source, vars(transformers), inputs, args.budget_gb * 2**30, compile_bodies=args.compile_bodies)
    with open(src_path, 'w') as f:
        f.write(out if out is not None else source)


if __name__ == '__main__':
    main()
