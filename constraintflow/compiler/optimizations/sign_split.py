from constraintflow.compiler.ir import (
    IrAccess, IrAddDimension, IrAssignment, IrBinaryOp, IrClamp,
    IrInnerProduct, IrMult, IrRemoveDimension, IrRepeat, IrSignSplit, IrVar,
)
from constraintflow.lib.jit_semantics import read_sources, traversal_sources


_WRAPPERS = (IrRemoveDimension, IrAddDimension, IrRepeat)


def _deref(expr):
    if isinstance(expr, IrVar) and expr.defs is not None:
        return expr.defs.children[1]
    return expr


def _unwrap(expr):
    expr = _deref(expr)
    if isinstance(expr, IrClamp):
        return expr, None
    if isinstance(expr, _WRAPPERS):
        inner = _deref(expr.children[0])
        if isinstance(inner, IrClamp):
            return inner, expr
    return None, None


def _unclamp(term):
    node = _deref(term)
    if not isinstance(node, (IrInnerProduct, IrMult)):
        return None
    for index, child in enumerate(node.children):
        clamp, wrapper = _unwrap(child)
        if clamp is None:
            continue
        value = clamp.children[0]
        if wrapper is not None:
            wrapper.update_parent_child([value] + list(wrapper.children[1:]))
        else:
            children = list(node.children)
            children[index] = value
            node.update_parent_child(children)
        return term
    return None


def _term(expr):
    node = _deref(expr)
    if not isinstance(node, (IrInnerProduct, IrMult)):
        return None
    for index, child in enumerate(node.children):
        clamp, wrapper = _unwrap(child)
        if clamp is not None:
            return node, clamp, index, wrapper, node.children[1 - index]
    return None


def _field(expr, seen=None):
    if seen is None:
        seen = set()
    expr = _deref(expr)
    if isinstance(expr, int) or id(expr) in seen:
        return None
    seen.add(id(expr))
    if isinstance(expr, IrAccess) and not expr.isMetadata:
        return expr.elem
    for child in expr.children:
        field = _field(child, seen)
        if field is not None:
            return field
    return None


def _semantic_split(expr):
    if isinstance(expr, IrSignSplit):
        return expr
    if type(expr) is not IrBinaryOp or expr.op != '+':
        return None
    lhs = _term(expr.children[0])
    rhs = _term(expr.children[1])
    if lhs is None or rhs is None:
        return None
    if lhs[1].min_true == rhs[1].min_true:
        return None
    if _deref(lhs[1].children[0]) != _deref(rhs[1].children[0]):
        return None
    fields = (_field(lhs[4]), _field(rhs[4]))
    if None in fields:
        return None
    split = IrSignSplit(
        expr.children[0], expr.children[1], lhs[1].children[0], fields)
    split.ttb_counter = expr.ttb_counter
    split.inside_while = expr.inside_while
    split.while_number = expr.while_number
    split.while_iteration = expr.while_iteration
    return split


def _site_sources(node, layer, traversals, reads):
    key = (
        layer, bool(node.inside_while), int(node.while_number),
        int(node.while_iteration),
    )
    sources = set(traversals.get(key, set()))
    if not sources:
        for field in node.fields:
            sources.update(reads.get((layer, field), set()))
    return sources


def _closed_loops(manifest, traversals):
    # A (layer, while_number) backward-substitution loop's recorded sources
    # are trace-derived: a branch can go unrecorded simply because its
    # coefficient was zero on the traced input, not because it structurally
    # can't happen. Trust the recorded sources for fusion only where they are
    # structurally self-consistent: every non-affine source's own parents
    # (per the input-independent layer graph) also appear somewhere in that
    # same loop's recorded sources -- i.e. nothing upstream looks silently
    # skipped. Loops that fail this, or that recorded no sources at all, are
    # left out, so _rewrite conservatively declines to fuse there.
    layer_types = manifest["layer_types"]
    layer_parents = manifest.get("layer_parents")
    if layer_parents is None:
        return set()
    by_loop = {}
    for (layer, _inside_while, while_number, _while_iteration), sources in (
            traversals.items()):
        by_loop.setdefault((layer, while_number), set()).update(sources)
    closed = set()
    for loop_key, sources in by_loop.items():
        if all(
            layer_types[source] in {"Linear", "Conv2D", "Input"}
            or set(layer_parents[source]).issubset(sources)
            for source in sources
        ):
            closed.add(loop_key)
    return closed


def _rewrite(expr, layer, facts, traversals, reads, affine_sources,
             closed_loops, stats):
    if isinstance(expr, int):
        return expr
    children = [
        _rewrite(
            child, layer, facts, traversals, reads, affine_sources,
            closed_loops, stats)
        for child in expr.children
    ]
    expr.update_parent_child(children)
    split = _semantic_split(expr)
    if split is None:
        return expr
    stats["candidates"] += 1
    if not split.inside_while:
        return expr
    if frozenset(split.fields) != {"L", "U"}:
        return expr
    sources = _site_sources(split, layer, traversals, reads)
    if not sources or not sources.issubset(affine_sources):
        return expr
    if (layer, split.while_number) not in closed_loops:
        return expr
    # pair = tuple(sorted(split.fields))
    # if any(pair not in facts.get(source, set()) for source in sources):
    #     return expr
    fused = _unclamp(split.children[0])
    if fused is None:
        raise RuntimeError("reuse: malformed sign split")
    stats["fused"] += 1
    return fused


def fuse_sign_splits(layer_cfgs, facts, manifest):
    traversals = traversal_sources(manifest)
    reads = read_sources(manifest)
    affine_sources = {
        index for index, name in enumerate(manifest["layer_types"])
        if name in {"Linear", "Conv2D"}
    }
    closed_loops = _closed_loops(manifest, traversals)
    stats = {"candidates": 0, "fused": 0}
    for layer, cfg in layer_cfgs.items():
        block = cfg.ir[cfg.entry_node]
        for statement in block.children:
            if isinstance(statement, IrAssignment):
                value = _rewrite(
                    statement.children[1], layer, facts, traversals, reads,
                    affine_sources, closed_loops, stats)
                statement.update_parent_child([statement.children[0], value])
            elif hasattr(statement, "outputs"):
                statement.update_parent_child([
                    _rewrite(
                        value, layer, facts, traversals, reads,
                        affine_sources, closed_loops, stats)
                    for value in statement.children
                ])
    return stats
