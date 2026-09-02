from constraintflow.compiler.ir import (
    IrAssignment, IrBreak, IrDeadValue, IrTransRetBasic, IrVar,
)
from constraintflow.lib.jit_semantics import live_outputs


def _vars(expr, result=None):
    if result is None:
        result = set()
    if isinstance(expr, int):
        return result
    if isinstance(expr, IrVar):
        result.add(expr.name)
        return result
    for child in expr.children:
        _vars(child, result)
    return result


def _slice(block):
    needed = set()
    kept = []
    for statement in reversed(block.children):
        if isinstance(statement, IrTransRetBasic):
            for value in statement.children:
                if not isinstance(value, IrDeadValue):
                    _vars(value, needed)
            kept.append(statement)
        elif isinstance(statement, IrAssignment):
            target, value = statement.children
            if not isinstance(target, IrVar):
                raise RuntimeError("reuse: unsupported assignment target")
            if target.name in needed:
                needed.remove(target.name)
                _vars(value, needed)
                kept.append(statement)
        elif isinstance(statement, IrBreak):
            continue
        else:
            raise RuntimeError(
                f"reuse: unsupported statement {type(statement).__name__}")
    block.update_parent_child(list(reversed(kept)))


def _merge_parents(manifest):
    # Layers feeding a multi-parent (Concat/Add) join. live_outputs() only
    # knows what this one trace actually read; a join's own branch can look
    # unread simply because its coefficient was zero on the traced input,
    # not because nothing downstream can ever need it. Never trust "unread"
    # for these -- treat their outputs as always live.
    parents = set()
    for layer_parents in manifest.get("layer_parents", []):
        if len(layer_parents) > 1:
            parents.update(layer_parents)
    return parents


def eliminate_dead_outputs(layer_cfgs, manifest):
    if not manifest["capabilities"].get("dead_outputs", False):
        return {"disabled": True, "dead": 0, "assignments": 0}
    live = live_outputs(manifest)
    merge_parents = _merge_parents(manifest)
    dead = 0
    removed = 0
    for layer, cfg in layer_cfgs.items():
        block = cfg.ir[cfg.entry_node]
        before = len(block.children)
        returns = [item for item in block.children
                   if isinstance(item, IrTransRetBasic)]
        if len(returns) != 1:
            raise RuntimeError("reuse: expected one transformer return")
        ret = returns[0]
        outputs = []
        for field, value in ret.outputs.items():
            if (layer, field) in live or layer in merge_parents:
                outputs.append(value)
            else:
                outputs.append(IrDeadValue(field, value.irMetadata))
                dead += 1
        ret.update_parent_child(outputs)
        _slice(block)
        removed += before - len(block.children)
    return {"disabled": False, "dead": dead, "assignments": removed}
