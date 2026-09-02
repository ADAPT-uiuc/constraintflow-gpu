from constraintflow.compiler.ir import IrAst, IrAssignment, IrConst, IrVar


_IGNORED = {
    "identifier", "parents", "children", "defs", "uses", "ttb_counter",
    "inside_while", "while_number", "while_iteration", "irMetadata",
}


def _scalar(value, definitions, seen):
    if isinstance(value, IrAst):
        return fingerprint(value, definitions, seen)
    if isinstance(value, (list, tuple)):
        return tuple(_scalar(item, definitions, seen) for item in value)
    if isinstance(value, dict):
        return tuple(sorted(
            (str(key), _scalar(item, definitions, seen))
            for key, item in value.items()))
    if hasattr(value, "tolist"):
        return _scalar(value.tolist(), definitions, seen)
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return repr(value)


def fingerprint(expr, definitions, seen=None):
    if seen is None:
        seen = set()
    if isinstance(expr, int):
        return ("int", expr)
    if isinstance(expr, IrVar):
        if expr.name in definitions and expr.name not in seen:
            return fingerprint(
                definitions[expr.name], definitions, seen | {expr.name})
        return ("var", expr.name)
    if isinstance(expr, IrConst):
        return ("const", expr.const, expr.irMetadata[-1].type)
    attrs = tuple(sorted(
        (name, _scalar(value, definitions, seen))
        for name, value in vars(expr).items()
        if name not in _IGNORED
    ))
    children = tuple(fingerprint(child, definitions, seen)
                     for child in expr.children)
    metadata = tuple(
        (
            item.type, item.isConst,
            _scalar(item.shape, definitions, seen),
            _scalar(item.broadcast, definitions, seen),
        )
        for item in expr.irMetadata
    )
    return (type(expr).__name__, attrs, metadata, children)


def output_fingerprints(cfg):
    block = cfg.ir[cfg.entry_node]
    definitions = {}
    outputs = None
    for statement in block.children:
        if isinstance(statement, IrAssignment):
            target, value = statement.children
            if isinstance(target, IrVar):
                definitions[target.name] = value
        elif hasattr(statement, "outputs"):
            outputs = {
                field: fingerprint(value, definitions)
                for field, value in statement.outputs.items()
            }
    if outputs is None:
        raise RuntimeError("reuse: transformer has no return")
    return outputs


def equality_facts(layer_cfgs):
    facts = {}
    for layer, cfg in layer_cfgs.items():
        outputs = output_fingerprints(cfg)
        layer_facts = set()
        names = list(outputs)
        for index, lhs in enumerate(names):
            for rhs in names[index + 1:]:
                if outputs[lhs] == outputs[rhs]:
                    layer_facts.add(tuple(sorted((lhs, rhs))))
        facts[layer] = layer_facts
    return facts
