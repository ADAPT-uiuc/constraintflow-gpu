"""
Based on def-use analysis on non-SSA IR.
If an intermediate variable is only used once, we inline its definition at the
    use site and remove the intermediate variable.
Normally in Python, a variable is released only at the end of the function.
This opt prevents the delay of the release of the intermediate value's memory.
"""

from __future__ import annotations
import bisect
from constraintflow.compiler.ir import *
from constraintflow.compiler.optimizations import uses
from constraintflow.compiler.representations import Graph
from constraintflow.compiler.optimizations.loopInvariantCodeMotion import \
    get_vars_expr


# def replace_var_with_expr(             
#         instructions: list[IrStatement], use_instr_index: int,
#         def_instr_index: int, var_name: str) -> None:
#     if isinstance(instructions[use_instr_index], IrAssignment):
#         new_expr = replace_all_occurrences_expr(
#             instructions[use_instr_index].children[1],
#             {var_name: instructions[def_instr_index].children[1]})
#         new_children = [instructions[def_instr_index].children[0], new_expr]
#         instructions[use_instr_index].update_parent_child(new_children)
#     elif isinstance(
#         instructions[use_instr_index], IrTransRetBasic):
#         new_children = []
#         for j in range(len(instructions[use_instr_index].children)):
#             new_expr = replace_all_occurrences_expr(
#             instructions[use_instr_index].children[j],
#             {var_name: instructions[def_instr_index].children[1]})
#             new_children.append(new_expr)
#         instructions[use_instr_index].update_parent_child(new_children)
#     else:
#         assert False


def get_generalized_children(expr) -> list[IrExpression]:
    ret: list[IrExpression] = []
    if isinstance(expr, IrSimpleUnary) and isinstance(expr.op, IrVar):
        ret.append(expr.op)
        ret.extend(expr.children)
    else:
        ret.extend(expr.children)
    return ret


def replace_all_occurrences_expr(expr, var_map: dict[str, IrExpression]):
    if isinstance(expr, IrVar) and expr.name in var_map.keys(): 
        return var_map[expr.name]
    if isinstance(expr, (int, float)):
        return expr
    if expr is None:
        return expr
    if isinstance(expr, list):
        return [replace_all_occurrences_expr(x, var_map) for x in expr]
    has_generalized_op_child = (
        isinstance(expr, IrSimpleUnary) and isinstance(expr.op, IrVar))
    generalized_children = get_generalized_children(expr)
    for i in range(len(generalized_children)):
        new_child = replace_all_occurrences_expr(generalized_children[i], var_map)
        if has_generalized_op_child and i == 0:
            expr.op = new_child
        elif has_generalized_op_child and i != 0:
            expr.children[i - 1] = new_child
        else:
            expr.children[i] = new_child
    return expr


def replace_var_with_expr(             
        instructions: list[IrStatement], use_instr_index: int,
        replace_expr: IrExpression, var: IrVar) -> None:
    var_name = var.name
    if isinstance(instructions[use_instr_index], IrAssignment):
        new_expr = replace_all_occurrences_expr(
            instructions[use_instr_index].children[1],
            {var_name: replace_expr})
        new_children = [instructions[use_instr_index].children[0], new_expr]
        instructions[use_instr_index].update_parent_child(new_children)
    elif isinstance(
        instructions[use_instr_index], IrTransRetBasic):
        new_children = []
        for j in range(len(instructions[use_instr_index].children)):
            new_expr = replace_all_occurrences_expr(
            instructions[use_instr_index].children[j],
            {var_name: replace_expr})
            new_children.append(new_expr)
        instructions[use_instr_index].update_parent_child(new_children)
    else:
        assert False


def compute_def_indices(
        instructions: list[IrStatement]) -> dict[str, list[int]]:
    """For each variable, the ascending list of statement indices that assign
    it. On a straight-line block (no branches/merges) this is all the
    reaching-definition information we need: the definition reaching any point
    is simply the nearest preceding one."""
    def_indices: dict[str, list[int]] = {}
    for idx, instr in enumerate(instructions):
        if isinstance(instr, IrAssignment):
            lhs = instr.children[0]
            if isinstance(lhs, IrVar):
                def_indices.setdefault(lhs.name, []).append(idx)
    return def_indices


def reaching_def_index(
        def_indices: dict[str, list[int]], name: str, point: int):
    """Index of the definition of `name` immediately preceding `point`
    (strictly earlier), or None if `name` has no definition in the block
    (i.e. it is a block-external value such as a transformer parameter)."""
    idxs = def_indices.get(name)
    if not idxs:
        return None
    j = bisect.bisect_left(idxs, point) - 1
    return idxs[j] if j >= 0 else None


def redefined_between(
        def_indices: dict[str, list[int]], name: str, lo: int, hi: int) -> bool:
    """True iff `name` is assigned at some index in the open interval (lo, hi)."""
    idxs = def_indices.get(name)
    if not idxs:
        return False
    k = bisect.bisect_right(idxs, lo)
    return k < len(idxs) and idxs[k] < hi


def resolve_value(
        instructions: list[IrStatement], def_indices: dict[str, list[int]],
        var: IrVar, point: int):
    """
    Return (expr, def_index): an expression equal to `var`'s value at `point`,
    together with the index of the statement whose RHS that expression is
    (or -1 when the value is a block-external variable / parameter).

    Copy chains (`t = a; a = expr`) are followed through *reaching* definitions:
    the value of `var` at `point` equals the value of its RHS variable at
    `var`'s own definition site, and so on. Because each hop moves to a strictly
    earlier index, this terminates, and it binds to the correct definition even
    for reassigned variables.
    """
    d = reaching_def_index(def_indices, var.name, point)
    if d is None:
        return var, -1
    rhs = instructions[d].children[1]
    if isinstance(rhs, IrVar):
        return resolve_value(instructions, def_indices, rhs, d)
    return rhs, d


def get_vars_expr_occurrences(expr) -> list[IrVar]:
    if isinstance(expr, (int, float)):
        return []
    if isinstance(expr, IrVar):
        return [expr]
    if isinstance(expr, list):
        vars: list[IrVar] = []
        for x in expr:
            vars.extend(get_vars_expr_occurrences(x))
        return vars
    vars: list[IrVar] = []
    if expr is None:
        return vars
    for child in get_generalized_children(expr):
        vars.extend(get_vars_expr_occurrences(child))
    return vars


def is_safe_to_inline(
        def_indices: dict[str, list[int]], expr, def_index: int,
        use_index: int) -> bool:
    """
    Moving `expr` (originally evaluated at `def_index`) to `use_index` is
    value-preserving iff none of the variables it reads is reassigned in the
    open interval (def_index, use_index) -- i.e. the definition of each read
    variable reaching the use is the same one that reached the original site.

    The interval is open at both ends: `def_index` defines the temporary and
    only reads `expr`'s variables (it does not reassign them), and at
    `use_index` the RHS is evaluated before that statement's own assignment, so
    a reassignment there does not affect `expr`.
    """
    for var in get_vars_expr_occurrences(expr):
        if redefined_between(def_indices, var.name, def_index, use_index):
            return False
    return True


def is_trivial(expr) -> bool:
    """A leaf that is free to recompute: a bare variable reference or a literal.
    Only such expressions may be folded into more than one use site -- copying
    anything heavier would duplicate real work."""
    return isinstance(expr, (IrVar, IrConst))


def copy_leaf(expr):
    """Fresh copy of a trivial leaf, so folding one value into several use sites
    never leaves them sharing a single node object. Non-leaves are returned
    unchanged (they are only ever inlined into a single site)."""
    if isinstance(expr, IrVar):
        return IrVar(expr.name, expr.irMetadata)
    if isinstance(expr, IrConst):
        return IrConst(expr.const, expr.irMetadata[-1].type)
    return expr


def try_inline_definition(
        instructions: list[IrStatement], def_indices: dict[str, list[int]],
        var: IrVar, def_stmt_index: int, use_indices: list[int],
        to_delete_indices: list[int]) -> None:
    """
    Fold `var`'s definition (at `def_stmt_index`, whose uses since that
    definition are `use_indices`) into its use sites:

      - 0 uses  -> the definition is dead; delete it.
      - 1 use   -> inline if the move is value-preserving.
      - >1 uses -> only if the resolved value is a trivial leaf (a constant or a
                   bare variable), which is free to duplicate; inline into every
                   use where the move is safe, and delete the definition only if
                   *all* uses were inlined (otherwise it is still live).

    `resolve_value` returns the bottom of the copy chain, so a trivial result is
    always either a literal or a block-external variable -- both always safe to
    duplicate -- whereas an alias that bottoms out in a heavy expression is
    (correctly) not treated as trivial.
    """
    if not use_indices:
        to_delete_indices.append(def_stmt_index)
        return
    inline_expr, value_def_index = resolve_value(
        instructions, def_indices, var, use_indices[0])
    if len(use_indices) > 1 and not is_trivial(inline_expr):
        return
    inlined_all = True
    for use_index in use_indices:
        if is_safe_to_inline(def_indices, inline_expr, value_def_index,
                             use_index):
            replace_var_with_expr(
                instructions, use_index, copy_leaf(inline_expr), var)
        else:
            inlined_all = False
    if inlined_all:
        to_delete_indices.append(def_stmt_index)


def indices_to_delete_and_replace_single_use(block: IrBlock):
    instructions: list[IrStatement] = block.children
    def_indices: dict[str, list[int]] = compute_def_indices(instructions)
    current_vars_def_index: dict[str, int] = {}
    uses_instr_count: dict[str, list[int]] = {}
    to_delete_indices: list[int] = []
    name_to_var: dict[str, IrVar] = {}
    for i in range(len(instructions)):
        if isinstance(instructions[i], IrDel):
            continue
        used_vars: list[str]
        temp: list[IrVar]
        if isinstance(instructions[i], IrAssignment):
            temp = get_vars_expr_occurrences(instructions[i].children[1])
        elif isinstance(instructions[i], IrTransRetBasic):
            temp = []
            for j in range(len(instructions[i].children)):
                temp.extend(
                    get_vars_expr_occurrences(instructions[i].children[j]))
        else:
            assert False, f'Unexpected instruction type: {type(instructions[i])}'
        used_vars = [var.name for var in temp]
        for var in used_vars:
            if var not in current_vars_def_index.keys():
                continue
            if var not in uses_instr_count.keys():
                uses_instr_count[var] = [i]
            else:
                uses_instr_count[var].append(i)
        if isinstance(instructions[i], IrAssignment):
            defined_var = instructions[i].children[0]
            name_to_var[defined_var.name] = defined_var
            assert isinstance(defined_var, IrVar)
            if defined_var.name in current_vars_def_index.keys():
                try_inline_definition(
                    instructions, def_indices, defined_var,
                    current_vars_def_index[defined_var.name],
                    uses_instr_count[defined_var.name], to_delete_indices)
            current_vars_def_index[defined_var.name] = i
            uses_instr_count[defined_var.name] = []
    
    for var in current_vars_def_index.keys():
        try_inline_definition(
            instructions, def_indices, name_to_var[var],
            current_vars_def_index[var],
            uses_instr_count.get(var, []), to_delete_indices)

    return to_delete_indices
                


def inline_subexp_block(block: IrBlock) -> None:
    """private"""
    while True:
        to_delete_indices = indices_to_delete_and_replace_single_use(block)
        if len(to_delete_indices) == 0:
            break
        for i in sorted(set(to_delete_indices), reverse=True):
            del block.children[i]


def inline_subexp_cfg(cfg: Graph) -> None:
    """
    private
    Currently, we assume that the IR we apply on has no control flow.
    So we do not need to deal with `jump` and `inner_jump` of IrBlock here.
    This is ineed the case in the current lowered-to-block IR.
    This function should be amenable to future extension to control flow.
    """
    # assert len(cfg.nodes) == 1
    if len(cfg.nodes) != 1:
        return
    block = cfg.ir[cfg.nodes[0]]
    inline_subexp_block(block)


def inline_subexp(ir: IrProgram) -> None:
    """
    public
    Requires:
    - `ir` is already optimized to block level.
    - Every abstract OP tranformer in `ir` has no control flow (i.e., only one
        block in its CFG).
    """
    for transformer in ir.tstore.keys():
        for i in range(len(ir.tstore[transformer])):
            transformer_ir = ir.tstore[transformer][i]
            if transformer_ir.layerwise_cfgs is not None:
                for cfg in transformer_ir.layerwise_cfgs.values():
                    inline_subexp_cfg(cfg)
            else:
                cfg: Graph = transformer_ir.cfg
                inline_subexp_cfg(cfg)
