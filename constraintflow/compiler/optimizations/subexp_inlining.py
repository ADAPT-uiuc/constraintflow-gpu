"""
Def-use based subexpression inlining on non-SSA IR.

If an intermediate variable is used exactly once, inline its definition at the
use site and delete the variable. Python releases locals only at function end,
so folding the value frees the intermediate's memory earlier.
"""

from __future__ import annotations
import bisect
from constraintflow.compiler.ir import *
from constraintflow.compiler.representations import Graph


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
    """Map each variable to the ascending list of statement indices that assign
    it. On a straight-line block the reaching definition at any point is just
    the nearest preceding one."""
    def_indices: dict[str, list[int]] = {}
    for idx, instr in enumerate(instructions):
        if isinstance(instr, IrAssignment):
            lhs = instr.children[0]
            if isinstance(lhs, IrVar):
                def_indices.setdefault(lhs.name, []).append(idx)
    return def_indices


def reaching_def_index(
        def_indices: dict[str, list[int]], name: str, point: int):
    """Index of the definition of `name` strictly before `point`, or None if
    `name` is never defined in the block (e.g. a transformer parameter)."""
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
    and the index of the statement whose RHS it is (-1 for a block-external
    variable).

    Copy chains (`t = a; a = expr`) are followed through reaching definitions.
    Each hop moves to a strictly earlier index, so this terminates and binds to
    the correct definition even for reassigned variables.
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
    Moving `expr` from `def_index` to `use_index` preserves its value iff none
    of the variables it reads is reassigned in the open interval
    (def_index, use_index). Both ends are open: `def_index` only reads those
    variables, and at `use_index` the RHS is evaluated before that statement's
    own assignment.
    """
    for var in get_vars_expr_occurrences(expr):
        if redefined_between(def_indices, var.name, def_index, use_index):
            return False
    return True


def is_trivial(expr) -> bool:
    """A leaf that is free to recompute: a bare variable or a literal. Only
    these may be folded into more than one use site; copying anything heavier
    would duplicate real work."""
    return isinstance(expr, (IrVar, IrConst))


def copy_leaf(expr):
    """Fresh copy of a trivial leaf so folding into several use sites never
    leaves them sharing one node object. Non-leaves (only ever inlined into a
    single site) are returned unchanged."""
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
    Fold `var`'s definition (at `def_stmt_index`, with uses `use_indices`) into
    its use sites:

      - 0 uses  -> dead definition; delete it.
      - 1 use   -> inline if the move is value-preserving.
      - >1 uses -> only if the resolved value is a trivial leaf (free to
                   duplicate); inline into every safe use, and delete the
                   definition only if all uses were inlined.

    `resolve_value` follows the copy chain to the bottom, so a trivial result is
    always a literal or a block-external variable (both safe to duplicate); an
    alias bottoming out in a heavy expression is correctly not trivial.
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


def indices_to_delete_and_replace_single_use(
        instructions: list[IrStatement]):
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
                


def inline_fixpoint(instructions: list[IrStatement]) -> None:
    """Iterate the delete/replace pass to a fixpoint over a straight-line
    instruction list. Uses are mutated in place; deleted definitions are removed
    from `instructions` in place, so a caller holding the list sees them."""
    while True:
        to_delete_indices = indices_to_delete_and_replace_single_use(
            instructions)
        if len(to_delete_indices) == 0:
            break
        for i in sorted(set(to_delete_indices), reverse=True):
            del instructions[i]


def inline_subexp_block(block: IrBlock) -> None:
    """private"""
    inline_fixpoint(block.children)


def inline_subexp_cfg(cfg: Graph) -> None:
    """
    private
    Inline single-use temporaries within one transformer CFG.

    Reuse-mode per-layer CFGs are collapsed to a single straight-line block by
    `tensor_to_block` (control flow resolved along the profiled live path), so
    the common case is inlined directly. A CFG that still has control flow (e.g.
    an unsupported transformer op that was not specialized) has no linearizable
    live path here and is left unchanged.
    """
    if len(cfg.nodes) == 1:
        inline_subexp_block(cfg.ir[cfg.nodes[0]])


def inline_subexp(ir: IrProgram) -> None:
    """
    public
    Requires:
    - `ir` is already optimized to block level.
    - `tensor_to_block` has run: per-layer CFGs are collapsed to a single
      straight-line block, with control flow resolved along the profiled live
      path.
    """
    for transformer in ir.tstore.keys():
        for i in range(len(ir.tstore[transformer])):
            transformer_ir = ir.tstore[transformer][i]
            if transformer_ir.layerwise_cfgs is not None:
                for cfg in transformer_ir.layerwise_cfgs.values():
                    inline_subexp_cfg(cfg)
            else:
                inline_subexp_cfg(transformer_ir.cfg)
