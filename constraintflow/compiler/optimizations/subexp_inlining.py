"""
Based on def-use analysis on non-SSA IR.
If an intermediate variable is only used once, we inline its definition at the
    use site and remove the intermediate variable.
Normally in Python, a variable is released only at the end of the function.
This opt prevents the delay of the release of the intermediate value's memory.
"""

from __future__ import annotations
from constraintflow.compiler.ir import *
from constraintflow.compiler.optimizations import uses
from constraintflow.compiler.representations import Graph
from constraintflow.compiler.optimizations.tensor_to_block import \
    replace_all_occurrences_expr
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


def recursively_find_def_expr(
        instructions: list[IrStatement], var: IrVar,
        current_vars_def_index: dict[str, int]) -> IrExpression:
    var_name = var.name
    if var_name not in current_vars_def_index.keys():
        return var
    immediate_def_index = current_vars_def_index[var_name]
    assert isinstance(instructions[immediate_def_index], IrAssignment)
    immediate_def_expr = instructions[immediate_def_index].children[1]
    if not isinstance(immediate_def_expr, IrVar):
        return immediate_def_expr
    else:
        return recursively_find_def_expr(
            instructions, immediate_def_expr, current_vars_def_index)


def get_vars_expr_occurrences(expr) -> list[IrVar]:
    if isinstance(expr, (int, float, list)):
        return []
    if isinstance(expr, IrVar):
        return [expr]
    vars: list[IrVar] = []
    if expr is None:
        return vars
    for child in expr.children:
        vars.extend(get_vars_expr_occurrences(child))
    return vars


def collect_op_var_names(expr) -> set:
    """Names of IrVars that appear in an ``op`` position (outside ``children``).

    Some nodes -- e.g. IrSimpleUnary -- carry an operand in ``self.op`` rather
    than in children; visitIrSimpleUnary renders it as ``(op)(operand)``. Such a
    variable is invisible to the children-based def-use walk, so without this it
    would look dead (wrongly deleted) and, because the inliner only substitutes
    children, an expression folded into ``op`` would also break codeGen (it
    requires op to be an IrVar or a known string). We therefore *pin* every such
    name: never inlined, never deleted -- leaving its assignment intact.
    """
    names: set = set()
    if expr is None or isinstance(expr, (int, float, list)):
        return names
    op = getattr(expr, 'op', None)
    if isinstance(op, IrVar):
        names.add(op.name)
    if isinstance(expr, IrVar):
        return names
    for child in expr.children:
        names |= collect_op_var_names(child)
    return names


def indices_to_delete_and_replace_single_use(block: IrBlock):
    instructions: list[IrStatement] = block.children
    current_vars_def_index: dict[str, int] = {}
    uses_instr_count: dict[str, list[int]] = {}
    to_delete_indices: list[int] = []
    name_to_var: dict[str, IrVar] = {}
    # Variables used in an op position (outside children); never inline/delete.
    pinned: set = set()
    for instr in instructions:
        if isinstance(instr, IrAssignment):
            pinned |= collect_op_var_names(instr.children[1])
        elif isinstance(instr, IrTransRetBasic):
            for c in instr.children:
                pinned |= collect_op_var_names(c)
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
            if (defined_var.name in current_vars_def_index.keys()
                    and defined_var.name not in pinned):
                if len(uses_instr_count[defined_var.name]) == 1:
                    use_instr_index = uses_instr_count[defined_var.name][0]
                    bottom_def = recursively_find_def_expr(
                        instructions, defined_var, current_vars_def_index)
                    replace_var_with_expr(
                        instructions, use_instr_index,
                        bottom_def,
                        defined_var)
                    to_delete_indices.append(
                        current_vars_def_index[defined_var.name])
                    
                elif len(uses_instr_count[defined_var.name]) == 0:
                    to_delete_indices.append(
                        current_vars_def_index[defined_var.name])
            current_vars_def_index[defined_var.name] = i
            uses_instr_count[defined_var.name] = []
    
    for var in current_vars_def_index.keys():
        if var in pinned:
            continue
        if var in uses_instr_count.keys() and len(uses_instr_count[var]) == 1:
            use_instr_index = uses_instr_count[var][0]
            bottom_def = recursively_find_def_expr(
                        instructions, name_to_var[var], current_vars_def_index)
            replace_var_with_expr(
                instructions, use_instr_index,
                bottom_def, name_to_var[var])
            to_delete_indices.append(current_vars_def_index[var])
        elif (var not in uses_instr_count.keys()
              or len(uses_instr_count[var]) == 0):
            to_delete_indices.append(current_vars_def_index[var])
    
    return to_delete_indices
                


def inline_subexp_block(block: IrBlock) -> None:
    """private"""
    while True:
        to_delete_indices = indices_to_delete_and_replace_single_use(block)
        if len(to_delete_indices) == 0:
            break
        for i in sorted(set(to_delete_indices), reverse=True):
            del block.children[i]


def inline_subexp_funcdef(funcdef) -> None:
    """private
    Inline single-use intermediates inside one wrapped-expansion function
    (an IrTtbFuncDef). The body is straight-line and SSA, so this is the easy
    case. We append a sentinel return node so that `return <return_var>` counts
    as the final use of return_var -- a single-use return value then folds into
    the return expression like any other single-use intermediate.
    """
    sentinel = IrTransRetBasic([funcdef.return_var])
    fake_block = IrBlock(funcdef.body + [sentinel])
    inline_subexp_block(fake_block)
    funcdef.body = fake_block.children[:-1]
    funcdef.return_var = fake_block.children[-1].children[0]


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
    - For wrapped (function-per-expansion) IR: tensor_to_block has run and
        CodeGen.finalize_ttb_params has populated funcdef.params and the
        IrTtbCall argument children, so this pass sees each call's dependencies.
    """
    # Inline inside the lifted per-expansion functions (straight-line, SSA).
    for funcdef in getattr(ir, 'ttb_funcdefs', None) or []:
        inline_subexp_funcdef(funcdef)
    # Inline inside the concrete transformer methods (Affine/Relu), where the
    # win is folding one ttb_func call into the next via its argument children.
    for transformer in ir.tstore.keys():
        for i in range(len(ir.tstore[transformer])):
            transformer_ir = ir.tstore[transformer][i]
            if transformer_ir.layerwise_cfgs is not None:
                for cfg in transformer_ir.layerwise_cfgs.values():
                    inline_subexp_cfg(cfg)
            else:
                cfg: Graph = transformer_ir.cfg
                inline_subexp_cfg(cfg)
