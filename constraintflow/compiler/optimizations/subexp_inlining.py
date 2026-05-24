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


def indices_to_delete_and_replace_single_use(block: IrBlock):
    instructions: list[IrStatement] = block.children
    current_vars_def_index: dict[str, int] = {}
    uses_instr_count: dict[str, list[int]] = {}
    to_delete_indices: list[int] = []
    name_to_var: dict[str, IrVar] = {}
    for i in range(len(instructions)):
        if isinstance(instructions[i], IrDel):
            continue
        used_vars: set[str]
        temp: set[IrVar]
        if isinstance(instructions[i], IrAssignment):
            temp = get_vars_expr(instructions[i].children[1])
        elif isinstance(instructions[i], IrTransRetBasic):
            temp = set()
            for j in range(len(instructions[i].children)):
                temp = temp.union(
                    get_vars_expr(instructions[i].children[j]))
        else:
            assert False, f'Unexpected instruction type: {type(instructions[i])}'
        used_vars = set(var.name for var in temp)
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
