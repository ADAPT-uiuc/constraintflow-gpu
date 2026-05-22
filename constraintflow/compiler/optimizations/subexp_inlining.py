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


def used_vars(statement: IrStatement) -> set[IrVar]:
    """private"""
    if isinstance(statement, IrDel):
        return set()
    if isinstance(statement, IrAssignment):
        stack = [statement.children[1]]
    else:
        stack = list(statement.children)
    vars = set()
    visited = set()
    while stack:
        node = stack.pop()
        if isinstance(node, int):
            continue
        if isinstance(node, (list, tuple)):
            stack.extend(node)
            continue
        if not isinstance(node, IrAst):
            continue
        node_id = id(node)
        if node_id in visited:
            continue
        visited.add(node_id)
        for metadata in node.irMetadata or []:
            if metadata is None:
                continue
            for item in metadata.shape:
                if isinstance(item, IrAst):
                    stack.append(item)
            for item in metadata.broadcast:
                if isinstance(item, IrAst):
                    stack.append(item)
        if isinstance(node, IrVar):
            vars.add(node)
        else:
            stack.extend(node.children)
    return vars


def replace_var_uses(var_name: str, replacement: IrExpression,
                     statement: IrStatement ) -> None:
    """private"""
    children = statement.children
    start = 1 if isinstance(statement, IrAssignment) else 0
    stack = []
    for i in range(start, len(children)):
        child = children[i]
        if isinstance(child, IrVar) and child.name == var_name:
            children[i] = replacement
        else:
            stack.append(child)
    while stack:
        node = stack.pop()
        if isinstance(node, int):
            continue
        if isinstance(node, list):
            for i in range(len(node)):
                child = node[i]
                if isinstance(child, IrVar) and child.name == var_name:
                    node[i] = replacement
                else:
                    stack.append(child)
            continue
        if isinstance(node, tuple):
            stack.extend(node)
            continue
        if not isinstance(node, IrAst):
            continue
        for metadata in node.irMetadata or []:
            if metadata is None:
                continue
            shape = metadata.shape
            for i in range(len(shape)):
                item = shape[i]
                if isinstance(item, IrVar) and item.name == var_name:
                    shape[i] = replacement
                else:
                    stack.append(item)
            broadcast = metadata.broadcast
            for i in range(len(broadcast)):
                item = broadcast[i]
                if isinstance(item, IrVar) and item.name == var_name:
                    broadcast[i] = replacement
                else:
                    stack.append(item)
        children = node.children
        for i in range(len(children)):
            child = children[i]
            if isinstance(child, IrVar) and child.name == var_name:
                children[i] = replacement
            else:
                stack.append(child)


def inline_subexp_block(block: IrBlock) -> None:
    """private"""
    instructions: list[IrStatement] = block.children
    assert all(isinstance(instr, IrStatement) for instr in instructions)
    i = 0
    while i < len(instructions):
        if isinstance(instructions[i], IrAssignment):
            instrs_using_def_i: list[int] = []
            this_def = instructions[i].children[0]
            assert isinstance(this_def, IrVar)
            this_def_name: str = this_def.name
            print(f'this_def_name: {this_def_name}')
            # with open('z_name.txt', 'a') as f:
            #     f.write(this_def_name + '\n')
            for j in range(i + 1, len(instructions)):
                # Until next re-definition of the same variable.
                if (isinstance(instructions[j], IrAssignment)
                    and instructions[j].children[0].name == this_def_name):
                    break
                instr_j_uses = used_vars(instructions[j])
                instr_j_uses_names: set[str] = \
                    set(var.name for var in instr_j_uses)
                if this_def_name in instr_j_uses_names:
                    instrs_using_def_i.append(j)
            if len(instrs_using_def_i) == 0:
                del instructions[i]
            elif len(instrs_using_def_i) == 1:
                def_of_instr_i = instructions[i].children[1]
                assert isinstance(def_of_instr_i, IrExpression)
                # print(f'this_def_name: {this_def_name}')
                # if isinstance(instructions[instrs_using_def_i[0]], IrAssignment):
                #     print(f'used by assignment to {instructions[instrs_using_def_i[0]].children[0].name}')
                #     afsjudli
                replace_var_uses(
                    this_def_name, def_of_instr_i,
                    instructions[instrs_using_def_i[0]])
                del instructions[i]
            else:
                i += 1
        else:
            i += 1


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
