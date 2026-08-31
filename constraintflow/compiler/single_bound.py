import copy

from constraintflow.ast_cflow import astcf as AST
from constraintflow.lib.globals import bound_lower, bound_upper

AFFINE_LAST_OP = 'Affine_last'
AFFINE_SKIP_OP = 'Affine_skip'


def _affine_op_stmts(stmt):
    """Yield every Affine OpStmtNode reachable from a Program's statement tree."""
    if isinstance(stmt, AST.SeqNode):
        yield from _affine_op_stmts(stmt.stmt1)
        yield from _affine_op_stmts(stmt.stmt2)
    elif isinstance(stmt, AST.TransformerNode):
        for op_stmt in stmt.oplist.olist:
            if op_stmt.op.op_name == 'Affine':
                yield stmt, op_stmt


def inject_single_bound(ast_program):
    """Add two extra Affine-shaped ops, Affine_last and Affine_skip, alongside the
    original (untouched) Affine op -- flow_sparse.py's Flow.flow() dispatches each
    Linear/Conv2D layer to whichever of the three applies, using layer.last_layer /
    layer.feeds_nonlin (already computed in parse.py) plus these flags. Leaving the
    original Affine op untouched means the common case (an interior layer feeding a
    nonlinearity) is byte-identical to the unflagged generic CFG that
    tensor_to_block.py specializes -- only the rare last_layer / feeds_nonlin=False
    layers get their own (tiny, unconditional) CFGs. An earlier version instead
    wrapped Affine's own return in a nested ternary; that tripled the generic
    Affine CFG that every layer's reuse-specialization has to process, which made
    unroll_while's live-node analysis blow up superlinearly on deep nets.

    No-op when both flags are set (the default): AST is left untouched, so output
    is byte-identical.
    """
    if bound_lower.get_flag() and bound_upper.get_flag():
        return
    if not bound_lower.get_flag() and not bound_upper.get_flag():
        raise ValueError("At least one of --bound-lower/--bound-upper must be set.")

    keep_lower = bound_lower.get_flag()

    for transformer_node, op_stmt in list(_affine_op_stmts(ast_program.stmt)):
        ret = op_stmt.ret
        if not isinstance(ret, AST.TransRetBasicNode):
            raise ValueError(
                "single_bound: Affine op's return must be a plain (l, u, L, U) "
                f"tuple, not {type(ret).__name__}; --bound-lower/--bound-upper "
                "does not support this transformer."
            )
        exprs = ret.exprlist.exprlist
        if len(exprs) != 4:
            raise ValueError(
                "single_bound: Affine op's return must have 4 components "
                f"(l, u, L, U), got {len(exprs)}."
            )
        lower_expr, upper_expr, l_poly, u_poly = exprs
        kept_expr = lower_expr if keep_lower else upper_expr

        last_ret = AST.TransRetBasicNode(AST.ExprListNode([
            copy.deepcopy(kept_expr) if keep_lower else AST.ConstIntNode(0),
            AST.ConstIntNode(0) if keep_lower else copy.deepcopy(kept_expr),
            copy.deepcopy(l_poly),
            copy.deepcopy(u_poly),
        ]))
        skip_ret = AST.TransRetBasicNode(AST.ExprListNode([
            AST.ConstIntNode(0), AST.ConstIntNode(0),
            copy.deepcopy(l_poly), copy.deepcopy(u_poly),
        ]))

        transformer_node.oplist.olist.append(
            AST.OpStmtNode(AST.OperatorNode(AFFINE_LAST_OP), last_ret))
        transformer_node.oplist.olist.append(
            AST.OpStmtNode(AST.OperatorNode(AFFINE_SKIP_OP), skip_ret))
