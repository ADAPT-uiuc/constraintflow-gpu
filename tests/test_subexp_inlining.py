"""Unit tests for liveness-based temp-name recycling in subexp_inlining.

The pass renames single-def `ttb_var_*` temporaries onto a minimal pool
(`ttb_r_*`), reusing a pool name only strictly after its current occupant's last
use, so a rebind frees the previous tensor. These tests build tiny straight-line
IR blocks by hand and assert on the resulting name map.
"""

from constraintflow.compiler.ir import (
    IrVar, IrConst, IrAssignment, IrTransRetBasic, IrDel,
    IrTorchFloat, IrTorchMatmul, IrMetadataElement,
)
from constraintflow.compiler.optimizations.subexp_inlining import (
    recycle_temp_names_block,
)


def _meta():
    return [IrMetadataElement([1], 'Float', [1], True)]


def _var(name):
    return IrVar(name, _meta())


def _assign(lhs_name, rhs):
    rhs.irMetadata = _meta()
    return IrAssignment(_var(lhs_name), rhs)


def _unary_use(name):
    """RHS that reads `name` once."""
    return IrTorchFloat(_var(name))


def _binary_use(a, b):
    """RHS that reads `a` and `b` (each once; pass the same name twice to model a
    single statement that dereferences one temp in two places)."""
    return IrTorchMatmul(_var(a), _var(b))


def test_nonoverlapping_ranges_share_a_pool_name():
    # ttb_var_0 dies at its use in stmt 1; ttb_var_2 is defined at stmt 2, so it
    # may reuse ttb_var_0's slot. ttb_var_1 is live to the return -> its own slot.
    instrs = [
        _assign('ttb_var_0', _unary_use('inp')),
        _assign('ttb_var_1', _unary_use('ttb_var_0')),
        _assign('ttb_var_2', _unary_use('inp')),
        IrTransRetBasic([_var('ttb_var_1'), _var('ttb_var_2')]),
    ]
    rename = recycle_temp_names_block(instrs)
    assert rename['ttb_var_0'] == rename['ttb_var_2']      # recycled
    assert rename['ttb_var_0'] != rename['ttb_var_1']      # 1 outlives 0
    # only two pool slots were needed
    assert len(set(rename.values())) == 2


def test_overlapping_ranges_get_distinct_names():
    # Both are read together in the return, so both are live simultaneously.
    instrs = [
        _assign('ttb_var_0', _unary_use('inp')),
        _assign('ttb_var_1', _unary_use('inp')),
        IrTransRetBasic([_binary_use('ttb_var_0', 'ttb_var_1')]),
    ]
    rename = recycle_temp_names_block(instrs)
    assert rename['ttb_var_0'] != rename['ttb_var_1']


def test_multiple_occurrences_in_one_statement_is_one_consumer():
    # ttb_var_0 is dereferenced twice in stmt 1 (models x.blocks[0], x.blocks[1]).
    # Its last use is still stmt 1, so ttb_var_2 (stmt 2) may reuse its slot --
    # the multi-occurrence must not extend liveness or block recycling.
    instrs = [
        _assign('ttb_var_0', _unary_use('inp')),
        _assign('ttb_var_1', _binary_use('ttb_var_0', 'ttb_var_0')),
        _assign('ttb_var_2', _unary_use('inp')),
        IrTransRetBasic([_var('ttb_var_1'), _var('ttb_var_2')]),
    ]
    rename = recycle_temp_names_block(instrs)
    assert rename['ttb_var_0'] == rename['ttb_var_2']


def test_del_markers_are_dropped():
    instrs = [
        _assign('ttb_var_0', _unary_use('inp')),
        IrDel(['ttb_var_0']),
        IrTransRetBasic([_var('ttb_var_0')]),
    ]
    recycle_temp_names_block(instrs)
    assert not any(isinstance(i, IrDel) for i in instrs)


def test_reassigned_names_are_left_untouched():
    # A name with two defs has no single live interval; leave it as-is.
    instrs = [
        _assign('ttb_var_0', _unary_use('inp')),
        _assign('ttb_var_0', _unary_use('inp')),
        IrTransRetBasic([_var('ttb_var_0')]),
    ]
    rename = recycle_temp_names_block(instrs)
    assert 'ttb_var_0' not in rename


def test_renaming_is_applied_to_all_occurrences():
    instrs = [
        _assign('ttb_var_0', _unary_use('inp')),
        IrTransRetBasic([_var('ttb_var_0')]),
    ]
    rename = recycle_temp_names_block(instrs)
    new = rename['ttb_var_0']
    # def LHS renamed
    assert instrs[0].children[0].name == new
    # use in the return renamed
    assert instrs[1].children[0].name == new
