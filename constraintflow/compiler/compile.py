import antlr4 as antlr

from constraintflow.ast_cflow import dslLexer
from constraintflow.ast_cflow import dslParser
from constraintflow.ast_cflow import astBuilder
from constraintflow.ast_cflow import astTC
from constraintflow.compiler import convertToIr as c2r
from constraintflow.compiler import representations
from constraintflow.compiler import codeGen
from constraintflow.compiler import builtin_ops
from constraintflow.compiler.optimizations import tensor_to_block
from constraintflow.compiler.optimizations import polyOpt
from constraintflow.compiler.optimizations import symexpCount
from constraintflow.compiler.optimizations import loopInvariantCodeMotion
from constraintflow.compiler.optimizations import copyPropagation
from constraintflow.compiler.optimizations import dce
from constraintflow.compiler.optimizations import cse
from constraintflow.compiler.optimizations import rewrite
from constraintflow.compiler.optimizations import subexp_inlining
from constraintflow.compiler.optimizations import constant_folding
from constraintflow.compiler.optimizations import sroa as sroa_pass
from constraintflow.compiler import single_bound
from constraintflow.lib.globals import *


optimizations_rewrite = [
    cse.cse,
    copyPropagation.copy_proagate,
    cse.cse,
    copyPropagation.copy_proagate,
    polyOpt.poly_opt,
    cse.cse,
    dce.dce,
    rewrite.rewrite,
    cse.cse,
    copyPropagation.copy_proagate,
    dce.dce,
    dce.dce,
    dce.dce,
    loopInvariantCodeMotion.licm,
    cse.cse,
    copyPropagation.copy_proagate,
    cse.cse,
    copyPropagation.copy_proagate,
    cse.cse,
    copyPropagation.copy_proagate,
    cse.cse,
    symexpCount.correct_symexp_size,
    copyPropagation.copy_proagate,
    ]


def sroa_build():
    """True only on the reuse pass of an --sroa build."""
    return reuse_mode.get_flag() and fused_flow.get_flag() and sroa.get_flag()


def _reset_compiler_state():
    representations.while_counter = -1
    rewrite.counter = -1
    rewrite.ttb_counter = 0
    cse.counter = 0
    symexpCount.counter = -1
    tensor_to_block.counter = -1


def compile(inputfile, output_path):
    _reset_compiler_state()
    lexer = dslLexer.dslLexer(antlr.FileStream(inputfile))
    tokens = antlr.CommonTokenStream(lexer)
    parser = dslParser.dslParser(tokens)
    tree = parser.prog()
    
    ast = astBuilder.ASTBuilder().visit(tree)
    if fuse_affine_subst.get_flag():
        single_bound.inject_affine_skip(ast)
    astTC.ASTTC().visit(ast)
    
    converter = c2r.ConvertToIr()
    ir = converter.visit(ast)
    builtin_ops.inject_builtin_ops(ir, converter)
    representations.ssa(ir)

    optimizations = optimizations_rewrite

    for opt in optimizations:
        opt(ir)
    representations.remove_phi(ir)

    replay = reuse_mode.get_flag()

    if reuse_mode.get_flag():
        tensor_to_block.tensor_to_block(ir)
        # copyPropagation.copy_proagate(ir)
        if sroa_build():
            tensor_to_block.splice_flow(ir, list(ir.shape.keys()))
            stats = sroa_pass.sroa(ir)
            print('[sroa] {aggregates} aggregates removed, {clones_dropped} clones and '
                  '{lambdas_dropped} identity lambdas and {casts_dropped} casts dropped, {dead_dropped} dead stores removed, {statements} tensor '
                  'statements, {params} flow params '
                  '(functional={functional}, {view_writes} view writes, {block_writes} block writes)'.format(**stats))
            if stats['survivors']:
                print('[sroa] {} values not scalarized:'.format(len(stats['survivors'])))
                for reason in sorted(set(stats['survivors']))[:10]:
                    print('[sroa]   ' + reason)
            subexp_inlining.inline_subexp_block(ir.flow_block)
            subexp_inlining.recycle_temp_names_block(ir.flow_block.children)
        else:
            subexp_inlining.inline_subexp(ir)
            subexp_inlining.recycle_temp_names(ir)
        # constant_folding.constant_fold(ir)
        # copyPropagation.copy_proagate(ir)

    cg = codeGen.CodeGen(output_path)
    cg.visit(ir)
    cg.finish()
    return True