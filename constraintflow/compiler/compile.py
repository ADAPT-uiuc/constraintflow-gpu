import antlr4 as antlr

from constraintflow.ast_cflow import dslLexer
from constraintflow.ast_cflow import dslParser
from constraintflow.ast_cflow import astBuilder
from constraintflow.ast_cflow import astTC
from constraintflow.compiler import convertToIr as c2r
from constraintflow.compiler import representations
from constraintflow.compiler import codeGen
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


def _reset_compiler_state():
    # These module-level counters accumulate across compilations within a single
    # process. A fresh process resets them implicitly, so the manual two-step
    # (--simulacrum then --reuse, in separate processes) always numbers whiles and
    # ttb-counters from scratch. When two compiles share a process (e.g. the
    # simulacrum-compile probe+reuse pair), reset them here so the reuse build
    # reproduces the same while_number / ttb_counter numbering the simulacrum run
    # captured -- otherwise the reuse build reads mismatched jit_* capture keys.
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
    astTC.ASTTC().visit(ast)
    
    ir = c2r.ConvertToIr().visit(ast)
    representations.ssa(ir)

    optimizations = optimizations_rewrite

    for opt in optimizations:
        opt(ir)
    representations.remove_phi(ir)
    # codeGen.CodeGen(output_path).visit(ir)

    replay = reuse_mode.get_flag()

    if reuse_mode.get_flag():
        tensor_to_block.tensor_to_block(ir)
        # copyPropagation.copy_proagate(ir)
        # subexp_inlining.inline_subexp(ir)
        # subexp_inlining.recycle_temp_names(ir)
        # constant_folding.constant_fold(ir)
        # copyPropagation.copy_proagate(ir)

    codeGen.CodeGen(output_path).visit(ir)

    return True