from constraintflow.compiler.ir import *


def sparse_tensors_to_blocks_cfg(cfg):
    assert False, 'TODO'


def sparse_tensors_to_blocks(ir):
    for transformer in ir.tstore.keys():
        for i in range(len(ir.tstore[transformer])):
            cfg = ir.tstore[transformer][i].cfg


    assert False, 'TODO'
    return ir
