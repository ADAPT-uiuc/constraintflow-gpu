import json
import operator
import os

from . import irVisitor 
import copy
from .ir import * 
from constraintflow.lib.globals import reuse_mode, load_capture, capture_exists

class CodeGen(irVisitor.IRVisitor):
    def __init__(self,folder):
        self.folder = folder 
        if self.folder.endswith('/'):
            self.folder = self.folder[:-1]
        self.main_file = self.folder + '/main.py'
        self.transformers_file = self.folder + '/transformers.py'
        self.shape = None
        open(self.main_file, "w").close()
        open(self.transformers_file, "w").close()

        self.file = open(self.main_file, "a")
        self.indent = 0

        self.current_layer_index = None

        self.write("import sys")
        self.write("import os")
        self.write("from constraintflow.lib.spec import *")
        self.write("from constraintflow.lib.flow_sparse import Flow")
        self.write("from constraintflow.lib.abs_elem import Abs_elem_sparse")
        self.write("from constraintflow.lib.symexp import *")
        self.write("from constraintflow.lib.globals import save_capture")
        self.write("from transformers import *")
        self.write("\n")
        # self.write("torch.cuda.reset_peak_memory_stats()")
        self.write("def run(network_file, batch_size, eps, dataset_X, dataset_y, dataset, train, print_intermediate_results, no_sparsity):")
        
        self.indent += 1
        self.visited = set()

        self.counter = 0

    # This function is used to get the branch information for a tuple (block_id, layer_index) from the jit file. 
    def get_profiled_branch(self, block_id):
        if not reuse_mode.get_flag():
            return None
        if self.current_layer_index is None:
            return None
        rel_path = f"jit_branch/branch_{self.current_layer_index}_{block_id}.json"
        if not capture_exists(rel_path):
            raise Exception(f"Profiled branch data not found for block_id {block_id} at {rel_path}")
        json_obj = load_capture(rel_path)
        taken = json_obj["taken"]
        if taken in ["then", "else"]:
            return taken
        return None


    def write(self, str, flag=True):
        self.file.write('\t'*self.indent + str)
        if flag:
            self.file.write('\n')

    def write_expr(self, str, flag=True):
        self.file.write(str)
        if flag:
            self.file.write('\n')


    def open(self, file):
        self.file.close()
        self.file = open(file, "a")

    def visitIrProgram(self, node):
        self.shape = node.shape
        temp_shape = copy.deepcopy(self.shape)
        temp_shape['llist'] = 'bool'
        temp_dict = "{"
        key = 'llist'
        i = 0

        temp_dict += "\'" + key + "\' : " + key + ', '
        for i, key in enumerate(node.shape.keys()):
            temp_dict += "\'" + key + "\' : " + key 
            if i < len(node.shape.keys())-1:
                temp_dict += ", "
        temp_dict += '}'


        self.write("network, l, u, L, U, Z, llist = get_network_and_input_spec(network_file, batch_size, dataset_X, dataset_y, dataset, eps=eps, train=train, no_sparsity=no_sparsity)")
        self.write("abs_elem = Abs_elem_sparse(" + temp_dict + ", " + str(temp_shape) + ", network, batch_size=batch_size, no_sparsity=no_sparsity)")
        



        # GENERATE TRANSFORMERS
        self.open(self.transformers_file)
        self.indent = 0
        self.write('import gc')
        self.write('import json')
        self.write('import os')
        self.write('import torch')
        self.write('import operator')
        self.write('from constraintflow.lib.globals import device_mode')
        self.write('from constraintflow.lib.polyexp import PolyExpSparse')
        if reuse_mode.get_flag():
            self.write('from constraintflow.lib.symexp import SymExpSparse')
            # self.write('from constraintflow.lib.symexp import get_new_eps')
            # self.write('from constraintflow.gbcsr.op_helper import binary_to_identity_unary')

        else:
            self.write('from constraintflow.lib.symexp import *')
        if reuse_mode.get_flag():
            self.write('import operator')
            self.write('import torch.nn.functional as F')
            self.write('from constraintflow.gbcsr.sparse_tensor import SparseTensor')
            # self.write('from constraintflow.gbcsr.tensor_ops import *')
            self.write('from constraintflow.gbcsr.sparse_block import SparseBlock, DenseBlock, DiagonalBlock, PatchesBlock, KernelBlock, RepeatBlock, ConstBlock')
        else:
            self.write('from constraintflow.gbcsr.sparse_tensor import SparseTensor')
        self.write('from constraintflow.lib.llist import Llist')
        if not reuse_mode.get_flag():
            self.write('from constraintflow.gbcsr.tensor_ops import *')
            

        for i, transformer_name in enumerate(node.tstore.keys()):
            self.write('class ' + transformer_name + ':')
            self.indent += 1

            transformerIr = node.tstore[transformer_name]

            for j, opStmtIr in enumerate(transformerIr):
                if opStmtIr.layerwise_cfgs is None:
                    self.write('def ' + opStmtIr.op + '(self, abs_elem, prev, curr, poly_size, curr_size, prev_size, input_size, batch_size, layer_index = None):')
                    self.indent += 1
                    # self.write('torch.cuda.memory._record_memory_history(')
                    # self.indent += 1
                    # self.write('max_entries=1000000')
                    # self.indent -= 1
                    # self.write(')')
    
                    cfg = opStmtIr.cfg
                    self.visit(cfg.ir[cfg.entry_node])
                    self.indent -= 1
                    self.write('', True)
                else:
                    self.write('def ' + opStmtIr.op + '(self, abs_elem, prev, curr, poly_size, curr_size, prev_size, input_size, batch_size, layer_index = None):')
                    self.indent += 1
                    # self.write('torch.cuda.memory._record_memory_history(')
                    # self.indent += 1
                    # self.write('max_entries=1000000')
                    # self.indent -= 1
                    # self.write(')')
                    for layer_index in opStmtIr.layerwise_cfgs.keys():
                        self.write('if layer_index == ' + str(layer_index) + ':')
                        self.indent += 1
                        self.write('return self.' + opStmtIr.op + '_' + str(layer_index) + '(abs_elem, prev, curr, poly_size, curr_size, prev_size, input_size, batch_size, layer_index = layer_index)')
                        self.indent -= 1
                    self.indent -= 1
                    self.write('')
                    
                    for layer_index in opStmtIr.layerwise_cfgs.keys():
                        self.write('def ' + opStmtIr.op + '_' + str(layer_index) + '(self, abs_elem, prev, curr, poly_size, curr_size, prev_size, input_size, batch_size, layer_index = None):')
                        self.indent += 1
                        self.write('while_iteration = -1')
                        cfg = opStmtIr.layerwise_cfgs[layer_index]
                        self.current_layer_index = layer_index
                        self.visit(cfg.ir[cfg.entry_node])
                        self.indent -= 1
                        self.write('', True)
            self.indent -=1

        self.open(self.main_file)
        for i in range(len(node.irNodes)):
            self.visit(node.irNodes[i])

    def visitIrBlock(self, node):
        if not node in self.visited:
            self.visited.add(node)
            ir_list = node.children
            for counter, i in enumerate(ir_list):
                self.visit(i)
            if node.inner_jump != None:
                if len(node.inner_jump)==3:
                    cond = self.visit(node.inner_jump[0])
                    block_id = node.block_id
                    profiled_branch = self.get_profiled_branch(block_id)
                    if profiled_branch == 'then':
                        self.visit(node.inner_jump[1])
                    elif profiled_branch == 'else':
                        self.visit(node.inner_jump[2])
                    else:
                        self.write('if(' + str(cond) + '):')
                        self.indent += 1
                        if block_id is not None:
                            self.write('if dummy_mode:')
                            self.indent += 1
                            self.write('save_capture("jit_branch/branch_" + str(layer_index) + "_' + str(block_id) + '.json", {"taken": "then"})')
                            self.indent -= 1
                        self.visit(node.inner_jump[1])
                        self.indent -= 1
                        self.write('else:')
                        self.indent += 1
                        if block_id is not None:
                            self.write('if dummy_mode:')
                            self.indent += 1
                            self.write('save_capture("jit_branch/branch_" + str(layer_index) + "_' + str(block_id) + '.json", {"taken": "else"})')
                            self.indent -= 1
                        self.visit(node.inner_jump[2])
                        self.indent -= 1
                elif not isinstance(node.inner_jump[1], IrWhileBlock):
                    cond = self.visit(node.inner_jump[0])
                    self.write('if(' + str(cond) + '):')
                    self.indent += 1
                    self.visit(node.inner_jump[1])
                    self.indent -= 1
                else:
                    cond = self.visit(node.inner_jump[0])
                    self.write('while_iteration = -1')
                    self.write('while(' + str(cond) + '):')
                    self.indent += 1
                    self.write('while_iteration += 1')
                    # self.write('print(\'while_iteration\', while_iteration)')
                    self.visit(node.inner_jump[1])
                    self.indent -= 1
                    self.write('if dummy_mode:')
                    self.indent += 1
                    self.write('json_obj = {"num_iterations": while_iteration}')
                    self.write('save_capture("jit_while/while_iterations_layer_" + str(layer_index) + "_while_" + str(' + str(node.inner_jump[1].while_number) + ') + ".json", json_obj)')
                    self.indent -= 1
            if node.jump != None:
                self.visit(node.jump[1])

    def visitIrAssignment(self, node):
        var = str(self.visit(node.children[0]))
        expr = str(self.visit(node.children[1]))
        self.write(var + ' = ' + expr )
        # node.counter = self.counter
        # self.counter += 1

    # For the del statements
    # Currently not used. 
    def visitIrDel(self, node):
        # self.write('del ' + ', '.join(node.var_names))
        # self.write('gc.collect()')
        # self.write('torch.cuda.empty_cache()')
        pass

    def visitIrBreak(self, node):
        self.write('break')

    def visitIrTransRetBasic(self, node):
        # self.write('try:')
        # self.indent += 1
        # self.write('torch.cuda.memory._dump_snapshot(f"memory_usage_{layer_index}.pickle")')
        # self.indent -= 1
        # self.write('except:')
        # self.indent += 1
        # self.write('raise Exception("CUDA memory snapshot failed. This can happen if the file prefix is too long or if there are issues with the CUDA setup. Please check your CUDA configuration and ensure that the file prefix is valid.")')
        # self.indent -= 1
        
        exprs = []
        for i in range(len(node.children)):
            expr = self.visit(node.children[i])
            exprs.append(expr)
        ret_expr = 'return '
        for i in range(len(node.children)-1):
            ret_expr += exprs[i]
            ret_expr += ', '
        ret_expr += exprs[-1] + ', '
        self.write(ret_expr)

    def visitIrTransRetIf(self, node):
        cond = self.visit(node.children[0])
        self.write('if(' + cond + '):')
        self.indent += 1
        if len(node.children[1])>0:
            for i in node.children[1]:
                self.visit(i)
        else:
            self.write('pass')
        self.indent -= 1
        if len(node.children[2])>0:
            self.write('else:')
            self.indent += 1
            for i in node.children[1]:
                self.visit(i)
            self.indent -= 1
        else:
            self.write('pass')
        self.write('return')

    def visitIrIte(self, node):
        cond = self.visit(node.children[0])
        self.write('if(' + cond + '):')
        self.indent += 1
        if len(node.children[1])>0:
            for i in node.children[1]:
                self.visit(i)
        else:
            self.write('pass')
        self.indent -= 1
        if len(node.children[2])>0:
            self.write('else:')
            self.indent += 1
            for i in node.children[1]:
                self.visit(i)
            self.indent -= 1

    # def visitIrWhile(self, node):
    #     cond = self.visit(node.children[0])
    #     self.write('while(' + str(cond) + '):')
    #     self.indent += 1
    #     for ir in node.children[1:]:
    #         self.visit(ir)
    #     self.indent -= 1
    

    def visitIrStr(self, node):
        return node
    
    def visitIrConst(self, node):
        return str(node.const)
    
    def visitInt(self, node):
        return str(node)
    
    def visitList(self, node):
        res = '['
        for i, child in enumerate(node):
            if i > 0:
                res += ', '
            res += self.visit(child)
        res += ']'
        return res

    def renderTraceIndex(self, index):
        parts = []
        for item in index:
            if isinstance(item, list):
                if len(item) == 2:
                    parts.append(self.visit(item[0]) + ':' + self.visit(item[1]))
                else:
                    parts.append(self.visit(item))
            elif isinstance(item, str):
                parts.append(item)
            elif isinstance(item, int) and item == 0:
                parts.append(':')
            else:
                parts.append(self.visit(item))
        return ', '.join(parts)

    def visitIrSparseTensor(self, node):
        args = [
            self.visit(node.start_indices),
            self.visit(node.children[0]),
            self.visit(node.dims),
            self.visit(node.total_size),
        ]

        kwargs = []
        if hasattr(node, "end_indices"):
            kwargs.append(f"end_indices={self.visit(node.end_indices)}")
        if hasattr(node, "type"):
            kwargs.append(f"type={self.visit(node.type)}")
        if hasattr(node, "dense_const"):
            kwargs.append(f"dense_const={self.visit(node.dense_const)}")

        return "SparseTensor(" + ", ".join(args + kwargs) + ")"

    def visitIrConstBlock(self, node):
        return 'ConstBlock(' + self.visit(node.children[0]) + ',' + self.visit(node.total_shape) + ')'

    def visitIrDenseBlock(self, node):
        return 'DenseBlock(' + self.visit(node.children[0]) + ')'

    def visitIrPatchesBlock(self, node):
        return (
            'PatchesBlock(' + self.visit(node.children[0]) + ', '
            + self.visit(node.total_shape) + ', '
            + str(node.ix) + ', ' + str(node.iy) + ', '
            + str(node.ox) + ', ' + str(node.oy) + ', '
            + str(node.sx) + ', ' + str(node.sy) + ', '
            + str(node.px) + ', ' + str(node.py) + ', '
            + str(node.kx) + ', ' + str(node.ky) + ', '
            + str(node.num_channels) + ', ' + str(node.num_kernels) + ')'
        )

    def visitIrKernelBlock(self, node):
        return (
            'KernelBlock(' + self.visit(node.children[0]) + ', '
            + self.visit(node.total_shape) + ', '
            + str(node.ix) + ', ' + str(node.iy) + ', '
            + str(node.ox) + ', ' + str(node.oy) + ', '
            + str(node.sx) + ', ' + str(node.sy) + ', '
            + str(node.px) + ', ' + str(node.py) + ')'
        )

    def visitIrRepeatBlock(self, node):
        return 'RepeatBlock(' + self.visit(node.children[0]) + ', ' + self.visit(node.total_shape) + ')'

    def visitIrDiagonalBlock(self, node):
        return (
            'DiagonalBlock(' + self.visit(node.children[0]) + ', '
            + self.visit(node.total_shape) + ', '
            + str(node.diag_index) + ')'
        )

    def visitIrTorchDiagonal(self, node):
        input_expr = self.visit(node.children[0])
        return (
            'torch.diagonal(' + input_expr
            + ', dim1=' + str(node.dim1)
            + ', dim2=' + str(node.dim2) + ')'
        )

    def visitIrTorchPermute(self, node):
        input_expr = self.visit(node.children[0])
        perm_args = ', '.join(str(i) for i in node.permutation)
        return input_expr + '.permute(' + perm_args + ')'

    def visitIrTorchTranspose(self, node):
        return self.visit(node.children[0]) + '.transpose(' + str(node.dim0) + ', ' + str(node.dim1) + ')'

    def visitIrTorchMatmul(self, node):
        return 'torch.matmul(' + self.visit(node.children[0]) + ', ' + self.visit(node.children[1]) + ')'

    def visitIrTorchUnsqueeze(self, node):
        return self.visit(node.children[0]) + '.unsqueeze(' + str(node.index) + ')'

    def visitIrTorchSqueeze(self, node):
        return self.visit(node.children[0]) + '.squeeze(' + str(node.index) + ')'

    def visitIrTorchReshape(self, node):
        return self.visit(node.children[0]) + '.reshape(' + self.visit(node.shape) + ')'

    def visitIrTorchView(self, node):
        return self.visit(node.children[0]) + '.view(' + self.visit(node.shape) + ')'

    def visitIrTorchRepeat(self, node):
        return self.visit(node.children[0]) + '.repeat(' + self.visit(node.repeats) + ')'

    def visitIrTorchExpand(self, node):
        return self.visit(node.children[0]) + '.expand(' + self.visit(node.shape) + ')'

    def visitIrTorchSum(self, node):
        return self.visit(node.children[0]) + '.sum(dim=' + str(node.dim) + ')'

    def visitIrTorchZeros(self, node):
        kwargs = []
        if node.device is not None:
            kwargs.append('device=' + self.visit(node.device))
        if node.dtype is not None:
            kwargs.append('dtype=' + self.visit(node.dtype))
        args = [self.visit(node.size)] + kwargs
        return 'torch.zeros(' + ', '.join(args) + ')'

    def visitIrTorchEye(self, node):
        kwargs = []
        if node.device is not None:
            kwargs.append('device=' + self.visit(node.device))
        if node.dtype is not None:
            kwargs.append('dtype=' + self.visit(node.dtype))
        args = [self.visit(node.size)] + kwargs
        return 'torch.eye(' + ', '.join(args) + ')'

    def visitIrTorchFloat(self, node):
        return self.visit(node.children[0]) + '.float()'

    def visitIrTorchDiagEmbed(self, node):
        return 'torch.diag_embed(' + self.visit(node.children[0]) + ')'

    def visitIrTorchStride(self, node):
        return self.visit(node.children[0]) + '.stride()'

    def visitIrTorchAsStrided(self, node):
        return (
            'torch.as_strided(' + self.visit(node.children[0]) + ', '
            + self.visit(node.size) + ', ' + self.visit(node.stride) + ')'
        )

    def visitIrTorchSlice(self, node):
        return self.visit(node.children[0]) + '[' + self.renderTraceIndex(node.index) + ']'

    def visitIrFConv2d(self, node):
        return (
            'F.conv2d(' + self.visit(node.children[0]) + ', '
            + self.visit(node.children[1]) + ', stride='
            + self.visit(node.stride) + ', padding=' + self.visit(node.padding) + ')'
        )

    def visitIrFConvTranspose2d(self, node):
        kwargs = []
        if node.stride is not None:
            kwargs.append('stride=' + self.visit(node.stride))
        if node.padding is not None:
            kwargs.append('padding=' + self.visit(node.padding))
        if node.output_padding is not None:
            kwargs.append('output_padding=' + self.visit(node.output_padding))
        return (
            'F.conv_transpose2d(' + self.visit(node.children[0]) + ', '
            + self.visit(node.children[1])
            + (', ' + ', '.join(kwargs) if kwargs else '') + ')'
        )

    def visitIrFUnfold(self, node):
        return (
            'F.unfold(' + self.visit(node.children[0]) + ', kernel_size='
            + self.visit(node.kernel_size) + ', padding=' + self.visit(node.padding)
            + ', stride=' + self.visit(node.stride) + ')'
        )

    def visitIrAssignToView(self, node):
        self.write(
            self.visit(node.children[0]) + '[' + self.renderTraceIndex(node.index) + '] = '
            + self.visit(node.children[1])
        )
    
    def visitIrEmptyList(self, node):
        return '[]'

    def visitType(self, node):
        return node.__name__
    
    def visitFloat(self, node):
        if node == float('inf'):
            return "float('inf')"
        if node == float('-inf'):
            return "float('-inf')"
        return str(node)
    
    def visitInt(self, node):
        return str(node)
    
    def visitIrAppendList(self, node):
        return self.visit(node.children[0]) + '+ [' + self.visit(node.children[1]) + ']'
    

    def visitIrListExtract(self, node):
        return '(' + self.visit(node.children[0]) + ')[' + self.visit(node.children[1]) + ']'
    
    def visitIrBlockExtract(self, node):
        return self.visit(node.children[0]) + '.blocks[' + self.visit(node.children[1]) + ']'

    def visitIrBlockCopy(self, node):
        return self.visit(node.children[0]) + '.copy()'
    
    def visitIrGetSparseTensorBlocks(self, node):
        the_sparse_tensor = self.visit(node.children[0])
        return the_sparse_tensor + '.blocks'

    def visitIrGetAbsElemSparseDKey(self, node):
        # the_abs_elem_sparse = self.visit(node.children[0])
        key = node.key
        # return the_abs_elem_sparse + f'.d[{key}]'
        return f'abs_elem.d[\'{key}\']'
    
    def visitIrGetPolyExpSparseConst(self, node):
        the_pes = self.visit(node.children[0])
        return the_pes + '.const'

    def visitIrGetPolyExpSparseMat(self, node):
        the_pes = self.visit(node.children[0])
        return the_pes + '.mat'

    def visitIrGetSymExpSparseConst(self, node):
        the_ses = self.visit(node.children[0])
        return the_ses + '.const'

    def visitIrGetSymExpSparseMat(self, node):
        the_ses = self.visit(node.children[0])
        return the_ses + '.mat'

    def visitIrGetKthLayerNetworkParam(self, node):
        return f'abs_elem.network[{node.layer_index}].{node.param}'

    def visitIrDenseBlock(self, node):
        return 'DenseBlock(' + self.visit(node.children[0]) + ')'
    
    def visitIrKernelBlock(self, node):
        return f'KernelBlock({self.visit(node.children[0])}, torch.tensor({node.total_shape}), ' \
               f'{node.ix}, {node.iy}, {node.ox}, {node.oy}, {node.sx}, {node.sy}, {node.px}, {node.py})'

    def get_operator_func(self, name: str):
        if not isinstance(name, str):
            name = name.__name__
        OP_MAP = {
            "add": 'operator.add',
            "sub": 'operator.sub',
            "eq": 'operator.eq',
            "ne": 'operator.ne',
            "ge": 'operator.ge',
            "gt": 'operator.gt',
            "le": 'operator.le',
            "lt": 'operator.lt',
            "or_": 'operator.or_',
            "and_": 'operator.and_',
            "mul": 'operator.mul',
            "truediv": 'operator.truediv',
            "floordiv": 'operator.floordiv',
            "mod": 'operator.mod',
            "pow": 'operator.pow',
        }

        try:
            return OP_MAP[name]
        except KeyError:
            raise ValueError(f"Unsupported operator: {name}, {type(name)}")


    def visitIrBlockBinaryOp(self, node):
        return self.visit(node.children[0]) + '.binary(' + self.visit(node.children[1]) + ', ' + self.get_operator_func(node.op) + ')'

    def visitIrBlockWhereBlock(self, node):
        return 'sp_where_block(' + self.visit(node.children[0]) + ', ' + self.visit(node.children[1]) + ', ' + self.visit(node.children[2]) + ')'

    def visitIrBlockUnaryOp(self, node):
        op = node.op
        if op == '-':
            op_str = 'operator.neg'
        elif op == 'not':
            op_str = 'operator.not_'
        elif op == 'sigma':
            op_str = "'sigma'"
        else:
            raise Exception('OP NOT IDENTIFIED', op)
        return self.visit(node.children[0]) + '.unary(' + op_str + ')'
    
    def visitIrGetSubBlockCustomRange(self, node):
        return self.visit(node.children[0]) + '.get_sub_block_custom_range(' + self.visit(node.start_index) + ', ' + self.visit(node.end_index) + ', ' + self.visit(node.block_id) + ', ' + str(node.tensor) + ')'

    def visitTorchTensor(self, node):
        return 'torch.tensor(' + str(node.tolist()) + ')'

    def visitIrVar(self, node):
        if node.name == 'sym_size':
            return 'SymExpSparse.count'
        return node.name
    
    def visitIrEpsilon(self, node):
        # num = self.visit(node.num)
        shape = '['
        for i in range(len(node.irMetadata)):
            for j in range(len(node.irMetadata[i].shape)):
                shape += self.visit(node.irMetadata[i].shape[j]) + ","
        shape += ']'
        if node.inside_while:
            return 'get_new_eps(abs_elem.network, torch.tensor(' + shape + '), layer_index = layer_index, counter = ' + str(node.ttb_counter) + ', inside_while = True, while_number = ' + str(node.while_number) + ', while_iteration=while_iteration) #' + str(node.ttb_counter)
        return 'get_new_eps(abs_elem.network, torch.tensor(' + shape + '), layer_index = layer_index, counter = ' + str(node.ttb_counter) + ', inside_while = False, while_number = ' + str(node.while_number) + ') #' + str(node.ttb_counter)

    def visitIrNewEps(self, node):
        [matIr, constIr] = node.children
        return 'SymExpSparse(abs_elem.network, ' + self.visit(matIr) + ', ' + self.visit(constIr) + ')'
    
    def visitIrPhi(self, node):
        s = 'phi(['
        for i in range(len(node.children)):
            s += self.visit(node.children[i])
            if i != len(node.children)-1:
                s += ', '
        s += '])'
        return s
    
    def visitIrConvertBoolToFloat(self, node):
        if reuse_mode.get_flag():
            return '('+self.visit(node.children[0])+').float()'
        return 'convert_to_float(' + self.visit(node.children[0]) + ')'

    def visitIrRepeat(self, node):
        repeat_dims = ''
        for i in range(1, len(node.children)):
            repeat_dims += self.visit(node.children[i])
            if i<len(node.children)-1:
                repeat_dims += ', '
        repeat_dims = 'torch.tensor([' + repeat_dims + '])'
        if node.inside_while:
            ret = 'repeat(' + self.visit(node.children[0]) + ', ' + repeat_dims + ', ' + 'layer_index = layer_index, ' + 'counter = ' + str(node.ttb_counter) + ', inside_while = True' + ', while_number = ' + str(node.while_number) + ', while_iteration=while_iteration) #' + str(node.ttb_counter)
        else:
            ret = 'repeat(' + self.visit(node.children[0]) + ', ' + repeat_dims + ', ' + 'layer_index = layer_index, ' + 'counter = ' + str(node.ttb_counter) + ', inside_while = False' + ', while_number = ' + str(node.while_number) + ') #' + str(node.ttb_counter)
        return ret

    def visitIrAddDimension(self, node):
        [inputIr] = node.children
        size = 0
        for i in range(len(node.irMetadata)-1):
            for j in range(len(node.irMetadata[i].broadcast)):
                size += 1
        size += len(inputIr.irMetadata[-1].shape)
        indices = []
        for i in range(len(node.irMetadata[-1].shape) - len(inputIr.irMetadata[-1].shape)):
            indices.append(str(size))
            size += 1
        if len(indices) == 0:
            return self.visit(inputIr)
        if node.inside_while:
            kw_suffix = (
                'layer_index = layer_index, '
                'counter = ' + str(node.ttb_counter) + ', inside_while = True, '
                'while_number = ' + str(node.while_number) + ', while_iteration=while_iteration)'
            )
        else:
            kw_suffix = (
                'layer_index = layer_index, '
                'counter = ' + str(node.ttb_counter) + ', inside_while = False, '
                'while_number = ' + str(node.while_number) + ')'
            )
        ret = '(' + self.visit(inputIr) + ')'
        for dim in indices:
            ret = ret + '.unsqueeze(' + dim + ', ' + kw_suffix
        return ret
    
    def visitIrRemoveDimension(self, node):
        [inputIr] = node.children
        if node.inside_while:
            ret = '(' + self.visit(inputIr) + ').squeeze(' + str(node.numDim) + ', ' + 'layer_index = layer_index, ' + 'counter = ' + str(node.ttb_counter) + ', inside_while = True' + ', while_number = ' + str(node.while_number) + ', while_iteration=while_iteration)'
        else:
            ret = '(' + self.visit(inputIr) + ').squeeze(' + str(node.numDim) + ', ' + 'layer_index = layer_index, ' + 'counter = ' + str(node.ttb_counter) + ', inside_while = False' + ', while_number = ' + str(node.while_number) + ')'
        return ret

    def visitIrAddDimensionConst(self, node):
        assert(isinstance(node, IrAddDimensionConst))
        inputIr = node.children[0]
        size = len(node.children)-1
        repeat_dims = ''
        for i in range(1, len(node.children)):
            repeat_dims += self.visit(node.children[i])
            if i<len(node.children)-1:
                repeat_dims += ', '
        if inputIr.irMetadata[-1].isConst:
            ret = 'SparseTensor([], [], 0, torch.tensor([]), dense_const=' + str(self.visit(inputIr)) + ', type= type(' + str(self.visit(inputIr)) + '))'
        else:
            ret = str(self.visit(inputIr))
        for i in range(size):
            ret += '.unsqueeze(' + str(i) + ')'
        ret += '.repeat(torch.tensor([' + repeat_dims + ']))'
        return ret
    
    def visitIrBinaryOp(self, node):
        op_name = None 
        flag = False
        if node.op == 'max':
            op_name = 'cf_max'
            flag = True
        elif node.op == 'min':
            op_name = 'cf_min'
            flag = True
        elif node.op == '+':
            op_name = 'operator.add'
        elif node.op == '-':
            op_name = 'operator.sub'
        elif node.op == '<=':
            op_name = 'operator.le'
        elif node.op == '<':
            op_name = 'operator.lt'
        elif node.op == '>=':
            op_name = 'operator.ge'
        elif node.op == '>':
            op_name = 'operator.gt'
        elif node.op == '==':
            op_name = 'operator.eq'
        elif node.op == '!=':
            op_name = 'operator.ne'
        elif node.op == 'and':
            op_name = 'operator.and_'
        elif node.op == 'or':
            op_name = 'operator.or_'
        else:
            raise Exception('OP NOT IDENTIFIED', node.op)
        
        [lhsIr, rhsIr] = node.children
        if flag:
            if node.inside_while:
                return op_name + '(' + self.visit(lhsIr) + ', ' + self.visit(rhsIr) + ', layer_index=layer_index, counter=' + str(node.ttb_counter) + ', inside_while=True, while_number=' + str(node.while_number) + ', while_iteration=while_iteration)'
            return op_name + '(' + self.visit(lhsIr) + ', ' + self.visit(rhsIr) + ', layer_index=layer_index, counter=' + str(node.ttb_counter) + ', inside_while=False, while_number=' + str(node.while_number) + ')'
        else:
            if node.inside_while:
                return 'binary(' + self.visit(lhsIr) + ', ' + self.visit(rhsIr) + ', ' + op_name + ', ' + 'layer_index = layer_index, ' + 'counter = ' + str(node.ttb_counter) + ', inside_while = True' + ', while_number = ' + str(node.while_number) + ', while_iteration=while_iteration) #' + str(node.ttb_counter)
            

            return 'binary(' + self.visit(lhsIr) + ', ' + self.visit(rhsIr) + ', ' + op_name + ', ' + 'layer_index = layer_index, ' + 'counter = ' + str(node.ttb_counter) + ', inside_while = False' + ', while_number = ' + str(node.while_number) + ') #' + str(node.ttb_counter)
    
    def visitIrUnaryOp(self, node):
        op_name = None 
        flag = False
        if node.op == '-':
            op_name = 'operator.neg'
        elif node.op == 'not':
            op_name = 'operator.not_'
        elif node.op == 'sigma':
            op_name = f"'sigma'"
        elif node.op == 'any':
            op_name = 'any'
            flag = True
        elif node.op == 'all':
            op_name = 'all'
            flag = True
        elif node.op == 'get_dims':
            op_name = 'get_dims'
            flag = True
        elif node.op == 'get_shape_1':
            op_name = 'get_shape_1'
            flag = True
        elif node.op == 'get_shape_0':
            op_name = 'get_shape_0'
            flag = True
        else:
            raise Exception('OP NOT IDENTIFIED', node.op)
        
        [inputIr] = node.children
        if flag and node.op in ('get_shape_1', 'get_shape_0'):
            return op_name + '(' + self.visit(inputIr) + ')'
        elif flag:
            if node.inside_while:
                return f'{op_name}({self.visit(inputIr)}, layer_index=layer_index, counter={node.ttb_counter}, inside_while=True, while_number={node.while_number}, while_iteration=while_iteration)'
            return f'{op_name}({self.visit(inputIr)}, layer_index=layer_index, counter={node.ttb_counter}, inside_while=False, while_number={node.while_number})'
        else:
            if node.inside_while:
                return 'unary(' + self.visit(inputIr) + ', ' + op_name + ', layer_index=layer_index, counter=' + str(node.ttb_counter) + ', inside_while=True, while_number=' + str(node.while_number) + ', while_iteration=while_iteration)'
            return 'unary(' + self.visit(inputIr) + ', ' + op_name + ', layer_index=layer_index, counter=' + str(node.ttb_counter) + ', inside_while=False, while_number=' + str(node.while_number) + ')'
        
    def visitIrSimpleUnary(self, node):
        op = node.op
        if isinstance(op, (IrVar, IrLambda)):
            return '(' + self.visit(op) + ')(' + self.visit(node.children[0]) + ')'
        if op == '-':
            op_str = 'operator.neg'
        elif op == 'not':
            op_str = 'operator.not_'
        elif op == 'sigma':
            return 'torch.sigmoid(' + self.visit(node.children[0]) + ')'
        else:
            raise Exception('OP NOT IDENTIFIED', op)
        return op_str + '(' + self.visit(node.children[0]) + ')'

    def visitIrLambda(self, node):
        if node.op in ('add', 'mul', 'and_', 'or_'):
            return 'lambda x: x'
        if node.op == 'sub':
            return 'lambda x: x.unary(operator.neg)'
        raise Exception('OP NOT IDENTIFIED', node.op)

    def visitIrSimpleBinary(self, node):
        [lhsIr, rhsIr] = node.children
        lhs = self.visit(lhsIr)
        rhs = self.visit(rhsIr)
        return self.get_operator_func(node.op) + '(' + lhs + ', ' + rhs + ')'

    def visitIrTorchWhere(self, node):
        return (
            'torch.where(' +
            self.visit(node.children[0]) + ', ' +
            self.visit(node.children[1]) + ', ' +
            self.visit(node.children[2]) +
            ')'
        )
    
    def visitIrTensorOnes(self, node):
        if isinstance(node.total_size, str):
            return 'torch.ones(*' + node.total_size + ')'
        return 'torch.ones(*' + str(node.total_size.tolist()) + ')'

    def visitTensorRepeat(self, node):
        return 'torch.repeat(' + self.visit(node.children[0]) + ', *' + node.repeat_dims + ')'

    def visitIrTensorRepeat(self, node):
        return self.visitTensorRepeat(node)
    
    def visitTensorClamp(self, node):
        if node.min_true:
            return 'torch.clamp(' + self.visit(node.children[0]) + ', min=' + str(node.const) + ')'
        else:
            return 'torch.clamp(' + self.visit(node.children[0]) + ', max=' + str(node.const) + ')'

    def visitIrTensorClamp(self, node):
        return self.visitTensorClamp(node)

    def visitIrBlockClamp(self, node):
        return self.visit(node.children[0]) + '.clamp(' + str(node.const) + ', min_true=' + str(node.min_true) + ')'

    def visitIrBlockSqueeze(self, node):
        return self.visit(node.children[0]) + '.squeeze(' + str(node.index) + ')'

    def visitIrBlockUnsqueeze(self, node):
        return self.visit(node.children[0]) + '.unsqueeze(' + str(node.index) + ')'

    

    def visitIrBlockRepeat(self, node):
        # repeat_dims = ''
        # for i in range(1, len(node.children)):
        #     repeat_dims += self.visit(node.children[i])
        #     if i<len(node.children)-1:
        #         repeat_dims += ', '
        # repeat_dims = 'torch.tensor([' + repeat_dims + '])'
        temp = (node.repeat_dims)
        return self.visit(node.children[0]) + '.repeat(torch.tensor(' + str(node.repeat_dims.tolist()) + ', dtype=torch.int64))'
        # ret = self.visit(node.children[0]) + '.repeat(torch.tensor(' + str(node.repeat_dims.tolist()) + ', dtype=torch.int64))'
        # print(ret)
        # return ret

    def visitIrGetDefaultStop(self, node):
        repeat_dims = ''
        for i in range(1, len(node.children)):
            repeat_dims += self.visit(node.children[i])
            if i<len(node.children)-1:
                repeat_dims += ', '
        if node.inside_while:
            return f'get_default_stop([{repeat_dims}], abs_elem, batch_size, curr_size, poly_size, layer_index=layer_index, counter={node.ttb_counter}, inside_while=True, while_number={node.while_number}, while_iteration=while_iteration)'
        return f'get_default_stop([{repeat_dims}], abs_elem, batch_size, curr_size, poly_size, layer_index=layer_index, counter={node.ttb_counter}, inside_while=False, while_number={node.while_number})'
    
    def visitIrGetPriorityLList(self, node):
        if node.inside_while:
            return f'get_max_priority({self.visit(node.children[0])}, {self.visit(node.children[1])}, layer_index=layer_index, counter={node.ttb_counter}, inside_while=True, while_number={node.while_number}, while_iteration=while_iteration)'
        return f'get_max_priority({self.visit(node.children[0])}, {self.visit(node.children[1])}, layer_index=layer_index, counter={node.ttb_counter}, inside_while=False, while_number={node.while_number})'
    
    # def visitIrGetPolyexpStop(self, node):
    #     return 'filter_trav_exp_stop(' + self.visit(node.children[0]) + ', ' + self.visit(node.children[1]) + ')'
    
    # def visitIrGetPolyexpNotStop(self, node):
    #     return 'filter_trav_exp_not_stop(' + self.visit(node.children[0]) + ', ' + self.visit(node.children[1]) + ')'
    
    def visitIrGetPolyexpStop(self, node):
        trav = self.visit(node.children[0])
        stop = self.visit(node.children[1])
        if node.inside_while:
            return f'filter_trav_exp_stop({trav}, {stop}, layer_index=layer_index, counter={node.ttb_counter}, inside_while=True, while_number={node.while_number}, while_iteration=while_iteration)'
        return f'filter_trav_exp_stop({trav}, {stop}, layer_index=layer_index, counter={node.ttb_counter}, inside_while=False, while_number={node.while_number})'

    def visitIrGetPolyexpNotStop(self, node):
        trav = self.visit(node.children[0])
        stop = self.visit(node.children[1])
        if node.inside_while:
            return f'filter_trav_exp_not_stop({trav}, {stop}, layer_index=layer_index, counter={node.ttb_counter}, inside_while=True, while_number={node.while_number}, while_iteration=while_iteration)'
        return f'filter_trav_exp_not_stop({trav}, {stop}, layer_index=layer_index, counter={node.ttb_counter}, inside_while=False, while_number={node.while_number})'

    def visitIrPolyExpMat(self, node):
        return self.visit(node.children[0]) + '.mat'

    def visitIrPolyExpNotStopFloat(self, node):
        if reuse_mode.get_flag():
            return '('+self.visit(node.children[0]) + '.unary(operator.not_)).float()'
        return 'convert_to_float(' + self.visit(node.children[0]) + '.unary(operator.not_))'

    def visitIrBlockPolyexpStop(self, node):
        return self.visit(node.children[0]) + '.create_similar(mat=' + self.visit(node.children[1]) + ')'

    def visitIrBlockPolyexpNotStop(self, node):
        return (self.visit(node.children[0]) + '.create_similar(mat=' +
                self.visit(node.children[1]) + ', const=' + self.visit(node.children[2]) + ')')

    def visitIrBlockAny(self, node):
        return self.visit(node.children[0]) + '.any()'

    def visitIrBlockAll(self, node):
        return self.visit(node.children[0]) + '.all()'

    def visitIrBlockGetDims(self, node):
        return self.visit(node.children[0]) + '.dims'
    
    def visitIrConvertToTensor(self, node):
        repeat_dims = ''
        for i in range(1, len(node.children)):
            repeat_dims += self.visit(node.children[i])
            if i<len(node.children)-1:
                repeat_dims += ', '
        return 'convert_to_tensor(' + self.visit(node.children[0]) + ', [' + repeat_dims +  '])'

    def visitIrMult(self, node):
        op_name = None 
        if node.op == '*':
            op_name = 'operator.mul'
        elif node.op == '/':
            op_name = 'operator.truediv'
        else:
            op_name = node.op
            raise Exception('OP NOT IDENTIFIED', node.op)
        
        [lhsIr, rhsIr] = node.children
        # node.ttb_counter = self.counter
        # self.counter += 1
        # return 'binary' + '(' + self.visit(lhsIr) + ', ' + self.visit(rhsIr) + ', ' + op_name + ')'
        if node.inside_while:
            return 'binary(' + self.visit(lhsIr) + ', ' + self.visit(rhsIr) + ', ' + op_name + ', ' + 'layer_index = layer_index, ' + 'counter = ' + str(node.ttb_counter) + ', inside_while = True' + ', while_number = ' + str(node.while_number) + ', while_iteration=while_iteration) #' + str(node.ttb_counter)
        return 'binary(' + self.visit(lhsIr) + ', ' + self.visit(rhsIr) + ', ' + op_name + ', ' + 'layer_index = layer_index, ' + 'counter = ' + str(node.ttb_counter) + ', inside_while = False' + ', while_number = ' + str(node.while_number) + ') #' + str(node.ttb_counter)
    
    def visitIrInnerProduct(self, node):
        op_name = 'inner_prod'
        
        [lhsIr, rhsIr] = node.children
        if node.inside_while:
            return op_name + '(' + self.visit(lhsIr) + ', ' + self.visit(rhsIr) + ', ' + 'layer_index = layer_index, ' + 'counter = ' + str(node.ttb_counter) + ', inside_while = True' + ', while_number = ' + str(node.while_number) + ', while_iteration=while_iteration) #' + str(node.ttb_counter)
        return op_name + '(' + self.visit(lhsIr) + ', ' + self.visit(rhsIr) + ', ' + 'layer_index = layer_index, ' + 'counter = ' + str(node.ttb_counter) + ', inside_while = False' + ', while_number = ' + str(node.while_number) + ') #' + str(node.ttb_counter)
    
    def visitIrBlockInnerProduct(self, node):
        if node.type == 'equal_dims':
            return self.visit(node.children[0]) + '.matmul_equal_dims(' + self.visit(node.children[1])+ ')'
        elif node.type == 'unequal_dims':
            return self.visit(node.children[0]) + '.matmul_unequal_dims(' + self.visit(node.children[1])+ ')'
        else:
            raise Exception('OP NOT IDENTIFIED', node.type)

    def visitIrDot(self, node):
        [lhsIr, rhsIr] = node.children
        dot_args = (
            ', layer_index = layer_index, '
            + 'counter = ' + str(node.ttb_counter)
            + ', inside_while = ' + ('True' if node.inside_while else 'False')
            + ', while_number = ' + str(node.while_number)
        )
        if node.inside_while:
            dot_args += ', while_iteration = while_iteration'
        if lhsIr.irMetadata[-1].type == 'Neuron':
            return self.visit(lhsIr) + '.dot(' + self.visit(rhsIr) + ', abs_elem.get_poly_size()' + dot_args + ", mats_input = 'rhs')"
        elif rhsIr.irMetadata[-1].type == 'Neuron':
            return self.visit(rhsIr) + '.dot(' + self.visit(lhsIr) + ', abs_elem.get_poly_size()' + dot_args + ", mats_input = 'lhs')"
        elif lhsIr.irMetadata[-1].type == 'Float':
            if node.inside_while:
                return 'inner_prod(' + self.visit(lhsIr) + ', ' + self.visit(rhsIr) + ', ' + 'layer_index = layer_index, ' + 'counter = ' + str(node.ttb_counter) + ', inside_while = True' + ', while_number = ' + str(node.while_number) + ', while_iteration=while_iteration) #' + str(node.ttb_counter)
            return 'inner_prod(' + self.visit(lhsIr) + ', ' + self.visit(rhsIr) + ', ' + 'layer_index = layer_index, ' + 'counter = ' + str(node.ttb_counter) + ', inside_while = False' + ', while_number = ' + str(node.while_number) + ') #' + str(node.ttb_counter)
        else:
            raise Exception('NOT IMPLEMENTED')

    def visitIrTernary(self, node):
        [condIr, lhsIr, rhsIr] = node.children
        return 'where(' + self.visit(condIr) + ', ' + self.visit(lhsIr) + ', ' + self.visit(rhsIr) + ', layer_index = layer_index, counter = ' + str(node.ttb_counter) + ', inside_while = ' + ('True' if node.inside_while else 'False') + ', while_number = ' + str(node.while_number) + ') #' + str(node.ttb_counter)

    def visitIrClamp(self, node):
        [inputIr, const] = node.children
        min_true = node.min_true 
        if node.inside_while:
            return 'clamp(' + self.visit(inputIr) + ', ' + str(const) + ', ' + str(min_true) + ', ' + 'layer_index = layer_index, ' + 'counter = ' + str(node.ttb_counter) + ', inside_while = True' + ', while_number = ' + str(node.while_number) + ', while_iteration=while_iteration) #' + str(node.ttb_counter)
        return 'clamp(' + self.visit(inputIr) + ', ' + str(const) + ', ' + str(min_true) + ', ' + 'layer_index = layer_index, ' + 'counter = ' + str(node.ttb_counter) + ', inside_while = False' + ', while_number = ' + str(node.while_number) + ') #' + str(node.ttb_counter)

    def visitIrCombineToPoly(self, node):
        [coeffIr, constIr, rows] = node.children
        cols = 'poly_size'
        return 'PolyExpSparse(abs_elem.network, ' + self.visit(coeffIr) + ' , ' + self.visit(constIr) + ')'

    def visitIrCombineToSym(self, node):
        [coeffIr, constIr, rows] = node.children
        cols = 'SymExpSparse.count'
        rows = self.visit(rows)
        return 'SymExpSparse(abs_elem.network,' + self.visit(coeffIr) + ', ' + self.visit(constIr) + ')'


    def visitIrExtractPolyCoeff(self, node):
        [inputIr] = node.children
        while_iteration = 'while_iteration' if node.inside_while else 'None'
        return self.visit(inputIr) + '.get_mat(abs_elem' \
            + ', layer_index=layer_index, counter=' + str(node.ttb_counter) + ', inside_while=' + str(node.inside_while) + ', while_number=' + str(node.while_number) + ', while_iteration=' + while_iteration + ')'
    
    def visitIrExtractSymCoeff(self, node):
        [inputIr] = node.children
        while_iteration = 'while_iteration' if node.inside_while else 'None'
        return self.visit(inputIr) + '.get_mat(SymExpSparse.count' \
            + ', layer_index=layer_index, counter=' + str(node.ttb_counter) + ', inside_while=' + str(node.inside_while) + ', while_number=' + str(node.while_number) + ', while_iteration=' + while_iteration + ')'

    def visitIrExtractPolyConst(self, node, ):
        [inputIr] = node.children
        if reuse_mode:
            return self.visit(inputIr) + '.const'
        return self.visit(inputIr) + '.get_const()'
    
    def visitIrExtractSymConst(self, node):
        [inputIr] = node.children
        if reuse_mode:
            return self.visit(inputIr) + '.const'
        return self.visit(inputIr) + '.get_const()'

    def visitIrConvertNeuronToPoly(self, node):
        [inputIr] = node.children
        return self.visit(inputIr) + '.convert_to_poly(abs_elem)'
    
    def visitIrConvertConstToPoly(self, node):
        [inputIr, rows] = node.children
        cols = 'poly_size'
        return 'PolyExpSparse(abs_elem.network, 0.0, ' + self.visit(inputIr) + ')'
        
    def visitIrConvertConstToSym(self, node):
        [inputIr, rows] = node.children
        cols = 'SymExpSparse.count'
        rows = self.visit(rows)
        return 'SymExpSparse(abs_elem.network,' + 'None, ' + self.visit(inputIr) + ')'
    
    def visitIrExpandSymExp(self, node):
        [inputIr] = node.children
        if not reuse_mode.get_flag():
            return self.visit(inputIr) + '.expand_symexp_mat(SymExpSparse.count)'
        var_name = self.visit(inputIr)
        self.write(var_name + '.total_size[-1] = SymExpSparse.count')
        return var_name

    def visitIrAccess(self, node):
        [lhsIr] = node.children
        while_iteration = 'while_iteration' if node.inside_while else 'None'
        if not node.isMetadata:
            return 'abs_elem.get_elem(\'' + node.elem + '\', ' + self.visit(lhsIr) \
                + ', layer_index=layer_index, counter=' + str(node.ttb_counter) + ', inside_while=' + str(node.inside_while) + ', while_number=' + str(node.while_number) + ', while_iteration=' + while_iteration + ')'
        else:
            return self.visit(lhsIr) + '.get_metadata(\'' + node.elem + '\', batch_size' \
                + ', layer_index=layer_index, counter=' + str(node.ttb_counter) + ', inside_while=' + str(node.inside_while) + ', while_number=' + str(node.while_number) + ', while_iteration=' + while_iteration + ')'
        
    def visitIrReduce(self, node):
        [inputIr] = node.children
        size = node.reduce_dim
        if node.inside_while:
            return '(' + self.visit(inputIr) + ').sum(' + str(size) + ', layer_index = layer_index, counter = ' + str(node.ttb_counter) + ', inside_while = True, while_number = ' + str(node.while_number) + ', while_iteration=while_iteration)'
        return '(' + self.visit(inputIr) + ').sum(' + str(size) + ', layer_index = layer_index, counter = ' + str(node.ttb_counter) + ', inside_while = False, while_number = ' + str(node.while_number) + ')'

    def visitIrMapCoeff(self, node):
        [inputIr] = node.children
        while_iteration = 'while_iteration' if node.inside_while else 'None'
        if inputIr.irMetadata[-1].type == 'PolyExp':    
            return self.visit(inputIr) + '.get_mat(abs_elem' \
                + ', layer_index=layer_index, counter=' + str(node.ttb_counter) + ', inside_while=' + str(node.inside_while) + ', while_number=' + str(node.while_number) + ', while_iteration=' + while_iteration + ')'
        return self.visit(inputIr) + '.get_mat(SymExpSparse.count' \
                + ', layer_index=layer_index, counter=' + str(node.ttb_counter) + ', inside_while=' + str(node.inside_while) + ', while_number=' + str(node.while_number) + ', while_iteration=' + while_iteration + ')'

    def visitIrMapNeuron(self, node):
        if node.dims:
            return 'Llist(abs_elem.network, [1]*(' + self.visit(node.children[0]) + '), None, None,' + "torch.nonzero(abs_elem.d['llist']).flatten().tolist())"
        else:
            return 'Llist(abs_elem.network, [1]*(' + self.visit(node.children[0]) + '.mat.dims-1), None, None,' + "torch.nonzero(abs_elem.d['llist']).flatten().tolist())"

    def visitIrSymbolic(self, node):
        return node.name
    
    def visitIrFlow(self, node):
        self.indent += 1
        self.write('flow = Flow(abs_elem, ' + str(node.transformer) + '(), network, print_intermediate_results, no_sparsity)')
        self.write('res = flow.flow()')
        self.write('print("Peak memory usage:", torch.cuda.max_memory_allocated() / 1024**2, "MB")')
        self.write('return res')
        self.indent -= 1

    def visitIrObjectLookup(self, node):
        if node.object_name == "block":
            return self.visit(node.children[0]) + ".block"
        raise Exception("NOT IMPLEMENTED")

    def visitIrBlockCreateSimilar(self, node):
        return (
            self.visit(node.children[0]) +
            ".create_similar(" +
            self.visit(node.children[1]) +
            ")"
        )

    def visitIrSetBlockTotalShapeLastDim(self, node):
        block_var = self.visit(node.children[0])
        value = self.visit(node.children[1])
        self.write(block_var + ".total_shape[-1] = " + value)

    def visitIrAssignToBlock(self, node):
        block_var = self.visit(node.children[0])
        value = self.visit(node.children[1])
        self.write(block_var + ".block = " + value)

    def visitIrBlockGetSubBlockCustomRange(self, node):
        return (
            self.visit(node.children[0]) +
            ".get_sub_block_custom_range(" +
            self.visit(node.start_index) + ", " +
            self.visit(node.end_index) + ", " +
            self.visit(node.block_start_index) +
            ")"
        )
