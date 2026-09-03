"""Scalar replacement of the Jit* aggregates: structure in the compiler, tensors in the code.

Requires splice_flow: one straight-line ir.flow_block. Sets ir.flow_params.
"""

from __future__ import annotations

import os
import re

from constraintflow.compiler.ir import *
from constraintflow.compiler.optimizations import tensor_to_block
from constraintflow.lib import entry_capture
from constraintflow.lib.globals import load_capture


class SroaUnsupported(Exception):
    pass


# aggregate constructors that evaporate
_BLOCK_NODES = {
    'IrDenseBlock': 'DenseBlock',
    'IrConstBlock': 'ConstBlock',
    'IrRepeatBlock': 'RepeatBlock',
    'IrDiagonalBlock': 'DiagonalBlock',
    'IrKernelBlock': 'KernelBlock',
    'IrPatchesBlock': 'PatchesBlock',
}

# in-place writes; their absence licenses dropping copies
_MUTATION = ('IrAssignToBlock', 'IrSetBlockTotalShapeLastDim', 'IrAssignToView')

_BOOL_OPS = {'gt', 'lt', 'ge', 'le', 'eq', 'ne', 'and_', 'or_'}


class TensorDesc:
    __slots__ = ('ir', 'dtype')

    def __init__(self, ir, dtype='f32'):
        self.ir = ir
        self.dtype = dtype


class ScalarDesc:
    __slots__ = ('value', 'dtype', 'ir')

    def __init__(self, value, dtype='py_float', ir=None):
        self.value = value
        self.dtype = dtype
        self.ir = ir


class BlockDesc:
    __slots__ = ('kind', 'payload', 'total_shape', 'fields')

    def __init__(self, kind, payload, total_shape, fields):
        self.kind = kind
        self.payload = payload
        self.total_shape = total_shape
        self.fields = fields


class SparseDesc:
    __slots__ = ('dims', 'total_size', 'start_indices', 'blocks', 'dense_const', 'type')

    def __init__(self, dims, total_size, start_indices, blocks, dense_const, type_):
        self.dims = dims
        self.total_size = total_size
        self.start_indices = start_indices
        self.blocks = blocks
        self.dense_const = dense_const
        self.type = type_


class PolyDesc:
    __slots__ = ('mat', 'const')

    def __init__(self, mat, const):
        self.mat = mat
        self.const = const


class ListDesc:
    __slots__ = ('items',)

    def __init__(self, items):
        self.items = tuple(items)


class LambdaDesc:
    __slots__ = ('op',)

    def __init__(self, op):
        self.op = op


class OpaqueDesc:
    __slots__ = ('reason',)

    def __init__(self, reason):
        self.reason = reason


_LITERAL_TENSOR = re.compile(r'^torch\.tensor\(\[([0-9,\s\.\-]*)\]'
                             r'(?:,\s*dtype=torch\.\w+)?\)$')


def _ints(value):
    """Static int tuple, or None when the value is not statically known."""
    if value is None:
        return None
    if isinstance(value, str):
        match = _LITERAL_TENSOR.match(value.strip())
        if match is None:
            return None
        body = match.group(1).strip()
        if not body:
            return ()
        try:
            return tuple(int(float(x)) for x in body.split(','))
        except ValueError:
            return None
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        try:
            return tuple(int(x) for x in value)
        except (TypeError, ValueError):
            return None
    return None


def _op_name(op):
    return op if isinstance(op, str) else getattr(op, '__name__', '')


def _walk(obj, seen, out):
    if id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, IrAst):
        out.append(obj)
    if isinstance(obj, (list, tuple, set)):
        for item in obj:
            _walk(item, seen, out)
        return
    if hasattr(obj, '__dict__'):
        for value in vars(obj).values():
            _walk(value, seen, out)


def _reads(stmt):
    """Names read by one statement, following .op as well as .children."""
    names, seen = set(), set()

    def walk(obj):
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, IrVar):
            names.add(obj.name)
            return
        if isinstance(obj, (list, tuple)):
            for item in obj:
                walk(item)
            return
        if hasattr(obj, 'children'):
            for child in obj.children:
                walk(child)
            op = getattr(obj, 'op', None)
            if isinstance(op, IrAst):
                walk(op)

    walk(stmt.children[1] if isinstance(stmt, IrAssignment)
         else getattr(stmt, 'children', []))
    return names


def dead_store_elim(stmts):
    """Backward liveness over a straight-line block; exact, multi-def safe."""
    live, keep = set(), []
    for stmt in reversed(stmts):
        if isinstance(stmt, IrDel):
            continue
        if isinstance(stmt, IrAssignment):
            name = stmt.children[0].name
            if name not in live:
                continue
            live.discard(name)
        live |= _reads(stmt)
        keep.append(stmt)
    keep.reverse()
    return keep


def _meta(expr):
    """Most IrTorch* nodes carry none; temps still need a usable element."""
    return expr.irMetadata or [IrMetadataElement([1], 'Float', [1], False)]


class _Scalarizer:
    def __init__(self, ir):
        self.ir = ir
        self.env = {}
        self.cache = {}
        self.out = []
        self.params = []
        self._by_path = {}
        self.functional = True
        self.clones_dropped = 0
        self.lambdas_dropped = 0
        self.casts_dropped = 0
        self.dead_dropped = 0
        self.sym_count = 0
        self.aggregates = 0
        self.survivors = []

    # -- flow parameters -------------------------------------------------
    def param(self, path, name, shape=None, type_='Float'):
        if path not in self._by_path:
            self._by_path[path] = name
            self.params.append((name, path))
        shape = list(shape) if shape else [1]
        return IrVar(self._by_path[path],
                     [IrMetadataElement(shape, type_, [1] * len(shape), False)])

    # -- entry descriptors -----------------------------------------------
    def seed_entry(self):
        capture = load_capture(entry_capture.CAPTURE_PATH)
        for key, var_name in self.ir.flow_entry.items():
            desc = self._from_capture(
                capture[key], "abs_elem.d['" + key + "']", 'in_' + key)
            if self.ir.shape.get(key) == 'SymExp':
                self._note_eps(getattr(desc, 'mat', None))
            self.env[var_name] = desc

    def _from_capture(self, node, path, name):
        kind = node['node']
        if kind == 'poly':
            return PolyDesc(
                self._from_capture(node['mat'], path + '.mat', name + '_mat'),
                self._from_capture(node['const'], path + '.const', name + '_const'))
        if kind == 'scalar':
            return ScalarDesc(node['value'])
        if kind != 'sparse':
            raise SroaUnsupported('entry value ' + name + ' is ' + kind)
        blocks = []
        for i, b in enumerate(node['blocks']):
            payload = TensorDesc(
                self.param(path + '.blocks[' + str(i) + '].block', name + '_' + str(i),
                           b['total_shape'], 'Bool' if b['dtype'] == 'bool' else 'Float'),
                b['dtype'])
            blocks.append(BlockDesc(b['kind'], payload, tuple(b['total_shape']),
                                    dict(b['fields'])))
        return SparseDesc(node['dims'], tuple(node['total_size']),
                          tuple(tuple(s) for s in node['start_indices']),
                          tuple(blocks), node['dense_const'], node['type'])

    # -- symbolic epsilons -----------------------------------------------
    def _note_eps(self, mat):
        """SymExpSparse.count, read back off the mat that last grew it."""
        size = getattr(mat, 'total_size', None)
        if size:
            self.sym_count = max(self.sym_count, size[-1])

    def _new_eps(self, mat, const):
        self._note_eps(mat)
        return PolyDesc(mat, const)

    def _expand_sym(self, desc):
        """expand_symexp_mat: widen the epsilon axis, leaving the blocks alone."""
        if isinstance(desc, PolyDesc):
            return PolyDesc(self._expand_sym(desc.mat), desc.const)
        if isinstance(desc, OpaqueDesc):
            raise SroaUnsupported(desc.reason)
        if not isinstance(desc, SparseDesc):
            raise SroaUnsupported('expand_symexp_mat on ' + type(desc).__name__)
        if not desc.total_size or desc.total_size[-1] >= self.sym_count:
            return desc
        return SparseDesc(desc.dims, desc.total_size[:-1] + (self.sym_count,),
                          desc.start_indices, desc.blocks, desc.dense_const, desc.type)

    # -- evaluation ------------------------------------------------------
    def eval(self, expr):
        if expr is None or isinstance(expr, (int, float, str, bool, type)):
            return expr
        if isinstance(expr, list):
            return ListDesc(())
        key = id(expr)
        if key in self.cache:
            return self.cache[key]
        desc = self._eval(expr)
        self.cache[key] = desc
        return desc

    def _need(self, desc, attr, what):
        if isinstance(desc, OpaqueDesc):
            raise SroaUnsupported(desc.reason)
        if not hasattr(desc, attr):
            raise SroaUnsupported(what + ' on ' + type(desc).__name__)
        return getattr(desc, attr)

    def _at(self, seq, index, what):
        if index >= len(seq) or index < -len(seq):
            raise SroaUnsupported(
                what + '[' + str(index) + '] out of range, len=' + str(len(seq)))
        return seq[index]

    def _index(self, child):
        if isinstance(child, int):
            return child
        if isinstance(child, IrConst):
            return int(child.const)
        desc = self.eval(child)
        if isinstance(desc, ScalarDesc):
            return int(desc.value)
        raise SroaUnsupported('non-constant index')

    def _eval(self, expr):
        name = type(expr).__name__

        if isinstance(expr, IrVar):
            if expr.name not in self.env:
                raise SroaUnsupported('unbound variable ' + expr.name)
            return self.env[expr.name]
        if isinstance(expr, IrConst):
            dtype = 'py_bool' if isinstance(expr.const, bool) else 'py_float'
            return ScalarDesc(expr.const, dtype, expr)
        if isinstance(expr, IrGetKthLayerNetworkParam):
            path = 'abs_elem.network[' + str(expr.layer_index) + '].' + expr.param
            return TensorDesc(self.param(
                path, 'net_' + str(expr.layer_index) + '_' + expr.param))

        # lists of blocks
        if isinstance(expr, IrEmptyList):
            self.aggregates += 1
            return ListDesc(())
        if isinstance(expr, IrAppendList):
            self.aggregates += 1
            head = self._need(self.eval(expr.children[0]), 'items', 'list append')
            return ListDesc(head + (self.eval(expr.children[1]),))
        if isinstance(expr, IrListExtract):
            items = self._need(self.eval(expr.children[0]), 'items', 'list extract')
            return self._at(items, self._index(expr.children[1]), 'list')
        if isinstance(expr, IrGetSparseTensorBlocks):
            return ListDesc(self._need(self.eval(expr.children[0]), 'blocks', 'blocks'))
        if isinstance(expr, IrBlockExtract):
            blocks = self._need(self.eval(expr.children[0]), 'blocks', 'block extract')
            return self._at(blocks, self._index(expr.children[1]), 'blocks')
        if isinstance(expr, IrBlockCopy):
            block = self.eval(expr.children[0])
            if self.functional:
                self.clones_dropped += 1
                return block
            raise SroaUnsupported('copy in a non-functional flow')
        if isinstance(expr, IrObjectLookup):
            if expr.object_name != 'block':
                raise SroaUnsupported('object lookup ' + str(expr.object_name))
            return self._need(self.eval(expr.children[0]), 'payload', '.block')

        # sparse tensors and blocks
        if isinstance(expr, IrSparseTensor):
            self.aggregates += 1
            blocks_child = expr.children[0] if expr.children else []
            items = (() if isinstance(blocks_child, list)
                     else self._need(self.eval(blocks_child), 'items', 'sparse blocks'))
            return SparseDesc(
                expr.dims, _ints(expr.total_size),
                tuple(_ints(s) for s in expr.start_indices),
                items, getattr(expr, 'dense_const', 0.0),
                getattr(expr, 'type', float))
        if name in _BLOCK_NODES:
            self.aggregates += 1
            kind = _BLOCK_NODES[name]
            fields = {f: getattr(expr, f) for f in entry_capture.BLOCK_GEOMETRY[kind]
                      if hasattr(expr, f)}
            return BlockDesc(kind, self.eval(expr.children[0]),
                             _ints(expr.total_shape), fields)

        # polyexps
        if isinstance(expr, (IrGetPolyExpSparseMat, IrPolyExpMat,
                             IrGetSymExpSparseMat)):
            return self._need(self.eval(expr.children[0]), 'mat', '.mat')
        if isinstance(expr, (IrGetPolyExpSparseConst, IrExtractPolyConst,
                             IrGetSymExpSparseConst, IrExtractSymConst)):
            return self._need(self.eval(expr.children[0]), 'const', '.const')
        if isinstance(expr, (IrCombineToPoly, IrCombineToSym)):
            self.aggregates += 1
            return PolyDesc(self.eval(expr.children[0]), self.eval(expr.children[1]))
        if isinstance(expr, IrNewEps):
            self.aggregates += 1
            return self._new_eps(self.eval(expr.children[0]),
                                 self.eval(expr.children[1]))
        if isinstance(expr, IrExpandSymExp):
            return self._expand_sym(self.eval(expr.children[0]))
        if isinstance(expr, IrBlockPolyexpStop):
            self.aggregates += 1
            return PolyDesc(self.eval(expr.children[1]),
                            self._need(self.eval(expr.children[0]), 'const', '.const'))
        if isinstance(expr, IrBlockPolyexpNotStop):
            self.aggregates += 1
            return PolyDesc(self.eval(expr.children[1]), self.eval(expr.children[2]))

        if isinstance(expr, IrConvertBoolToFloat):
            return self._to_float(self.eval(expr.children[0]))
        if isinstance(expr, IrLambda):
            return LambdaDesc(expr.op)
        if isinstance(expr, IrBlockGetDims):
            dims = self._need(self.eval(expr.children[0]), 'dims', '.dims')
            return ScalarDesc(dims, 'py_float', IrConst(dims, 'Int'))
        if isinstance(expr, IrConvertConstToPoly):
            self.aggregates += 1
            return PolyDesc(ScalarDesc(0.0), self.eval(expr.children[0]))
        if isinstance(expr, IrConvertConstToSym):
            self.aggregates += 1
            return PolyDesc(None, self.eval(expr.children[0]))
        if isinstance(expr, IrSimpleUnary):
            return self._simple_unary(expr)

        return self._tensor_op(expr)

    def _to_float(self, desc):
        """.float() maps over structure; a payload already float is left alone."""
        if isinstance(desc, SparseDesc):
            blocks = tuple(BlockDesc(b.kind, self._to_float(b.payload),
                                     b.total_shape, b.fields) for b in desc.blocks)
            return SparseDesc(desc.dims, desc.total_size, desc.start_indices,
                              blocks, float(desc.dense_const), float)
        if isinstance(desc, BlockDesc):
            return BlockDesc(desc.kind, self._to_float(desc.payload),
                             desc.total_shape, desc.fields)
        if isinstance(desc, ScalarDesc):
            return ScalarDesc(float(desc.value), 'py_float')
        if isinstance(desc, TensorDesc):
            if desc.dtype == 'f32':
                self.casts_dropped += 1
                return desc
            node = IrConvertBoolToFloat(desc.ir)
            var = IrVar(tensor_to_block.get_var(), _meta(node))
            self.out.append(IrAssignment(var, node))
            return TensorDesc(var, 'f32')
        if isinstance(desc, OpaqueDesc):
            raise SroaUnsupported(desc.reason)
        raise SroaUnsupported('.float() on ' + type(desc).__name__)

    def _simple_unary(self, expr):
        """Fold the identity lambdas; lower the negating one to operator.neg."""
        op = expr.op
        if isinstance(op, IrVar) and op.name in self.env:
            op = self.env[op.name]
        if isinstance(op, IrLambda):
            op = LambdaDesc(op.op)
        if isinstance(op, LambdaDesc):
            if op.op in ('add', 'mul', 'and_', 'or_'):
                self.lambdas_dropped += 1
                return self.eval(expr.children[0])
            if op.op != 'sub':
                raise SroaUnsupported('lambda op ' + str(op.op))
            expr.op = '-'
        return self._tensor_op(expr)

    # -- tensor ops ------------------------------------------------------
    def _tensor_op(self, expr):
        op = getattr(expr, 'op', None)
        if isinstance(op, IrVar) and op.name in self.env:
            raise SroaUnsupported('computed operator ' + op.name)
        descs = [self.eval(child) for child in expr.children]
        try:
            expr.update_parent_child([self.to_ir(d) for d in descs])
        except SroaUnsupported as exc:
            raise SroaUnsupported(
                type(expr).__name__ + '('
                + ', '.join(type(d).__name__ for d in descs) + '): ' + str(exc)) from None
        var = IrVar(tensor_to_block.get_var(), _meta(expr))
        self.out.append(IrAssignment(var, expr))
        return TensorDesc(var, self._result_dtype(expr, descs))

    def to_ir(self, desc):
        if isinstance(desc, TensorDesc):
            return desc.ir
        if isinstance(desc, ScalarDesc):
            if desc.ir is not None:
                return desc.ir
            return IrConst(desc.value, 'Bool' if desc.dtype == 'py_bool' else 'Float')
        if isinstance(desc, (int, float, str)) or desc is None:
            return desc
        if isinstance(desc, OpaqueDesc):
            raise SroaUnsupported(desc.reason)
        raise SroaUnsupported(type(desc).__name__ + ' used where a tensor was expected')

    def _result_dtype(self, expr, descs):
        name = type(expr).__name__
        if name == 'IrConvertBoolToFloat':
            return 'f32'
        if name == 'IrSimpleBinary' and _op_name(expr.op) in _BOOL_OPS:
            return 'bool'
        if name == 'IrSimpleUnary' and _op_name(expr.op) in ('not', 'not_'):
            return 'bool'
        kinds = {d.dtype for d in descs if isinstance(d, (TensorDesc, ScalarDesc))}
        if name == 'IrTorchWhere' and len(descs) == 3:
            return getattr(descs[1], 'dtype', 'f32')
        if 'f32' in kinds or 'py_float' in kinds or not kinds:
            return 'f32'
        return 'bool'

    # -- result ----------------------------------------------------------
    def densify(self, desc):
        if not isinstance(desc, SparseDesc):
            raise SroaUnsupported('densify expects a sparse tensor')
        blocks = desc.blocks
        if (len(blocks) == 1 and blocks[0].kind == 'DenseBlock'
                and desc.start_indices and desc.start_indices[0] == (0,) * desc.dims
                and blocks[0].total_shape is not None
                and blocks[0].total_shape == desc.total_size):
            return blocks[0].payload
        raise SroaUnsupported(
            'densify: only a single full-cover DenseBlock is supported, got '
            + str([(b.kind, b.total_shape) for b in blocks])
            + ' total_size=' + str(desc.total_size)
            + ' start=' + str(desc.start_indices) + ' dims=' + str(desc.dims))

    def run(self):
        block = self.ir.flow_block
        nodes = []
        _walk(block.children, set(), nodes)
        self.functional = not any(type(n).__name__ in _MUTATION for n in nodes)

        self.seed_entry()
        live_stmts = dead_store_elim(block.children)
        self.dead_dropped = len(block.children) - len(live_stmts)
        ret = None
        for stmt in live_stmts:
            if isinstance(stmt, IrAssignment):
                mark_out, mark_cache = len(self.out), set(self.cache)
                try:
                    desc = self.eval(stmt.children[1])
                except (SroaUnsupported, AttributeError, IndexError, KeyError) as exc:
                    if os.environ.get('CF_SROA_STRICT'):
                        raise
                    del self.out[mark_out:]
                    for key in set(self.cache) - mark_cache:
                        del self.cache[key]
                    desc = OpaqueDesc(str(exc))
                    self.survivors.append(str(exc))
                self.env[stmt.children[0].name] = desc
            elif isinstance(stmt, IrTransRetBasic):
                ret = stmt
            elif isinstance(stmt, IrDel):
                continue
            else:
                raise SroaUnsupported('statement ' + type(stmt).__name__)
        if ret is None:
            raise SroaUnsupported('flow has no return')

        results = [self.densify(self.eval(child)) for child in ret.children[:2]]
        self.out.append(IrTransRetBasic([self.to_ir(d) for d in results]))
        block.update_parent_child(self.out)
        self.ir.flow_params = self.params
        return self


def sroa(ir):
    """
    public
    Requires: splice_flow has run, so ir.flow_block is one straight-line block.
    Ensures: ir.flow_block holds only tensor ops; ir.flow_params names its inputs.
    """
    run = _Scalarizer(ir).run()
    return {
        'aggregates': run.aggregates,
        'clones_dropped': run.clones_dropped,
        'lambdas_dropped': run.lambdas_dropped,
        'casts_dropped': run.casts_dropped,
        'dead_dropped': run.dead_dropped,
        'functional': run.functional,
        'params': len(run.params),
        'statements': len(run.out),
        'survivors': run.survivors,
    }
