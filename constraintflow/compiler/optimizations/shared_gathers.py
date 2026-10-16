#FEAT: new pass, enabled by CF_SHARED_GATHERS=1
"""Share repeated small unfolds of bound/slope images across large row batches.

Keep each selected unfold at its first use and expose its result to the region
partitioner. This prevents a fused consumer from recomputing image-coordinate
arithmetic once for every coefficient row. Only functional SSA is supported.
"""
import copy
from collections import Counter
import torch
from constraintflow.compiler.ir import IrAssignment, IrVar, IrFUnfold
from constraintflow.compiler.optimizations import subexp_inlining
from constraintflow.compiler.optimizations.product_grids import _meta_env


def run(block, render, sized, batch_size, min_bytes=256*1024, max_bytes=64*1024**2):
    env = _meta_env(sized, batch_size)
    defs = {s.children[0].name:i for i,s in enumerate(block.children)
            if isinstance(s,IrAssignment)}
    occupied = {v.name for s in block.children for v in
                subexp_inlining.get_vars_expr_occurrences(s.children)}
    groups = {}
    sources = {}
    def walk(e,index):
        if not getattr(e,'children',None):
            return
        for c in e.children:walk(c,index)
        if isinstance(e,IrFUnfold):
            text = sources.setdefault(id(e), (e,render(e)))[1]
            groups.setdefault(text,[]).append((index,e))
    for i,s in enumerate(block.children):
        walk(s.children[1] if isinstance(s,IrAssignment) else s,i)
    inserts,replace,barriers = {},{},set()
    counter=0
    for text,uses in groups.items():
        if len(uses)<2:
            continue
        first,expr=uses[0]
        deps=subexp_inlining.get_vars_expr_occurrences(expr.children)
        if any(defs.get(v.name,-1)>=first for v in deps):
            continue
        try:
            with torch.device('meta'):
                x=eval(render(expr.children[0]),dict(env))
                y=eval(text,dict(env))
            size=y.numel()*y.element_size()
            if x.ndim!=4 or x.shape[0]>batch_size or not min_bytes<=size<=max_bytes:
                continue
        except Exception:
            continue
        while 'shared_gather_'+str(counter) in occupied:counter+=1
        name='shared_gather_'+str(counter);counter+=1;occupied.add(name)
        value=IrVar(name,expr.irMetadata)
        inserts.setdefault(first,[]).append((value,expr))
        barriers.add(name)
        for _,node in uses:replace[id(node)]=value
    def rewrite(e,root=True):
        if root and id(e) in replace:return copy.copy(replace[id(e)])
        if not getattr(e,'children',None):return e
        out=copy.copy(e);out.update_parent_child([rewrite(c) for c in e.children]);return out
    out=[]
    for i,s in enumerate(block.children):
        for value,expr in inserts.get(i,()):out.append(IrAssignment(value,rewrite(expr,False)))
        if isinstance(s,IrAssignment):s.update_parent_child([s.children[0],rewrite(s.children[1])])
        else:s.update_parent_child([rewrite(c) for c in s.children])
        out.append(s)
    block.update_parent_child(out)
    return barriers
