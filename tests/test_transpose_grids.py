import operator
import tempfile
import unittest
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from constraintflow.compiler.ir import *
from constraintflow.compiler.codeGen import CodeGen
from constraintflow.compiler.optimizations.product_grids import run

class Shapes(dict):
    inputs={}

class TransposeGridsTest(unittest.TestCase):
    def test_full_product_grid(self):
        torch.manual_seed(9)
        v=lambda n:IrVar(n,[])
        tensors={n:torch.randn(2,3,4,5) for n in ('x','y')}
        tensors.update({n:torch.randn(3,2,3,2) for n in ('w','z')})
        nodes=[IrFConvTranspose2d(v(x),v(w),[2,1],[1,0],[1,0]) for x in ('x','y') for w in ('w','z')]
        block=IrBlock([IrAssignment(v('r'+str(i)),e) for i,e in enumerate(nodes)]+[IrTransRetBasic([v('r'+str(i)) for i in range(4)])])
        sized=Shapes();sized.inputs={n:{'shape':list(x.shape),'dtype':str(x.dtype)} for n,x in tensors.items()}
        with tempfile.TemporaryDirectory() as d:
            cg=CodeGen(d);cg.file.close()
            before=[cg.visit(e) for e in nodes]
            self.assertEqual(run(block,cg.visit,sized,2),4)
            env=dict(tensors,torch=torch,F=F,operator=operator,conv_transpose2d=F.conv_transpose2d)
            expected=[eval(e,env) for e in before]
            for s in block.children:
                if isinstance(s,IrAssignment):env[s.children[0].name]=eval(cg.visit(s.children[1]),env)
                else:actual=[eval(cg.visit(c),env) for c in s.children]
            for x,y in zip(actual,expected):torch.testing.assert_close(x,y)

    def test_dependent_convolution_is_not_hoisted(self):
        v=lambda n:IrVar(n,[])
        a=IrFConvTranspose2d(v('x'),v('w'),[1,1],[1,1],[0,0])
        b=IrFConvTranspose2d(v('a'),v('w'),[1,1],[1,1],[0,0])
        block=IrBlock([IrAssignment(v('a'),a),IrAssignment(v('b'),b),IrTransRetBasic([v('b'),v('b')])])
        sized=Shapes();sized.inputs={n:{'shape':s,'dtype':'torch.float32'} for n,s in [('x',[1,2,4,4]),('w',[2,2,3,3])]}
        sized[0]={'name':'a','shape':[1,2,4,4],'dtype':'torch.float32'}
        with tempfile.TemporaryDirectory() as d:
            cg=CodeGen(d);cg.file.close()
            self.assertEqual(run(block,cg.visit,sized,1),0)

if __name__=='__main__':unittest.main()
