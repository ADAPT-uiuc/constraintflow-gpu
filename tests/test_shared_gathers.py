import operator,tempfile,unittest
import torch
import torch.nn.functional as F
from constraintflow.compiler.ir import *
from constraintflow.compiler.codeGen import CodeGen
from constraintflow.compiler.optimizations.shared_gathers import run

class Shapes(dict):inputs={}

class SharedGathersTest(unittest.TestCase):
    def test_shared_gather_is_reused_without_changing_values(self):
        v=lambda n:IrVar(n,[])
        def gather():return IrFUnfold(v('x'),[3,2],[1,0],[1,1])
        a,b=gather(),gather()
        block=IrBlock([IrAssignment(v('r'),IrSimpleBinary(a,b,'add')),IrTransRetBasic([v('r'),v('r')])])
        sized=Shapes();sized.inputs={'x':{'shape':[1,3,5,6],'dtype':'torch.float32'}}
        with tempfile.TemporaryDirectory() as d:
            cg=CodeGen(d);cg.file.close();before=cg.visit(block.children[0].children[1])
            names=run(block,cg.visit,sized,1,min_bytes=0)
            self.assertEqual(len(names),1)
            x=torch.randn(1,3,5,6)
            env=dict(x=x,torch=torch,F=F,operator=operator)
            expected=eval(before,env)
            for s in block.children:
                if isinstance(s,IrAssignment):env[s.children[0].name]=eval(cg.visit(s.children[1]),env)
                else:actual=eval(cg.visit(s.children[0]),env)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    def test_query_row_images_are_not_materialized(self):
        v=lambda n:IrVar(n,[])
        a=IrFUnfold(v('x'),[3,3],[1,1],[1,1])
        block=IrBlock([IrAssignment(v('r'),IrSimpleBinary(a,a,'add')),IrTransRetBasic([v('r'),v('r')])])
        sized=Shapes();sized.inputs={'x':{'shape':[16,3,5,6],'dtype':'torch.float32'}}
        with tempfile.TemporaryDirectory() as d:
            cg=CodeGen(d);cg.file.close()
            self.assertFalse(run(block,cg.visit,sized,1,min_bytes=0))

if __name__=='__main__':unittest.main()
