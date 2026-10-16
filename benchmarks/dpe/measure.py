"""Measure an isolated DPE kernel and compare saved reference bounds/timings."""
import argparse,json,math,os,statistics,subprocess,sys,hashlib
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--kernel',required=True)
p.add_argument('--program',default='examples/compiler_examples/deeppoly_efficient.cf')
p.add_argument('--config',default='N33',choices=['N33','N34','N35','N36','N37'])
p.add_argument('--images',type=int,default=10)
p.add_argument('--warmup',type=int,default=5)
p.add_argument('--tag',required=True)
p.add_argument('--eager',action='store_true')
p.add_argument('--reference',help='Saved JSONL bounds for the same model, epsilon, and images; requires max absolute difference <= 5e-5')
a=p.parse_args()
root=Path(__file__).resolve().parents[2]
harness=Path('/home/db50/cf-harness')
nets={'N33':('ResNetTiny_DiffAI',.001),'N34':('ResNetTiny_PGD',.00004),'N35':('ResNet18_DiffAI',.001),'N36':('ResNet18_PGD',.00004),'N37':('ResNet34_DiffAI',.001)}
name,eps=nets[a.config]
out=root/'optimization-results'/a.tag
out.mkdir(parents=True,exist_ok=True)
env=dict(os.environ,PYTHONPATH=str(root)+':'+str(harness),CF_DISABLE_TF32='1',NVIDIA_TF32_OVERRIDE='0',TORCHINDUCTOR_COALESCE_TILING_ANALYSIS='0',PYTHONUNBUFFERED='1')
if a.eager:env['TORCHDYNAMO_DISABLE']='1'
cmd=[sys.executable,'constraintflow/cli.py','run',a.program,'--network',str(harness/'nets/cifar'/f'{name}.onnx'),'--dataset','cifar','--eps',str(eps),'--batch-size','1','--device','gpu','--inductor','--warmup',str(a.warmup),'--num-images',str(a.images),'--results-file',str(out/'cf.jsonl'),'--output-path',str(Path(a.kernel).resolve())]
(out/'command.json').write_text(json.dumps(cmd,indent=2))
with (out/'run.log').open('w') as f:
 result=subprocess.run(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
if result.returncode:sys.exit(result.returncode)
read=lambda path:[r for line in path.read_text().splitlines() if (r:=json.loads(line)).get('kind')=='batch']
rows=read(out/'cf.jsonl')
if len(rows)!=a.images:raise RuntimeError(f'Expected {a.images} measured batches, got {len(rows)}')
def flat_bounds(x):
 if isinstance(x,list):return [v for sub in x for v in flat_bounds(sub)]
 return [x]
report={'config':a.config,'program':a.program,'kernel':str(Path(a.kernel).resolve()),'kernel_sha256':hashlib.sha256((Path(a.kernel)/'transformers.py').read_bytes()).hexdigest(),'eps':eps,'tf32':False,'median_s':statistics.median(r['time_s'] for r in rows),'peak_bytes':max(r['peak_bytes'] for r in rows),'num_images':len(rows),'eager':a.eager}
report['bounds_finite']=all(math.isfinite(x) for r in rows for side in ('lower','upper') for x in flat_bounds(r[side]))
report['upper_ge_lower']=all(u>=l-1e-5 for r in rows for l,u in zip(flat_bounds(r['lower']),flat_bounds(r['upper'])))
for label,prefix,cert,tool in [('dpe','dpe-resnets','deeppoly_efficient','cf'),('deeppoly_add','resnets-v2','deeppoly_add','cf'),('lirpa','resnets-v2','deeppoly_add','lirpa')]:
 path=harness/'experiments/results'/prefix/cert/'tools'/tool/a.config/f'{tool}.jsonl'
 if not path.exists():continue
 ref=read(path)
 if not ref:continue
 report[label+'_median_s']=statistics.median(r['time_s'] for r in ref)
 report[label+'_speedup']=report[label+'_median_s']/report['median_s']
 diffs=[];ok=True;verified_agree=0
 for r,s in zip(rows,ref):
  verified_agree += all(v > 0 for v in flat_bounds(r['lower'])) == all(v > 0 for v in flat_bounds(s['lower']))
  for side in ('lower','upper'):
   def flat(x):
    if isinstance(x,list):
     return [v for sub in x for v in flat(sub)]
    return [x]
   x,y=flat(r[side]),flat(s[side])
   if len(x)!=len(y):raise ValueError('bound shape mismatch')
   for u,v in zip(x,y):
    diffs.append(abs(u-v));ok &= math.isfinite(u) and math.isfinite(v) and abs(u-v)<=.001+.001*abs(v)
 report[label+'_max_abs_diff']=max(diffs)
 report[label+'_bounds_match']=ok
 report[label+'_atol_5e_5_match']=max(diffs)<=5e-5
 report[label+'_verified_agree']=verified_agree
fresh=out.parent/(a.config+'-lirpa-fresh.jsonl')
if fresh.exists():
 ref=read(fresh)
 if ref:
  report['lirpa_fresh_median_s']=statistics.median(r['time_s'] for r in ref)
  report['lirpa_fresh_speedup']=report['lirpa_fresh_median_s']/report['median_s']
  report['lirpa_fresh_max_abs_diff']=max(abs(u-v) for r,s in zip(rows,ref) for side in ('lower','upper') for u,v in zip(flat_bounds(r[side]),flat_bounds(s[side])))
if a.reference:
 path=Path(a.reference).resolve();ref=read(path)
 if len(ref)<len(rows):raise RuntimeError('Reference must cover every measured image')
 diffs=[];agree=0;finite=True
 for r,s in zip(rows,ref):
  if r['images']!=s['images']:raise RuntimeError('Reference image indices do not match')
  agree+=all(x>0 for x in flat_bounds(r['lower']))==all(x>0 for x in flat_bounds(s['lower']))
  for side in ('lower','upper'):
   x,y=flat_bounds(r[side]),flat_bounds(s[side])
   if len(x)!=len(y):raise RuntimeError('Reference bound shapes do not match')
   finite &= all(math.isfinite(v) for v in x+y)
   diffs.extend(abs(u-v) for u,v in zip(x,y))
 report.update(reference=str(path),reference_images=len(rows),reference_max_abs_diff=max(diffs),reference_verified_agree=agree,reference_bounds_match=finite and max(diffs)<=5e-5 and agree==len(rows))
(out/'summary.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)

if a.reference and not report['reference_bounds_match']:raise RuntimeError('Bounds or verification decisions changed relative to the saved reference')
