"""Profile one CROWN-IBP training step (as simple_training.py does it) on N32 for one tool.

    python profile_step.py {lirpa,cf} OUT_DIR   (run from a folder with data/cifar-10-batches-py)
"""
import sys
import time

import torch
import torch.nn.functional as F

H = '/home/db50/cf-harness'
sys.path[:0] = [f'{H}/auto_LiRPA', f'{H}/auto_LiRPA/examples/vision']
from simple_verification import get_pytorch_from_onnx

tool, out = sys.argv[1], sys.argv[2]
torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(0)
BS, EPS = 256, 2 / 255
mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).cuda()
std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1).cuda()

model = get_pytorch_from_onnx(f'{H}/nets/cifar/convBigRELU__DiffAI.onnx')
model[-1] = torch.nn.Identity()  # as compare_driver: lets the spec fold into the last Linear
model = model.cuda().train()
opt = torch.optim.Adam(model.parameters(), lr=5e-4)
x = ((torch.rand(BS, 3, 32, 32, device='cuda') - mean) / std)
y = torch.randint(0, 10, (BS,), device='cuda')
x_L = torch.max(x - EPS / std, (0 - mean) / std)
x_U = torch.min(x + EPS / std, (1 - mean) / std)
c = torch.eye(10, device='cuda')[y].unsqueeze(1) - torch.eye(10, device='cuda').unsqueeze(0)
c = c[y.unsqueeze(1) != torch.arange(10, device='cuda').unsqueeze(0)].view(BS, 9, 10)

if tool == 'lirpa':
    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm
    bounded = BoundedModule(model, torch.zeros(2, 3, 32, 32), bound_opts={'conv_mode': 'patches'}, device='cuda')

    def begin():
        pass

    def forward():
        return bounded(BoundedTensor(x, PerturbationLpNorm(norm=float('inf'), eps=EPS, x_L=x_L, x_U=x_U)))

    def bounds():  # CROWN-FAST in simple_training.py
        bounded.compute_bounds(IBP=True, C=c, method=None)
        return bounded.compute_bounds(IBP=False, C=c, method='backward', bound_upper=False)[0]
else:
    from constraintflow.lib.training import TrainingBounds
    cf = TrainingBounds(model, torch.zeros(2, 3, 32, 32), 'cifar', ['CROWN-IBP'], [BS], f'{out}/cf_cache')

    def begin():
        cf.begin_step()

    def forward():
        return model(x)

    def bounds():
        return cf.compute_bounds('CROWN-IBP', x_L, x_U, y)[0]


def step(times=None):
    def mark(name, t0):
        torch.cuda.synchronize()
        t = time.perf_counter()
        if times is not None:
            times[name] = times.get(name, 0) + t - t0
        return t
    torch.cuda.synchronize()
    t = time.perf_counter()
    begin()
    opt.zero_grad()
    logits = forward()
    ce = F.cross_entropy(logits, y)
    t = mark('model forward', t)
    lb = bounds()
    t = mark('bounds', t)
    robust = F.cross_entropy(-torch.cat([torch.zeros_like(lb[:, :1]), lb], 1), torch.zeros_like(y))
    robust.backward()
    t = mark('backward', t)
    opt.step()
    mark('optimizer', t)
    return ce


for _ in range(5):
    step()
times, n = {}, 20
for _ in range(n):
    step(times)
total = sum(times.values()) / n
print(f'{tool}: {total * 1000:.1f} ms/step  ' + '  '.join(f'{k} {v / n * 1000:.1f} ms' for k, v in times.items()))

from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(3):
        step()
    torch.cuda.synchronize()
table = prof.key_averages().table(sort_by='cuda_time_total', row_limit=25)
with open(f'{out}/profile_{tool}.txt', 'w') as f:
    f.write(table)
print(table)
