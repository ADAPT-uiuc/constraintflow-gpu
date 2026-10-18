"""Run auto_LiRPA's or cf-gpu's simple_training.py on an ONNX MNIST network, measured the same way for both.

    python compare_driver.py --tool {lirpa,cf} --onnx NET.onnx --out DIR [--gpu_data] -- <simple_training.py args>
    python compare_driver.py ... --bounds_ckpt CKPT -- <args>   # bounds and gradients on fixed weights

Timed: setup (BoundedModule / TrainingBounds construction) and every Train() call.
Not timed: cf-gpu code generation (done ahead, into its cache) and warmup.
"""
import argparse
import importlib.util
import inspect
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def find_lirpa_dir(path=HERE):
    while not os.path.isdir(os.path.join(path, 'auto_LiRPA', 'examples', 'vision')):
        if path == os.path.dirname(path):
            raise RuntimeError('auto_LiRPA checkout not found')
        path = os.path.dirname(path)
    return os.path.join(path, 'auto_LiRPA')


LIRPA_DIR = find_lirpa_dir()
sys.path[:0] = [LIRPA_DIR, os.path.join(LIRPA_DIR, 'examples', 'vision')]

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import models  # auto_LiRPA's example model registry
from simple_verification import get_pytorch_from_onnx  # the verification sweeps' ONNX loader
from auto_LiRPA.eps_scheduler import FixedScheduler

METHODS = ['IBP', 'CROWN-IBP']
MNIST_SIZES = (60000, 10000)  # train, test


class GPULoader:
    """DataLoader replacement with MNIST on the GPU; same batches and default-RNG use as DataLoader."""

    def __init__(self, dataset, batch_size=1, shuffle=False, **ignored):
        assert isinstance(dataset, datasets.MNIST) and isinstance(dataset.transform, transforms.ToTensor)
        self.dataset, self.batch_size, self.shuffle = dataset, batch_size, shuffle
        self.x = (dataset.data.float() / 255).unsqueeze(1).cuda()  # divide on the CPU, like ToTensor
        self.y = dataset.targets.cuda()

    def __setattr__(self, name, value):  # simple_training sets mean/std as CPU tensors
        super().__setattr__(name, value.cuda() if name in ('mean', 'std') else value)

    def __len__(self):
        return -(-len(self.y) // self.batch_size)

    def __iter__(self):
        torch.empty((), dtype=torch.int64).random_()  # DataLoader draws this base seed every epoch
        n = len(self.y)
        order = torch.tensor(list(torch.utils.data.RandomSampler(range(n)))) if self.shuffle else torch.arange(n)
        for idx in order.cuda().split(self.batch_size):
            yield self.x[idx], self.y[idx]


class FixedBatches(list):
    """Warmup loader: the same batch, three times (CUDA graphs warm up, record, then replay)."""

    def __init__(self, batch_size):
        data = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
        x = torch.stack([data[i][0] for i in range(batch_size)])
        y = torch.tensor([data[i][1] for i in range(batch_size)])
        super().__init__([(x, y)] * 3)
        self.dataset, self.batch_size = [None] * (3 * batch_size), batch_size
        self.mean, self.std = torch.tensor([0.0]), torch.tensor([1.0])


class WarmupScheduler(FixedScheduler):
    def get_max_eps(self):  # CROWN-IBP's mixing factor becomes 0.5, so both bound methods run
        return 2 * self.max_eps


class NoStepOptimizer:
    def __init__(self, params):
        self.params = list(params)

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        pass


def load_script(tool, argv):
    """Import the tool's simple_training.py (it parses argv at import; main() is not run)."""
    path = os.path.join(LIRPA_DIR, 'examples', 'vision', 'simple_training.py') if tool == 'lirpa' else \
        os.path.join(HERE, 'simple_training.py')
    sys.argv = [path] + argv
    spec = importlib.util.spec_from_file_location('simple_training_' + tool, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synced_seconds(fn, *args, **kwargs):
    torch.cuda.synchronize()
    start = time.perf_counter()
    fn(*args, **kwargs)
    torch.cuda.synchronize()
    return time.perf_counter() - start


def batch_sizes(batch_size):
    return sorted({batch_size} | {n % batch_size for n in MNIST_SIZES} - {0})


class Recorder:
    """Times setup and Train() calls in a loaded simple_training module and records losses and meters."""

    def __init__(self, st, tool):
        self.st, self.args = st, st.args
        self.result = {'tool': tool, 'args': vars(st.args), 'records': []}
        self.losses, self.meters, self.warm = [], [], False
        self.train_signature = inspect.signature(st.Train)
        self.original_train = st.Train
        st.Train = self.timed_train
        st.CrossEntropyLoss = self._wrap_loss(st.CrossEntropyLoss)
        st.MultiAverageMeter = self._wrap_meter(st.MultiAverageMeter)
        setup_name = 'BoundedModule' if tool == 'lirpa' else 'TrainingBounds'
        setattr(st, setup_name, self._wrap_setup(getattr(st, setup_name)))

    def _wrap_loss(self, cls):
        losses = self.losses

        class RecordedLoss(cls):
            def forward(self, *x, **k):
                loss = super().forward(*x, **k)
                losses.append(loss.detach())
                return loss
        return RecordedLoss

    def _wrap_meter(self, cls):
        meters = self.meters

        class RecordedMeter(cls):
            def __init__(self, *x, **k):
                super().__init__(*x, **k)
                meters.append(self)
        return RecordedMeter

    def _wrap_setup(self, cls):
        result = self.result

        class TimedSetup(cls):
            def __init__(self, *x, **k):
                result['setup_seconds'] = synced_seconds(super().__init__, *x, **k)
        return TimedSetup

    def warmup(self, call):
        """Run Train() on fixed batches of every batch size, train and eval, without updating weights."""
        model = call.arguments['model']
        for batch_size in batch_sizes(self.args.batch_size):
            for train in (True, False):
                warm = self.train_signature.bind(*call.args, **call.kwargs)
                warm.arguments.update(loader=FixedBatches(batch_size), eps_scheduler=WarmupScheduler(self.args.eps),
                                      train=train, opt=NoStepOptimizer(model.parameters()) if train else None)
                with torch.set_grad_enabled(train):
                    self.original_train(*warm.args, **warm.kwargs)
        model.zero_grad(set_to_none=True)
        self.losses.clear()
        self.meters.clear()

    def timed_train(self, *args, **kwargs):
        call = self.train_signature.bind(*args, **kwargs)
        if not self.warm:
            self.warm = True
            self.result['warmup_seconds'] = synced_seconds(self.warmup, call)
        self.losses.clear()
        seconds = synced_seconds(self.original_train, *args, **kwargs)
        meter = self.meters[-1]
        self.result['records'].append({
            'epoch': call.arguments['t'], 'train': bool(call.arguments['train']), 'seconds': seconds,
            'ce': [loss.item() for loss in self.losses],
            'avg': {k: meter.sum_meter[k] / meter.counts_meter[k] for k in meter.sum_meter if meter.counts_meter[k]},
        })


def train(tool, st, out):
    args = st.args
    recorder = Recorder(st, tool)
    if tool == 'cf':  # code generation ahead of time; setup then only loads from the cache
        from constraintflow.lib.training import compile_programs
        start = time.perf_counter()
        compile_programs(models.Models[args.model](), torch.zeros(2, 1, 28, 28), 'mnist', METHODS,
                         batch_sizes(args.batch_size), args.cf_cache_dir)
        recorder.result['compile_seconds'] = time.perf_counter() - start
    args.save_model = args.save_model or os.path.join(out, 'ckpt')
    st.main(args)
    with open(os.path.join(out, 'record.json'), 'w') as f:
        json.dump(recorder.result, f)


def lirpa_bounds(model, x, x_L, x_U, y, args):
    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm
    c = torch.eye(10).type_as(x)[y].unsqueeze(1) - torch.eye(10).type_as(x).unsqueeze(0)
    c = c[y.unsqueeze(1) != torch.arange(10).type_as(y).unsqueeze(0)].view(len(y), 9, 10)
    bounded = BoundedModule(model, torch.zeros(2, 1, 28, 28), bound_opts={'conv_mode': args.conv_mode}, device='cuda')
    ptb = PerturbationLpNorm(norm=float('inf'), eps=args.eps, x_L=x_L, x_U=x_U)
    for method in METHODS:
        bounded(BoundedTensor(x, ptb))
        lb, _ = bounded.compute_bounds(IBP=True, C=c, method=None)
        if method == 'CROWN-IBP':
            lb, _ = bounded.compute_bounds(IBP=False, C=c, method='backward', bound_upper=False)
        yield method, lb


def cf_bounds(model, x, x_L, x_U, y, args):
    from constraintflow.lib.training import TrainingBounds
    cf = TrainingBounds(model, torch.zeros(2, 1, 28, 28), 'mnist', METHODS, [len(y)], args.cf_cache_dir)
    for method in METHODS:
        cf.begin_step()
        yield method, cf.compute_bounds(method, x_L, x_U, y)[0]


def bounds(tool, st, net, ckpt, out):
    """Lower bounds and d(sum lb)/dθ for both methods on the first 256 test images."""
    model = models.Models[net]().cuda()
    model.load_state_dict(torch.load(ckpt)['state_dict'])
    data = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())
    x = torch.stack([data[i][0] for i in range(256)]).cuda()
    y = torch.tensor([data[i][1] for i in range(256)]).cuda()
    x_L, x_U = (x - st.args.eps).clamp(min=0), (x + st.args.eps).clamp(max=1)
    result = {}
    for method, lb in (lirpa_bounds if tool == 'lirpa' else cf_bounds)(model, x, x_L, x_U, y, st.args):
        model.zero_grad()
        lb.sum().backward()
        result[method] = (lb.detach().cpu(), {k: p.grad.cpu() for k, p in model.named_parameters()})
    torch.save(result, os.path.join(out, f'bounds_{os.path.basename(ckpt)}_{tool}.pt'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tool', choices=['lirpa', 'cf'], required=True)
    p.add_argument('--onnx', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--bounds_ckpt')
    p.add_argument('--gpu_data', action='store_true', help='GPULoader instead of DataLoader, for both tools')
    p.add_argument('rest', nargs=argparse.REMAINDER, help='simple_training.py args, after --')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # same precision in both

    net = os.path.splitext(os.path.basename(a.onnx))[0]
    st = load_script(a.tool, ['--model', net] + [r for r in a.rest if r != '--'])
    assert st.args.data == 'MNIST'
    models.Models[net] = lambda in_ch=1, in_dim=28: get_pytorch_from_onnx(a.onnx)
    if a.gpu_data:
        torch.utils.data.DataLoader = GPULoader
    if a.bounds_ckpt:
        bounds(a.tool, st, net, a.bounds_ckpt, a.out)
    else:
        train(a.tool, st, a.out)


if __name__ == '__main__':
    main()
