"""Run auto_LiRPA's or cf-gpu's simple_training.py on an ONNX network, measured the same way for both.

    python compare_driver.py --tool {lirpa,cf} --onnx NET.onnx --out DIR [--gpu_data] [--normalize] -- <simple_training.py args>
    python compare_driver.py ... --bounds_ckpt CKPT -- <args>     # bounds and gradients on fixed weights
    python compare_driver.py ... --precision_ckpt CKPT -- <args>  # precision per method (cli.get_precision)
    python compare_driver.py ... --probe_memory 16,32,64 -- <args>  # peak memory of one bound + backward

Timed: setup (BoundedModule / TrainingBounds construction) and every Train() call.
Not timed: cf-gpu code generation (done ahead, into its cache) and warmup.
"""
import argparse
import glob
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
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import models  # auto_LiRPA's example model registry
from simple_verification import get_pytorch_from_onnx  # the verification sweeps' ONNX loader
from auto_LiRPA.eps_scheduler import FixedScheduler

#FEAT: CROWN (DeepPoly in cf-gpu) and zid (cf-gpu only) next to IBP and CROWN-IBP.
METHODS = {'lirpa': ['IBP', 'CROWN-IBP', 'CROWN'], 'cf': ['IBP', 'CROWN-IBP', 'CROWN', 'zid']}
#FEAT: CIFAR next to MNIST.
SIZES = {'MNIST': (60000, 10000), 'CIFAR': (50000, 10000)}  # train, test
SHAPES = {'MNIST': (1, 28, 28), 'CIFAR': (3, 32, 32)}
# The verification sweeps' input normalization (simple_training.py normalizes CIFAR itself, not MNIST).
NORMALIZATION = {'MNIST': ([0.1307], [0.3081]), 'CIFAR': ([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])}


def dataset(name, train):
    make = datasets.MNIST if name == 'MNIST' else datasets.CIFAR10
    return make('./data', train=train, download=True, transform=transforms.ToTensor())


def normalization(name):
    mean, std = NORMALIZATION[name]
    return torch.tensor(mean).view(1, -1, 1, 1), torch.tensor(std).view(1, -1, 1, 1)


class GPULoader:
    """DataLoader replacement with the dataset on the GPU; same batch order and default-RNG use as DataLoader."""

    def __init__(self, dataset, batch_size=1, shuffle=False, **ignored):
        self.dataset, self.batch_size, self.shuffle = dataset, batch_size, shuffle
        self.mnist, self.augment = isinstance(dataset, datasets.MNIST), False
        if self.mnist:
            assert isinstance(dataset.transform, transforms.ToTensor)
            self.x = (dataset.data.float() / 255).unsqueeze(1).cuda()  # divide on the CPU, like ToTensor
            self.y = dataset.targets.cuda()
            if NORMALIZE:
                mean, std = normalization('MNIST')
                self.x = (self.x - mean.cuda()) / std.cuda()
            return
        #FEAT: CIFAR on the GPU. The test set matches the CPU pipeline; the training set's flip + 4-px crop run
        # per sample on the GPU (same distribution, own generator seeded like the script) since workers' draws can't be replayed.
        assert isinstance(dataset, datasets.CIFAR10)
        steps = [type(t).__name__ for t in dataset.transform.transforms]
        self.augment = steps == ['RandomHorizontalFlip', 'RandomCrop', 'ToTensor', 'Normalize']
        assert self.augment or steps == ['ToTensor', 'Normalize'], steps
        mean, std = normalization('CIFAR')
        x = torch.from_numpy(dataset.data).permute(0, 3, 1, 2).float().div(255)  # ToTensor, on the CPU
        self.x = (x if self.augment else x.sub(mean).div(std)).cuda()
        self.y = torch.tensor(dataset.targets).cuda()
        self.mean_std = mean.cuda(), std.cuda()
        self.generator = torch.Generator(device='cuda')
        self.generator.manual_seed(torch.initial_seed())  # simple_training seeds torch with --seed before building loaders

    def __setattr__(self, name, value):  # simple_training sets mean/std as CPU tensors
        if NORMALIZE and getattr(self, 'mnist', False) and name in ('mean', 'std'):
            value = torch.tensor(NORMALIZATION['MNIST'][name == 'std'])
        super().__setattr__(name, value.cuda() if name in ('mean', 'std') else value)

    def __len__(self):
        return -(-len(self.y) // self.batch_size)

    def _augmented(self, x):
        """RandomHorizontalFlip, RandomCrop(32, 4) with zero padding, then Normalize; per sample."""
        n, (h, w), g = x.shape[0], x.shape[2:], self.generator
        flip = torch.rand(n, device=x.device, generator=g) < 0.5
        x = F.pad(torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x), (4, 4, 4, 4))
        top = torch.randint(0, 9, (n, 1), device=x.device, generator=g) + torch.arange(h, device=x.device)
        left = torch.randint(0, 9, (n, 1), device=x.device, generator=g) + torch.arange(w, device=x.device)
        x = x[torch.arange(n, device=x.device).view(-1, 1, 1, 1), torch.arange(x.shape[1], device=x.device).view(1, -1, 1, 1),
              top.view(n, 1, h, 1), left.view(n, 1, 1, w)]
        mean, std = self.mean_std
        return x.sub(mean).div(std)

    def __iter__(self):
        torch.empty((), dtype=torch.int64).random_()  # DataLoader draws this base seed every epoch
        n = len(self.y)
        order = torch.tensor(list(torch.utils.data.RandomSampler(range(n)))) if self.shuffle else torch.arange(n)
        for idx in order.cuda().split(self.batch_size):
            yield (self._augmented(self.x[idx]) if self.augment else self.x[idx]), self.y[idx]


#FEAT: --normalize applies the sweeps' MNIST normalization (the ONNX loader drops the nets' own Sub/Div).
NORMALIZE = False


class NormalizedLoader(torch.utils.data.DataLoader):
    """DataLoader that normalizes MNIST batches and reports the matching mean/std to simple_training."""

    def __setattr__(self, name, value):
        if name in ('mean', 'std'):
            value = torch.tensor(NORMALIZATION['MNIST'][name == 'std'])
        super().__setattr__(name, value)

    def __iter__(self):
        mean, std = normalization('MNIST')
        for x, y in super().__iter__():
            yield (x - mean) / std, y


class FixedBatches(list):
    """Warmup loader: the same batch, three times (CUDA graphs warm up, record, then replay)."""

    def __init__(self, batch_size, data):
        images = dataset(data, train=True)
        x = torch.stack([images[i][0] for i in range(batch_size)])
        y = torch.tensor([images[i][1] for i in range(batch_size)])
        self.mean, self.std = torch.zeros(SHAPES[data][0]), torch.ones(SHAPES[data][0])
        if NORMALIZE or data == 'CIFAR':
            mean, std = normalization(data)
            x, self.mean, self.std = (x - mean) / std, mean.flatten(), std.flatten()
        super().__init__([(x, y)] * 3)
        self.dataset, self.batch_size = [None] * (3 * batch_size), batch_size


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


#FIX FOR: the ONNX loader ends in Flatten after the last Linear, so auto_LiRPA could not fold the margin
# spec into it and its IBP was ~30% looser than cf-gpu's (which folds); the Flatten is a no-op on logits.
def no_trailing_flatten(model):
    if isinstance(model[-1], torch.nn.Flatten):
        model[-1] = torch.nn.Identity()
    return model


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


def batch_sizes(batch_size, data):
    return sorted({batch_size} | {n % batch_size for n in SIZES[data]} - {0})


def dummy_input(data):
    return torch.zeros(2, *SHAPES[data])


class Recorder:
    """Times setup and Train() calls in a loaded simple_training module and records losses and meters."""

    def __init__(self, st, tool):
        self.st, self.args = st, st.args
        self.result = {'tool': tool, 'args': vars(st.args), 'normalize': NORMALIZE, 'records': []}
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
        for batch_size in batch_sizes(self.args.batch_size, self.args.data):
            for train in (True, False):
                warm = self.train_signature.bind(*call.args, **call.kwargs)
                warm.arguments.update(loader=FixedBatches(batch_size, self.args.data),
                                      eps_scheduler=WarmupScheduler(self.args.eps),
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


def train(tool, st, out, ckpt_every=0):
    args = st.args
    recorder = Recorder(st, tool)
    if tool == 'cf':  # code generation ahead of time; setup then only loads from the cache
        from constraintflow.lib.training import compile_programs
        start = time.perf_counter()
        compile_programs(models.Models[args.model](), dummy_input(args.data), args.data.lower(),
                         st.TRAINING_METHODS[args.bound_type], batch_sizes(args.batch_size, args.data),
                         args.cf_cache_dir)
        recorder.result['compile_seconds'] = time.perf_counter() - start
    args.save_model = args.save_model or os.path.join(out, 'ckpt')
    #FEAT: --ckpt_every: keep a copy of every Nth epoch's checkpoint (saved by simple_training after its untimed test pass).
    if ckpt_every:
        os.makedirs(os.path.join(out, 'ckpts'), exist_ok=True)
        save = torch.save

        def save_and_keep(obj, path, *x, **k):
            save(obj, path, *x, **k)
            if path == args.save_model and obj['epoch'] % ckpt_every == 0:
                save(obj, os.path.join(out, 'ckpts', f"epoch_{obj['epoch']:03d}.pt"))
        torch.save = save_and_keep
    st.main(args)
    with open(os.path.join(out, 'record.json'), 'w') as f:
        json.dump(recorder.result, f)


# --------------------------------------------------------------------------
# Bounds on fixed weights: --bounds_ckpt, --precision_ckpt, --probe_memory
# --------------------------------------------------------------------------

def spec_matrix(y):
    c = torch.eye(10).type_as(y.float())[y].unsqueeze(1) - torch.eye(10, device=y.device).unsqueeze(0)
    return c[y.unsqueeze(1) != torch.arange(10, device=y.device).unsqueeze(0)].view(len(y), 9, 10)


def lirpa_bounder(model, method, args):
    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm
    bounded = BoundedModule(model, dummy_input(args.data), bound_opts={'conv_mode': args.conv_mode}, device='cuda')

    def bound(x, x_L, x_U, y):
        c = spec_matrix(y)
        bounded(BoundedTensor(x, PerturbationLpNorm(norm=float('inf'), eps=args.eps, x_L=x_L, x_U=x_U)))
        if method == 'CROWN':  # this module never ran IBP, so intermediate bounds come from CROWN too
            return bounded.compute_bounds(IBP=False, C=c, method='backward', bound_upper=False)[0]
        lb = bounded.compute_bounds(IBP=True, C=c, method=None)[0]
        if method == 'CROWN-IBP':
            lb = bounded.compute_bounds(IBP=False, C=c, method='backward', bound_upper=False)[0]
        return lb
    return bound


def cf_bounder(model, method, args, sizes):
    from constraintflow.lib.training import TrainingBounds
    cf = TrainingBounds(model, dummy_input(args.data), args.data.lower(), [method], sizes, args.cf_cache_dir)

    def bound(x, x_L, x_U, y):
        cf.begin_step()
        return cf.compute_bounds(method, x_L, x_U, y)[0]
    return bound


def bounder(tool, model, method, args, sizes):
    return lirpa_bounder(model, method, args) if tool == 'lirpa' else cf_bounder(model, method, args, sizes)


def load_model(net, ckpt):
    model = models.Models[net]().cuda()
    if ckpt:
        model.load_state_dict(torch.load(ckpt)['state_dict'])
    return model.eval()


def test_box(args, n):
    """The first n test images (normalized as for training) and their Linf box, as simple_training builds it."""
    images = dataset(args.data, train=False)
    x = torch.stack([images[i][0] for i in range(n)])
    y = torch.tensor([images[i][1] for i in range(n)])
    if NORMALIZE or args.data == 'CIFAR':
        mean, std = normalization(args.data)
    else:
        mean, std = torch.zeros(1, SHAPES[args.data][0], 1, 1), torch.ones(1, SHAPES[args.data][0], 1, 1)
    x = (x - mean) / std
    x_L = torch.max(x - args.eps / std, (0 - mean) / std)
    x_U = torch.min(x + args.eps / std, (1 - mean) / std)
    return x.cuda(), x_L.cuda(), x_U.cuda(), y.cuda()


def bounds(tool, st, net, ckpt, out, num_images=256):
    """Lower bounds and d(sum lb)/dθ for every method on the first num_images test images."""
    model = load_model(net, ckpt)
    x, x_L, x_U, y = test_box(st.args, num_images)
    result = {}
    for method in METHODS[tool]:
        lb = bounder(tool, model, method, st.args, [len(y)])(x, x_L, x_U, y)
        model.zero_grad()
        lb.sum().backward()
        result[method] = (lb.detach().cpu(), {k: p.grad.cpu() for k, p in model.named_parameters()})
    name = os.path.basename(ckpt) if ckpt else 'onnx'
    torch.save(result, os.path.join(out, f'bounds_{name}_{tool}.pt'))


#FEAT: --precision_ckpt
def precision(tool, st, net, ckpt, out, num_images, eval_batch_size):
    """cf-gpu's precision (cli.get_precision): the fraction of images whose margin lower bounds are all >= 0,
    plus clean accuracy. A folder of checkpoints gives one entry per checkpoint (precision_curve_<tool>.json)."""
    assert num_images % eval_batch_size == 0, '--num_images must be a multiple of --eval_batch_size'
    #FEAT: a folder evaluates every checkpoint in it; cf-gpu reuses its programs (weights are loaded in place).
    ckpts = sorted(glob.glob(os.path.join(ckpt, '*.pt'))) if ckpt and os.path.isdir(ckpt) else [ckpt]
    model = load_model(net, ckpts[0])
    x, x_L, x_U, y = test_box(st.args, num_images)
    chunks = [slice(i, i + eval_batch_size) for i in range(0, num_images, eval_batch_size)]
    results, bounders = [], {}
    for path in ckpts:
        if path:
            state = torch.load(path)
            model.load_state_dict(state['state_dict'])
        result = {'tool': tool, 'ckpt': path, 'epoch': state['epoch'] if path else 0, 'eps': st.args.eps,
                  'normalize': NORMALIZE, 'num_images': num_images, 'methods': {}}
        with torch.no_grad():
            result['clean'] = torch.cat([model(x[s]).argmax(1) == y[s] for s in chunks]).float().mean().item()
        for method in METHODS[tool]:
            if tool == 'lirpa' or method not in bounders:  # BoundedModule keeps its own copy of the weights
                bounders[method] = bounder(tool, model, method, st.args, [eval_batch_size])
            with torch.no_grad():
                verified = torch.cat([(bounders[method](x[s], x_L[s], x_U[s], y[s]) >= 0).all(1) for s in chunks])
            result['methods'][method] = {'verified': int(verified.sum()), 'precision': verified.float().mean().item()}
        print(f"epoch {result['epoch']}: clean {result['clean']:.4f}",
              {m: r['precision'] for m, r in result['methods'].items()}, flush=True)
        results.append(result)
    if len(ckpts) > 1 or (ckpt and os.path.isdir(ckpt)):
        with open(os.path.join(out, f'precision_curve_{tool}.json'), 'w') as f:
            json.dump(results, f, indent=2)
    else:
        name = os.path.splitext(os.path.basename(ckpt))[0] if ckpt else 'onnx'
        with open(os.path.join(out, f'precision_{name}_{tool}.json'), 'w') as f:
            json.dump(results[0], f, indent=2)


#FEAT: --probe_memory
def probe_memory(tool, st, net, sizes, out):
    """Per method and batch size: peak GPU memory (GB) of one bound computation and its backward, the
    seconds of one such step after a warmup step, and the epoch estimate (train size / batch size steps)."""
    model = load_model(net, None).train()
    x, x_L, x_U, y = test_box(st.args, max(sizes))
    result = {}
    for method in METHODS[tool]:
        result[method] = {}
        for n in sizes:
            torch.cuda.empty_cache()
            def step():  # zero_grad as in training: CUDA-graph outputs must not be accumulated into
                model.zero_grad(set_to_none=True)
                bound(x[:n], x_L[:n], x_U[:n], y[:n]).sum().backward()
            try:
                bound = bounder(tool, model, method, st.args, [n])
                step(); step()  # warmup: cf-gpu compiles, then records its CUDA graph
                torch.cuda.reset_peak_memory_stats()
                seconds = synced_seconds(step)
                steps = -(-SIZES[st.args.data][0] // n)
                result[method][n] = {'peak_gb': round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
                                     'step_s': round(seconds, 4), 'epoch_s': round(seconds * steps, 1)}
            except torch.cuda.OutOfMemoryError:
                result[method][n] = 'OOM'
            except Exception as e:  # e.g. a cf-gpu build that fails
                result[method][n] = f'{type(e).__name__}: {str(e)[:200]}'
            model.zero_grad(set_to_none=True)
            print(method, n, result[method][n], flush=True)
    with open(os.path.join(out, f'memory_{tool}.json'), 'w') as f:
        json.dump(result, f, indent=2)


def main():
    global NORMALIZE
    p = argparse.ArgumentParser()
    p.add_argument('--tool', choices=['lirpa', 'cf'], required=True)
    p.add_argument('--onnx', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--bounds_ckpt', help="checkpoint to compare bounds on; 'onnx' for the ONNX file's own weights")
    p.add_argument('--methods', help='comma-separated subset of the bound methods (default: all the tool has)')
    p.add_argument('--precision_ckpt', help="checkpoint (or folder of checkpoints) to evaluate; 'onnx' for the ONNX file's own weights")
    p.add_argument('--ckpt_every', type=int, default=0, help='training: also keep every Nth epoch checkpoint in OUT/ckpts')
    p.add_argument('--probe_memory', help='comma-separated batch sizes')
    p.add_argument('--num_images', type=int, default=1000)
    p.add_argument('--eval_batch_size', type=int, default=100)
    p.add_argument('--gpu_data', action='store_true', help='GPULoader instead of DataLoader, for both tools (MNIST, CIFAR)')
    p.add_argument('--normalize', action='store_true', help="the sweeps' MNIST normalization, for both tools")
    p.add_argument('rest', nargs=argparse.REMAINDER, help='simple_training.py args, after --')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # same precision in both
    NORMALIZE = a.normalize
    if a.methods:
        METHODS[a.tool] = [m for m in a.methods.split(',') if m in METHODS[a.tool]]

    net = os.path.splitext(os.path.basename(a.onnx))[0]
    st = load_script(a.tool, ['--model', net] + [r for r in a.rest if r != '--'])
    models.Models[net] = lambda in_ch=1, in_dim=28: no_trailing_flatten(get_pytorch_from_onnx(a.onnx))
    if a.gpu_data:
        torch.utils.data.DataLoader = GPULoader
    elif NORMALIZE and st.args.data == 'MNIST':
        torch.utils.data.DataLoader = NormalizedLoader
    if a.bounds_ckpt:
        ckpt = None if a.bounds_ckpt == 'onnx' else a.bounds_ckpt
        bounds(a.tool, st, net, ckpt, a.out, a.num_images if a.num_images != 1000 else 256)
    elif a.precision_ckpt:
        ckpt = None if a.precision_ckpt == 'onnx' else a.precision_ckpt
        precision(a.tool, st, net, ckpt, a.out, a.num_images, a.eval_batch_size)
    elif a.probe_memory:
        probe_memory(a.tool, st, net, [int(n) for n in a.probe_memory.split(',')], a.out)
    else:
        train(a.tool, st, a.out, a.ckpt_every)


if __name__ == '__main__':
    main()
