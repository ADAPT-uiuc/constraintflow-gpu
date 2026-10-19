"""
A simple script to train certified defense using the auto_LiRPA library.

We compute output bounds under input perturbations using auto_LiRPA, and use
them to form a "robust loss" for certified defense.  Several different bound
options are supported, such as IBP, CROWN, and CROWN-IBP. This is a basic
example on MNIST and CIFAR-10 datasets with Lp (p>=0) norm perturbation. For
faster training, please see our examples with loss fusion such as
cifar_training.py and tinyimagenet_training.py

CF-GPU: auto_LiRPA 0.7.2's examples/vision/simple_training.py with bounds from constraintflow-gpu;
changes are marked "CF-GPU". Run with auto_LiRPA and auto_LiRPA/examples/vision on PYTHONPATH.
"""

import time
import random
import multiprocessing
import argparse
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from constraintflow.lib.training import TrainingBounds  # CF-GPU: replaces BoundedModule
from auto_LiRPA.perturbations import *
from auto_LiRPA.utils import MultiAverageMeter
from auto_LiRPA.eps_scheduler import LinearScheduler, AdaptiveScheduler, SmoothedScheduler, FixedScheduler
import models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()

parser.add_argument("--verify", action="store_true", help='verification mode, do not train')
parser.add_argument("--load", type=str, default="", help='Load pretrained model')
parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help='use cpu or cuda')
parser.add_argument("--data", type=str, default="MNIST", choices=["MNIST", "CIFAR"], help='dataset')
parser.add_argument("--seed", type=int, default=100, help='random seed')
parser.add_argument("--eps", type=float, default=0.3, help='Target training epsilon')
parser.add_argument("--norm", type=float, default='inf', help='p norm for epsilon perturbation')
parser.add_argument("--bound_type", type=str, default="CROWN-IBP",
                    choices=["IBP", "CROWN-IBP", "CROWN", "CROWN-FAST", "zid"], help='method of bound analysis')  # CF-GPU: + zid
parser.add_argument("--model", type=str, default="resnet", help='model name (mlp_3layer, cnn_4layer, cnn_6layer, cnn_7layer, resnet)')
parser.add_argument("--num_epochs", type=int, default=100, help='number of total epochs')
parser.add_argument("--batch_size", type=int, default=256, help='batch size')
parser.add_argument("--lr", type=float, default=5e-4, help='learning rate')
parser.add_argument("--scheduler_name", type=str, default="SmoothedScheduler",
                    choices=["LinearScheduler", "AdaptiveScheduler", "SmoothedScheduler", "FixedScheduler"], help='epsilon scheduler')
parser.add_argument("--scheduler_opts", type=str, default="start=3,length=60", help='options for epsilon scheduler')
parser.add_argument("--bound_opts", type=str, default=None, choices=["same-slope", "zero-lb", "one-lb"],
                    help='bound options')
parser.add_argument("--conv_mode", type=str, choices=["matrix", "patches"], default="patches")
parser.add_argument("--save_model", type=str, default='')
parser.add_argument("--cf_cache_dir", type=str, default='cf_training_cache')  # CF-GPU: compiled programs

args = parser.parse_args()
# CF-GPU: the cf-gpu programs each bound type needs (CROWN = DeepPoly; zid has no auto_LiRPA counterpart)
TRAINING_METHODS = {'IBP': ['IBP'], 'CROWN-IBP': ['IBP', 'CROWN-IBP'], 'CROWN-FAST': ['CROWN-IBP'],
                    'CROWN': ['CROWN'], 'zid': ['zid']}


def Train(model, cf, t, loader, eps_scheduler, norm, train, opt, bound_type, method='robust'):  # CF-GPU: cf
    num_class = 10
    meter = MultiAverageMeter()
    if train:
        model.train()
        eps_scheduler.train()
        eps_scheduler.step_epoch()
        eps_scheduler.set_epoch_length(int((len(loader.dataset) + loader.batch_size - 1) / loader.batch_size))
    else:
        model.eval()
        eps_scheduler.eval()

    for i, (data, labels) in enumerate(loader):
        start = time.time()
        cf.begin_step()  # CF-GPU
        eps_scheduler.step_batch()
        eps = eps_scheduler.get_eps()
        # For small eps just use natural training, no need to compute LiRPA bounds
        batch_method = method
        if eps < 1e-20:
            batch_method = "natural"
        if train:
            opt.zero_grad()
        # generate specifications
        c = torch.eye(num_class).type_as(data)[labels].unsqueeze(1) - torch.eye(num_class).type_as(data).unsqueeze(0)
        # remove specifications to self
        I = (~(labels.data.unsqueeze(1) == torch.arange(num_class).type_as(labels.data).unsqueeze(0)))
        c = (c[I].view(data.size(0), num_class - 1, num_class))
        # bound input for Linf norm used only
        if norm == np.inf:
            data_max = torch.reshape((1. - loader.mean) / loader.std, (1, -1, 1, 1))
            data_min = torch.reshape((0. - loader.mean) / loader.std, (1, -1, 1, 1))
            data_ub = torch.min(data + (eps / loader.std).view(1,-1,1,1), data_max)
            data_lb = torch.max(data - (eps / loader.std).view(1,-1,1,1), data_min)
        else:
            data_ub = data_lb = data

        if list(model.parameters())[0].is_cuda:
            data, labels, c = data.cuda(), labels.cuda(), c.cuda()
            data_lb, data_ub = data_lb.cuda(), data_ub.cuda()

        # Specify Lp norm perturbation.
        # When using Linf perturbation, we manually set element-wise bound x_L and x_U. eps is not used for Linf norm.
        if norm > 0:
            ptb = PerturbationLpNorm(norm=norm, eps=eps, x_L=data_lb, x_U=data_ub)
        elif norm == 0:
            ptb = PerturbationL0Norm(eps = eps_scheduler.get_max_eps(), ratio = eps_scheduler.get_eps()/eps_scheduler.get_max_eps())
        x = data  # CF-GPU: no BoundedTensor; cf-gpu takes the box (data_lb, data_ub)

        output = model(x)
        regular_ce = CrossEntropyLoss()(output, labels)  # regular CrossEntropyLoss used for warming up
        meter.update('CE', regular_ce.item(), x.size(0))
        meter.update('Err', torch.sum(torch.argmax(output, dim=1) != labels).cpu().detach().numpy() / x.size(0), x.size(0))

        if batch_method == "robust":
            if bound_type == "IBP":
                lb, ub = cf.compute_bounds('IBP', data_lb, data_ub, labels)  # CF-GPU
            elif bound_type in ("CROWN", "zid"):
                lb, ub = cf.compute_bounds(bound_type, data_lb, data_ub, labels)  # CF-GPU
            elif bound_type == "CROWN-IBP":
                # lb, ub = model.compute_bounds(ptb=ptb, IBP=True, x=data, C=c, method="backward")  # pure IBP bound
                # we use a mixed IBP and CROWN-IBP bounds, leading to better performance (Zhang et al., ICLR 2020)
                factor = (eps_scheduler.get_max_eps() - eps) / eps_scheduler.get_max_eps()
                ilb, iub = cf.compute_bounds('IBP', data_lb, data_ub, labels)  # CF-GPU
                if factor < 1e-5:
                    lb = ilb
                else:
                    clb, cub = cf.compute_bounds('CROWN-IBP', data_lb, data_ub, labels)  # CF-GPU
                    lb = clb * factor + ilb * (1 - factor)
            elif bound_type == "CROWN-FAST":
                # Similar to CROWN-IBP but no mix between IBP and CROWN bounds.
                lb, ub = cf.compute_bounds('CROWN-IBP', data_lb, data_ub, labels)  # CF-GPU


            # Pad zero at the beginning for each example, and use fake label "0" for all examples
            lb_padded = torch.cat((torch.zeros(size=(lb.size(0),1), dtype=lb.dtype, device=lb.device), lb), dim=1)
            fake_labels = torch.zeros(size=(lb.size(0),), dtype=torch.int64, device=lb.device)
            robust_ce = CrossEntropyLoss()(-lb_padded, fake_labels)
        if batch_method == "robust":
            loss = robust_ce
        elif batch_method == "natural":
            loss = regular_ce
        if train:
            loss.backward()
            eps_scheduler.update_loss(loss.item() - regular_ce.item())
            opt.step()
        meter.update('Loss', loss.item(), data.size(0))
        if batch_method != "natural":
            meter.update('Robust_CE', robust_ce.item(), data.size(0))
            # For an example, if lower bounds of margins is >0 for all classes, the output is verifiably correct.
            # If any margin is < 0 this example is counted as an error
            meter.update('Verified_Err', torch.sum((lb < 0).any(dim=1)).item() / data.size(0), data.size(0))
        meter.update('Time', time.time() - start)
        if i % 50 == 0 and train:
            print('[{:2d}:{:4d}]: eps={:.8f} {}'.format(t, i, eps, meter))
    print('[{:2d}:{:4d}]: eps={:.8f} {}'.format(t, i, eps, meter))

def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    ## Step 1: Initial original model as usual, see model details in models/example_feedforward.py and models/example_resnet.py
    if args.data == 'MNIST':
        model_ori = models.Models[args.model](in_ch=1, in_dim=28)
    else:
        model_ori = models.Models[args.model](in_ch=3, in_dim=32)
    if args.load:
        state_dict = torch.load(args.load)['state_dict']
        model_ori.load_state_dict(state_dict)

    ## Step 2: Prepare dataset as usual
    if args.data == 'MNIST':
        dummy_input = torch.randn(2, 1, 28, 28)
        train_data = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
        test_data = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
    elif args.data == 'CIFAR':
        dummy_input = torch.randn(2, 3, 32, 32)
        normalize = transforms.Normalize(mean = [0.4914, 0.4822, 0.4465], std = [0.2023, 0.1994, 0.2010])
        train_data = datasets.CIFAR10("./data", train=True, download=True,
                transform=transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize]))
        test_data = datasets.CIFAR10("./data", train=False, download=True, 
                transform=transforms.Compose([transforms.ToTensor(), normalize]))

    train_data = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=min(multiprocessing.cpu_count(),4))
    test_data = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, pin_memory=True, num_workers=min(multiprocessing.cpu_count(),4))
    if args.data == 'MNIST':
        train_data.mean = test_data.mean = torch.tensor([0.0])
        train_data.std = test_data.std = torch.tensor([1.0])
    elif args.data == 'CIFAR':
        train_data.mean = test_data.mean = torch.tensor([0.4914, 0.4822, 0.4465])
        train_data.std = test_data.std = torch.tensor([0.2023, 0.1994, 0.2010])

    ## Step 3: wrap model with auto_LiRPA
    # The second parameter dummy_input is for constructing the trace of the computational graph.
    # CF-GPU: plain model plus cf-gpu bounds (default ReLU relaxation, Linf); one program per method and batch size.
    assert args.bound_opts is None and args.norm == np.inf
    model = model_ori.to(args.device)
    batch_sizes = {args.batch_size} | {len(d.dataset) % args.batch_size for d in (train_data, test_data)} - {0}
    cf = TrainingBounds(model, dummy_input, args.data.lower(), TRAINING_METHODS[args.bound_type], batch_sizes,
                        args.cf_cache_dir)  # CF-GPU

    ## Step 4 prepare optimizer, epsilon scheduler and learning rate scheduler
    opt = optim.Adam(model.parameters(), lr=args.lr)
    norm = float(args.norm)
    lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    eps_scheduler = eval(args.scheduler_name)(args.eps, args.scheduler_opts)
    print("Model structure: \n", str(model_ori))

    ## Step 5: start training
    if args.verify:
        eps_scheduler = FixedScheduler(args.eps)
        with torch.no_grad():
            Train(model, cf, 1, test_data, eps_scheduler, norm, False, None, args.bound_type)
    else:
        timer = 0.0
        for t in range(1, args.num_epochs+1):
            if eps_scheduler.reached_max_eps():
                # Only decay learning rate after reaching the maximum eps
                lr_scheduler.step()
            print("Epoch {}, learning rate {}".format(t, lr_scheduler.get_lr()))
            start_time = time.time()
            Train(model, cf, t, train_data, eps_scheduler, norm, True, opt, args.bound_type)
            epoch_time = time.time() - start_time
            timer += epoch_time
            print('Epoch time: {:.4f}, Total time: {:.4f}'.format(epoch_time, timer))
            print("Evaluating...")
            with torch.no_grad():
                Train(model, cf, t, test_data, eps_scheduler, norm, False, None, args.bound_type)
            torch.save({'state_dict': model_ori.state_dict(), 'epoch': t}, args.save_model if args.save_model != "" else args.model)


if __name__ == "__main__":
    main(args)
