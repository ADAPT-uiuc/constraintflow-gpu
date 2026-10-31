"""Write normalized CIFAR verification boxes without downloading a dataset."""
import argparse
import numpy as np
import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from constraintflow.lib.spec import ImageDataset

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--data-root', required=True)
p.add_argument('--output', required=True)
p.add_argument('--eps', type=float, default=0.00004)
p.add_argument('--images', type=int, default=11)
a = p.parse_args()
data = CIFAR10(a.data_root, train=False, download=False, transform=ToTensor())
x = torch.stack([data[i][0] for i in range(a.images)])
mean, std = ImageDataset._normalization(x, 'cifar')
lower = ((x - a.eps).clamp(0, 1) - mean) / std
upper = ((x + a.eps).clamp(0, 1) - mean) / std
np.savez(a.output, down=lower.flatten(1).numpy(), up=upper.flatten(1).numpy(),
         labels=np.array([data[i][1] for i in range(a.images)]))
