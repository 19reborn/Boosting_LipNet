import os
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import re
import socket
import src.CImageNetRanges
import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST
# import pickle
# from collections.abc import Iterable


IMAGENET_DIR = "./data"

mean = {
    'MNIST': np.array([0.1307]),
    'FashionMNIST': np.array([0.2860]),
    'CIFAR10': np.array([0.4914, 0.4822, 0.4465]),
}
std = {
    'MNIST': 0.3081,
    'FashionMNIST': 0.3520,
    'CIFAR10': 0.2009, #np.array([0.2023, 0.1994, 0.2010])
}
train_transforms = {
    'MNIST': [transforms.RandomCrop(28, padding=1)],
    'FashionMNIST': [transforms.RandomCrop(28, padding=1)],
    'CIFAR10': [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()],
}
input_dim = {
    'MNIST': np.array([1, 28, 28]),
    'FashionMNIST': np.array([1, 28, 28]),
    'CIFAR10': np.array([3, 32, 32]),
}
default_eps = {
    'MNIST': 0.3,
    'FashionMNIST': 0.1,
    'CIFAR10': 0.03137,
}

def get_mean_sigma(device, dataset):
    if dataset == 'cifar10':
        mean = torch.FloatTensor([0.4914, 0.4822, 0.4465]).view((3, 1, 1))
        sigma = torch.FloatTensor([0.2023, 0.1994, 0.2010]).view((3, 1, 1))
    elif re.match("[r,c]{0,1}imagenet([0-9]*)", dataset) is not None:
        if re.match("[r,c]{1}imagenet([0-9]*)", dataset) is not None:
            ds = robustness.datasets.RestrictedImageNet(os.path.abspath(IMAGENET_DIR))
        else:
            ds = robustness.datasets.ImageNet(os.path.abspath(IMAGENET_DIR))
        mean = ds.mean.view((3, 1, 1))
        sigma = ds.std.view((3, 1, 1))
    elif re.match("timagenet", dataset) is not None:
        mean = torch.FloatTensor([0.4802, 0.4481, 0.3975]).view((3, 1, 1))
        sigma = torch.FloatTensor([0.2302, 0.2265, 0.2262]).view((3, 1, 1))
    elif dataset == "mnist":
        mean = torch.FloatTensor([0.1307]).view((1, 1, 1))
        sigma = torch.FloatTensor([0.3081]).view((1, 1, 1))
    else:
        raise RuntimeError(f"Dataset {dataset:s} is unknown.")
    return mean.to(device), sigma.to(device)

def get_statistics(dataset):
    return mean[dataset], std[dataset]


def get_mnist():
    return 28, 1, 10


def get_cifar10(debug=False):
    return 32, 3, 10




def get_tinyimagenet():
    return 56, 3, 200


def get_dataset(dataset, datadir, augmentation=True, classes=None):
    default_transform = [
        transforms.ToTensor(),
        transforms.Normalize(mean[dataset], [std[dataset]] * len(mean[dataset]))
    ]
    train_transform = transforms.Compose((train_transforms[dataset] if augmentation else []) + default_transform)
    test_transform = transforms.Compose(default_transform)
    Dataset = globals()[dataset]
    train_dataset = Dataset(root=datadir, train=True, download=True, transform=train_transform)
    test_dataset = Dataset(root=datadir, train=False, download=True, transform=test_transform)
    if classes is not None:
        train_dataset = TruncateDataset(train_dataset, classes)
        test_dataset = TruncateDataset(test_dataset, classes)
    return train_dataset, test_dataset

def get_loaders(args):

    if args.dataset == 'cifar10':
        input_size, input_channels, n_class = get_cifar10()
    # elif args.dataset == 'cifar10_MILP_cert':
    #     train_set, test_set, input_size, input_channels, n_class = get_cifar_MILP_cert()
    elif args.dataset == 'mnist':
        input_size, input_channels, n_class = get_mnist()
    elif args.dataset == 'timagenet':
        input_size, input_channels, n_class = get_tinyimagenet()
    else:
        raise NotImplementedError('Unknown dataset')
    #args.input_min, args.input_max = 0, 1

    train_sampler = None
    train_set, test_set = get_dataset(args.dataset.upper(), './data', augmentation=True, classes=None)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.train_batch, shuffle=True,
                                               num_workers=args.num_workers,pin_memory=True, sampler=train_sampler, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch, shuffle=False,
                                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    return len(train_set), train_loader, test_loader, input_size, input_channels, n_class
