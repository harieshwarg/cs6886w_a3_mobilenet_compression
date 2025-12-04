
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

MEAN=(0.4914,0.4822,0.4465)
STD =(0.2470,0.2435,0.2616)

def get_cifar10_loaders(batch_size=128, val_ratio=0.1, seed=42):
    train_tfms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    test_tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_full = torchvision.datasets.CIFAR10("./data", train=True,  download=True, transform=train_tfms)
    testset    = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=test_tfms)

    val_sz = int(len(train_full)*val_ratio)
    train_sz = len(train_full) - val_sz

    trainset, valset = random_split(train_full, [train_sz, val_sz],
                                    generator=torch.Generator().manual_seed(seed))

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    valloader   = DataLoader(valset,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    testloader  = DataLoader(testset,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return trainloader, valloader, testloader
