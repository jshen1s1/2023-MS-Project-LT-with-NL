import numpy as np 
import torchvision.transforms as transforms
from bias_cifar import CIFAR10_bias, CIFAR100_bias
from torchvision.datasets import CIFAR10,CIFAR100


train_cifar10_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_cifar10_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
train_cifar100_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

test_cifar100_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])
def input_dataset(dataset, noise_type, noise_ratio,lt_type,lt_ratio,random_state):
    if dataset == 'cifar10':
        train_dataset = CIFAR10_bias(root='./data/',
                                download=False,  
                                train=True, 
                                transform = train_cifar10_transform,
                                noise_type=noise_type,
                                noise_rate=noise_ratio,
                                lt_type = lt_type,
                                lt_ratio = lt_ratio,
                                random_state=random_state
                           )
        test_dataset = CIFAR10(root='data', train=False, transform=test_cifar10_transform, download=True)
    elif dataset == 'cifar100':
        train_dataset = CIFAR100_bias(root='./data/',
                                download=False,  
                                train=True, 
                                transform = train_cifar100_transform,
                                noise_type=noise_type,
                                noise_rate=noise_ratio,
                                lt_type = lt_type,
                                lt_ratio = lt_ratio,
                                random_state=random_state
                           )
        test_dataset = CIFAR100(root='data', train=False, transform=test_cifar100_transform, download=True)
    return train_dataset, test_dataset


