import os
import datasets
import torchvision
from torch.utils.data import DataLoader

def download(dataset="MNIST", path="./data/MNIST", train=False):
    # Download dataset (MNIST, CIFAR10, CIFAR100) or construct it (ImageNet)
    arbitrary_dset = getattr(torchvision.datasets, dataset)

    print('Downloading ... {}'.format(dataset))
    dataset_obj = arbitrary_dset(path, train=train, download=True)
    data_loader = DataLoader(dataset_obj,
         shuffle=True, pin_memory=False, num_workers=1, batch_size=1)
    split = "train" if train else "val"
    print('# of {} images: {}\n'.format(split, len(iter(dl))))

def test_imagenet(path, train=False):
    os.environ['DATAPATH'] = path 
    imagenet = datasets.ImageNet(train=train)
    data_loader = DataLoader(imagenet,
         shuffle=True, pin_memory=False, num_workers=1, batch_size=1)

    n=1
    # n=len(data_iters)
    data_iters = iter(data_loader)
    for i in range(n):
        batch, label = next(data_iters)
        print(batch)

download('MNIST', './data/MNIST', train=True)
download('CIFAR10', './data/CIFAR10', train=True)
download('CIFAR100', './data/CIFAR100', train=True)
# download('ImageNet', './data/ImageNet', train=True)

# test_imagenet('/home/younghwan/workspace/shrinkbench/data', train=True)
