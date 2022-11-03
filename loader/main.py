"""
Title: main_loading.py
Description: The all-in-one loading functions.
"""
from .loader_toy import ToyLoader
from .loader_mnist import MNISTLoader
from .loader_fmnist import FashionMNISTLoader
from .loader_cifar10 import CIFAR10Loader
from .loader_cifar100 import CIFAR100Loader
from .loader_tiny_imagenet import TinyImageNetLoader


def load_dataset(loader_name: str='toy',
                 root: str='./data/',
                 filename: str='toy',
                 test_size: float=0.2,
                 random_state: int=42,
                 download: int=0):

    if loader_name == 'toy':
        return ToyLoader(root,
                         filename,
                         test_size,
                         random_state)

    if loader_name == 'mnist':
        return MNISTLoader(root,
                           filename,
                           random_state)

    if loader_name == 'fmnist':
        return FashionMNISTLoader(root,
                                 filename,
                                 random_state)

    if loader_name == 'cifar10':
        return CIFAR10Loader(root,
                             filename,
                             random_state)

    if loader_name == 'cifar100':
        return CIFAR100Loader(root,
                              filename,
                              random_state)

    if loader_name == 'tiny_imagenet':
        return TinyImageNetLoader(root,
                                  filename,
                                  random_state,
                                  download)

    return None
