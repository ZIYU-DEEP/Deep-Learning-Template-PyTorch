"""
Title: loader_image.py
Description: A general customized data loader for
             image data, re-writing based on ImageFolder.

             You may customize the transform.
"""

from PIL import Image
from pathlib import Path
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import torchvision.transforms as transforms


# #########################################################################
# 1. Base Loader
# #########################################################################
class BaseDataset(ABC):
    def __init__(self, root: str):
        super().__init__()

        self.root = Path(root)
        self.train_set = None
        self.test_set = None
        self.val_set = None

    @abstractmethod
    def loaders(self,
                batch_size: int=128,
                shuffle=True,
                num_workers: int = 0):
        pass

    def __repr__(self):
        return self.__class__.__name__


# #########################################################################
# 0. Helper Classes and Functions
# #########################################################################
class ImageFolder(ImageFolder):
    """
    A dataset object for Retinal OCT.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)

        if self.transform is None:
            transform = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor()])
        else:
            transform = self.transform

        img = transform(img)
        return img, int(target), index


# #########################################################################
# 2. Loader for Retinal OCT Dataset
# #########################################################################
class ImageLoader(BaseDataset):
    def __init__(self,
                 root: str='../data/',
                 filename: str='OCT2017'):
        super().__init__(root)

        # Initialization
        self.path = self.root / filename

        # Set the transform type
        transform_train = transforms.Compose([transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor()])

        transform_test = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor()])


        # Read in full set
        self.train_set = ImageFolder(root=self.path / 'train',
                                     transform=transform_train)

        self.test_set = ImageFolder(root=self.path / 'test',
                                    transform=transform_test)

        self.val_set = ImageFolder(root=self.path / 'val',
                                   transform=transform_test)


    def loaders(self,
                batch_size: int=512,
                shuffle_train: bool=True,
                shuffle_test: bool=False,
                shuffle_val: bool=True,
                num_workers: int=0) -> (DataLoader, DataLoader, DataLoader):

        train_loader = DataLoader(dataset=self.train_set,
                                  batch_size=batch_size,
                                  shuffle=shuffle_train,
                                  num_workers=num_workers,
                                  drop_last=False)

        test_loader = DataLoader(dataset=self.test_set,
                                 batch_size=batch_size,
                                 shuffle=shuffle_test,
                                 num_workers=num_workers,
                                 drop_last=False)

        val_loader = DataLoader(dataset=self.val_set,
                                batch_size=batch_size,
                                shuffle=shuffle_val,
                                num_workers=num_workers,
                                drop_last=False)

        return train_loader, test_loader, val_loader
