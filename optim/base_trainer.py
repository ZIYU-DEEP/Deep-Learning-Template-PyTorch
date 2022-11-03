"""
Title: base_trainer.py
Description: The base trainer and evaluater.
"""

from abc import ABC, abstractmethod
from pathlib import Path


# #########################################################################
# 1. Base Trainer
# #########################################################################
class BaseTrainer(ABC):
    def __init__(self,
                 optimizer_name: str,
                 momentum: float,
                 lr: float,
                 lr_schedule: str,
                 lr_milestones: str,
                 lr_gamma: float,
                 n_epochs: int,
                 batch_size: int,
                 weight_decay: float,
                 device: str,
                 n_jobs_dataloader: int,
                 reg_type: str,
                 reg_weight: float,
                 final_path: str):
        """
        The base trainer for the model.

        Notice that the last parameter prune only helps you to determine the way
        to extract gradients; it is not an indicator if pruning is done during training.
        """

        super().__init__()
        self.optimizer_name = optimizer_name
        self.momentum = momentum
        self.lr = lr
        self.n_epochs = n_epochs
        self.lr_schedule = lr_schedule
        self.lr_milestones = [int(i) for i in lr_milestones.strip().split('-')]
        self.lr_gamma = lr_gamma
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device
        self.n_jobs_dataloader = n_jobs_dataloader
        self.reg_type = reg_type
        self.reg_weight = reg_weight
        self.final_path = Path(final_path)

    @abstractmethod
    def train(self, dataset, net):
        pass

    @abstractmethod
    def test(self, dataset, net):
        pass
