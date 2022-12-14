"""
Title: base_net.py
Description: The base network.
Reference: https://github.com/lukasruff/Deep-SAD-PyTorch/tree/master/src/networks
"""

import logging
import numpy as np
import torch.nn as nn


# #########################################################################
# 1. Base Net
# #########################################################################
class BaseNet(nn.Module):
    """
    Base class for all neural networks.
    """

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self, *input):
        raise NotImplementedError

    def summary(self):
        net_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in net_parameters])
        self.logger.info('Trainable parameters: {}'.format(params))
        self.logger.info(self)
