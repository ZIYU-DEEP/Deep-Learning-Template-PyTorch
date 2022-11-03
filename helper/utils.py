"""
[Title] utils.py
[Use] A general helper file.
[TOC] 1. General helper functions;
      2. Helpers for networks;
      3. Helpers for optimizers;
      4. Calculating SNR;
      5. Calculating Fisher information.
"""

from network.main import build_network
from collections import OrderedDict
from .pruner import global_prune
import torch.nn.functional as F
from functools import reduce
from torch import nn
import torch
import time
import psutil
import os

# ----------- Handle the TPU Training Framework ----------- #
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
except:
    pass

try:
    from functorch.experimental import replace_all_batch_norm_modules_
    from functorch import make_functional_with_buffers, vmap, grad
except:
    pass
# --------------------------------------------------------- #


# #########################################################################
# 1. General Helper Functions
# #########################################################################
def str_to_list(arg: str = '10-7-5-4-3'):
    '''
    Turn a string of numbers into a list.

    Inputs:
      arg: (str) must be of the form like "1-2-3-4".

    Returns:
      (list) example: [1, 2, 3, 4].
    '''

    return [int(i) for i in arg.strip().split('-')]


# #########################################################################
# 2. Helper for network and model training
# #########################################################################
def act_dict(act_name: str = 'tanh'):
    """
    Get a nn activation layer with a string input.
    This will later be used in files in ../.network/*.py.
    """
    assert act_name in ['relu', 'softmax', 'tanh', 'sigmoid', 'identity']

    dict_ = {'relu': nn.ReLU(),
             'softmax': nn.Softmax(),
             'tanh': nn.Tanh(),
             'sigmoid': nn.Sigmoid(),
             'identity': nn.Identity()}

    return dict_[act_name]


def loss_dict(loss_name: str = 'ce'):
    dict_ = {'ce': nn.CrossEntropyLoss()}

    return dict_[loss_name]


def get_module_by_name(net, access_string='cnn_model.0'):
    """
    Inputs:
        net: (nn.Module)
        access_string: (str) the name of the module
    Returns:
        (torch.nn.modules.conv/linear/...)
    """
    names = access_string.split(sep='.')
    return reduce(getattr, names, net)


# =========================================================================
# Helper for initialization
# =========================================================================
def init_weights_xavier(net):
    """
    Initialize with Xavier's method.
    """
    for module in net.named_modules():
        if isinstance(module[1], (nn.Linear, nn.Conv2d)):
            # Initialize for weights
            try:
                torch.nn.init.xavier_normal_(module[1].weight_orig)
            except:
                torch.nn.init.xavier_normal_(module[1].weight)

            # Initialize for bias
            if module[1].bias is not None:
                module[1].bias.data.fill_(0.01)


def init_weights_kaiming(net):
    """
    Initialize with Kaiming's method.
    """

    # ---------- Last layer: normal init to keep variance --------- #
    # This is actually foolish
    # Useless when the last linear layer is followed by a softmax layer
    module_last = [module for module in net.modules()][-1]

    if isinstance(module_last, nn.Linear):
        # Initialize for bias
        if module_last.bias is not None:
            nn.init.constant_(module_last.bias, 0)

        # Initialize for weights
        try:
            nn.init.normal_(module_last.weight_orig, 0, 0.01)
        except:
            nn.init.normal_(module_last.weight, 0, 0.01)

    # ------------- All other layers: use Kaiming init ------------ #
    module_list = [module for module in net.modules()][:-1]

    for module in module_list:
        # [Linear and Conv2d]: use Kaiming init
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Initialize for bias
            if module.bias is not None:
                module.bias.data.zero_()

            # Initialize for weights
            try:
                nn.init.kaiming_normal_(module.weight_orig,
                                        mode='fan_out',
                                        nonlinearity='relu')
            except:
                nn.init.kaiming_normal_(module.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')

        # [BatchNorm]: directl fill with 1 and 0
        elif isinstance(module, nn.BatchNorm2d):
            # Initialize for bias
            if module.bias is not None:
                module.bias.data.zero_()

            # Initialize for weights
            try:
                if module.weight_orig is not None:
                    module.weight_orig.data.fill_(1.0)
            except:
                if module.weight is not None:
                    module.weight.data.fill_(1.0)


# =========================================================================
# Get layer output in order to calculate MI per layer
# =========================================================================
def get_layer_output(module_name, layer_outputs_dict):
    """
    This is used to get the output from each layer.
    You should be expected to use the function as follows:
    ```
    layer_outputs_dict = {}
    for module_name, layer in net.named_modules():
        layer.register_forward_hook(get_layer_outputs(module_name,
                                                      layer_outputs_dict))
    ```
    """

    def hook(model, inputs, outputs):
        layer_outputs_dict[module_name] = outputs.detach()

    return hook


# =========================================================================
# One-line set up for network in experiment-fisher.py
# =========================================================================
def setup_network(net_name, in_dim, out_dim, hidden_act,
                  out_act, hidden_dims, depth, widen_factor,
                  dropRate, growthRate, compressionRate,
                  mode, prune_method, prune_ratio, device,
                  prune_last=False, parallel=True):
    """
    This function will be used in experiment.py to shorten code.
    """
    # Build up the network
    net = build_network(net_name, in_dim, out_dim, hidden_act,
                        out_act, hidden_dims, depth, widen_factor,
                        dropRate, growthRate, compressionRate)

    # Set the network in case it is a pruned one (for convenience of loading)
    if mode in ['sparse-finetune', 'sparse-scratch', 'lottery']:
        # This will simply add weight_orig and weight_mask to state_dict
        global_prune(net, prune_method, prune_ratio, prune_last)

    # Make assertion on the device
    assert device in ['cuda', 'cpu'], 'TPU is not supported in experiments.'

    # Load the network to the device
    if parallel:
        net = torch.nn.DataParallel(net).to(device)
    else:
        net = net.to(device)

    return net


# #########################################################################
# 3. Helper for optimizer
# #########################################################################
# =========================================================================
# Get accuracy in a batch
# =========================================================================
def get_acc(outputs, y, batch_size):
    """
    Get the accuracy for a batch of predictions.

    Inputs:
        outputs: (torch.tensor) the predicted label distributions
                 shape=(batch_size, n_classes)
        y: (torch.tensor) the true labels
           shape=(batch_size,)
        batch_size: (int) as the name suggested

    Return:
        The accuracy for this prediction.
    """
    acc = (outputs.argmax(dim=1) == y).type(torch.float).sum()
    acc /= batch_size
    return acc.detach()


# =========================================================================
# Get current memory
# =========================================================================
def get_memory():
    """
    Get current memory in GB.
    """
    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3


# =========================================================================
# Calculating Moving Averages
# =========================================================================
class Tracker():
    """
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.mean = 0
        self.sum = 0
        self.count = 0

    def update(self, val, batch_size=1):
        self.val = val
        self.sum += val * batch_size
        self.count += batch_size
        self.mean = self.sum / self.count


# =========================================================================
# Calculating Moving Averages
# =========================================================================
class Welford(object):
    """
    Implements Welford's algorithm for computing a running mean
    and standard deviation as described at:
        http://www.johndcook.com/standard_deviation.html
    can take single values or iterables
    Properties:
        mean    - returns the mean
        std     - returns the std
        meanfull- returns the mean and std of the mean
    Usage:
        >>> foo = Welford()
        >>> foo(range(100))
        >>> foo
        <Welford: 49.5 +- 29.0114919759>
        >>> foo([1]*1000)
        >>> foo
        <Welford: 5.40909090909 +- 16.4437417146>
        >>> foo.mean
        5.409090909090906
    """

    def __init__(self, lst=None):
        self.k = 0
        self.M = 0
        self.S = 0

        self.__call__(lst)

    def update(self, x):
        if x is None:
            return
        self.k += 1
        newM = self.M + (x - self.M) * 1. / self.k
        newS = self.S + (x - self.M) * (x - newM)
        self.M, self.S = newM, newS

    def consume(self, lst):
        lst = iter(lst)
        for x in lst:
            self.update(x)

    def __call__(self, x):
        if hasattr(x, "__iter__"):
            self.consume(x)
        else:
            self.update(x)

    @property
    def mean(self):
        return self.M

    @property
    def meanfull(self):
        return self.mean, self.std / torch.sqrt(self.k)

    @property
    def std(self):
        if self.k == 1:
            return 0
        return torch.sqrt(self.S / (self.k - 1))

    def __repr__(self):
        return "<Welford: {} +- {}>".format(self.mean, self.std)


class WelfordBs(object):
    """
    Batch-wise welford algorithm.
    """

    def __init__(self, lst=None):
        self.k = 0
        self.M = 0
        self.S = 0

        self.__call__(lst)

    def update(self, x):
        if x is None:
            return
        self.k += 1
        newM = self.M + (x - self.M) * 1. / self.k
        newS = self.S + (x - self.M) * (x - newM)
        self.M, self.S = newM, newS

    def __call__(self, x):
        self.update(x)

    @property
    def mean(self):
        return self.M

    @property
    def meanfull(self):
        return self.mean, self.std / torch.sqrt(self.k)

    @property
    def std(self):
        if self.k == 1:
            return 0
        return torch.sqrt(self.S / (self.k - 1))


# =========================================================================
# Save state dict
# =========================================================================
def load_state_dict_(net, state_dict):
    """
    A helper function to handle the loading issue.
    """
    try:
        net.load_state_dict(state_dict)
    except:
        try:
            net.module.load_state_dict(state_dict)
        except:
            for key in list(state_dict.keys()):
                state_dict[key.replace('module.', '')] = state_dict.pop(key)
            net.load_state_dict(state_dict)
    return net


def get_net_dict(net, device):
    """
    This unnecessary function shows you how unfriendly Torch is on TPU.
    """
    try:
        # Handle the case when nn.DataParallel() wraps a module
        net_dict = net.module.state_dict()
    except:
        net_dict = net.state_dict()

    return net_dict


def save_net_dict(net_dict,
                  epoch: int = 0,
                  is_last_epoch: bool = False,
                  path: str = 'final_path/state_dicts/epoch_0.pkl'):
    """
    Save essential information according to the epoch number.
    """
    try:
        save_func = xm.save
    except:
        save_func = torch.save

    if epoch <= 50 and not epoch % 2:
        save_func(net_dict, path)

    elif epoch <= 100 and not epoch % 5:
        save_func(net_dict, path)

    elif epoch <= 200 and not epoch % 10:
        save_func(net_dict, path)

    elif epoch <= 500 and not epoch % 30:
        save_func(net_dict, path)

    elif epoch <= 2000 and not epoch % 50:
        save_func(net_dict, path)

    elif epoch <= 5000 and not epoch % 80:
        save_func(net_dict, path)

    elif epoch > 5000 and not epoch % 120:
        save_func(net_dict, path)

    elif is_last_epoch:
        save_func(net_dict, path)

    return None


def get_saved_epoch_list(n_epoch):
    """
    This is a silly function.
    Just to get a list of epoch id list.
    This will be used in experiments to decide which to load.
    """

    def get_saved_epoch(epoch):
        if epoch <= 50:
            return epoch

        elif epoch <= 100 and not epoch % 5:
            return epoch

        elif epoch <= 200 and not epoch % 10:
            return epoch

        elif epoch <= 500 and not epoch % 30:
            return epoch

        elif epoch <= 2000 and not epoch % 50:
            return epoch

        elif epoch <= 5000 and not epoch % 80:
            return epoch

        return None

    # Add the saved epochs to the list
    saved_epoch_list = []
    for epoch in range(n_epoch):
        if get_saved_epoch(epoch) is not None:
            saved_epoch_list.append(get_saved_epoch(epoch))

    # Make sure you do not miss the last epoch
    if n_epoch not in saved_epoch_list:
        saved_epoch_list.append(n_epoch)

    return saved_epoch_list


def save_net_dicts_tpu(net, final_path, n_epochs, device):
    """
    Resave the net dicts in CPU.
    We cannot do this in training as the optimizer will hate us.

    Input:
        net (nn.Module), final_path (path.Path)
    Returns:
        None. The net dicts will be resaved.
    """

    # Make an anouncement
    print('I, the machine, am horsing around with TPU.')

    # Get the list of net dicts' path
    net_dict_list = get_saved_epoch_list(n_epochs)
    net_dict_paths = [final_path / 'state_dicts' / f'epoch_{i}.pkl'
                      for i in net_dict_list]
    net_dict_paths.append(final_path / 'model.tar')
    net_dict_paths.append(final_path / 'model_best.tar')

    # Re-save the net dicts in CPU format
    # We cannot do this in training as optimizer will hate that
    for net_dict_path in net_dict_paths:
        # Load the state dict
        try:
            net_dict = torch.load(net_dict_path, map_location='cpu')
        except:
            pass

        # Move net to CPU
        net = net.cpu()
        net.load_state_dict(net_dict)
        net_dict = net.cpu().state_dict()

        # Save the CPU version net dict
        torch.save(net_dict, net_dict_path)

        # Move back net from CPU to TPU for next iter
        net = xmp.MpModelWrapper(net)
        net = net.to(device=device)

    return None


# =========================================================================
# 4. Calculating gradients
# =========================================================================
def get_snr(net,
            data_loader,
            criterion,
            optimizer,
            grad_dict,
            epoch,
            device,
            final_path,
            prune_indicator):
    """
    This function with save the mean/std for gradients and weights of each layer.
    We first record the moving mean/std, then calculate the norm of the moving mean/std.

    The detailed description for the arguments can be found in ../optim/trainer.py.
    """

    # ------------ Set up an epoch-wise dictionary ------------- #
    # Set up the layer dict, where the key is the layer name
    # Notice that this is a transient dict
    # The information will be extracted and then saved in grad_dict
    layer_dict = OrderedDict()
    for name, layer in net.named_modules():
        if type(layer) in (nn.Linear, nn.Conv2d):
            layer_dict[name] = {'weight_welford': WelfordBs(),
                                'grad_welford': WelfordBs()}

    # ------------ Get statistics for (moving) gradient mean ------------- #
    for data in data_loader:
        # Load data
        inputs, y, _ = data
        if device in ['cpu', 'cuda']:
            inputs, y = inputs.to(device), y.to(device)

        # Calculate gradients
        outputs = net(inputs)
        loss = criterion(outputs, y)
        # optimizer.zero_grad()  # Should better use net.zero_grad()
        net.zero_grad()
        loss.backward()

        # Record gradients; appending n_batches_grad per epoch
        for name, layer in net.named_modules():
            if type(layer) in (nn.Linear, nn.Conv2d):
                # Get gradients according to pruning or not
                if prune_indicator:
                    weight_mask_ = layer.weight_mask.detach().reshape(-1).bool()
                    grad_ = layer.weight_orig.grad.detach().reshape(-1)[weight_mask_]
                    weight_ = layer.weight.detach().reshape(-1)[weight_mask_]
                else:
                    grad_ = layer.weight.grad.detach().reshape(-1)
                    weight_ = layer.weight.detach().reshape(-1)

                # Get the moving average
                layer_dict[name]['grad_welford'](grad_)
                layer_dict[name]['weight_welford'](torch.norm(weight_))

                # Debug note
                # The calculating for weights seems to be redundant
                # We use the same weights (network) during this process

        # optimizer.zero_grad()
        net.zero_grad()
        del grad_, weight_

    # ------------ Get mean of epoch-wise gradients ------------- #
    # Get essential data to draw SNR plot
    for name, layer in net.named_modules():
        if type(layer) in (nn.Linear, nn.Conv2d):
            # Get gradient mean and its norm
            grad_welford_mean_norm = torch.norm(layer_dict[name]['grad_welford'].mean)
            grad_dict[f'epoch {epoch}']['grad_mean'].append(grad_welford_mean_norm)

            # Get gradient std
            grad_welford_std_norm = torch.norm(layer_dict[name]['grad_welford'].std)
            grad_dict[f'epoch {epoch}']['grad_std'].append(grad_welford_std_norm)

            # Get weight norm
            weight_welford_norm = layer_dict[name]['weight_welford'].mean
            grad_dict[f'epoch {epoch}']['weight_norm'].append(weight_welford_norm)

    # ------------ Save results ------------- #
    torch.save(grad_dict, final_path / 'grad_dict.pkl')
    return None
