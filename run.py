"""
[Title] run.py
[Usage] The file to train a model and save model information.
"""

from helper import plotter, utils
from loader import load_dataset
from optim import Model
from args import p
from pathlib import Path

import os
import time
import torch
import random
import logging
import numpy as np

# ----------- Handle the TPU Training Framework ----------- #
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
except:
    pass
# --------------------------------------------------------- #


# ############################################
# 1. Preparation
# ############################################
# ===========================================
# 1.1. Parameters
# ===========================================

# ===========================================
# 1.2. Setup random states and devices
# ===========================================
# Set a random random seed
if not p.random_state:
    random_state = np.random.randint(low=0,
                                     high=np.iinfo(np.int32).max,
                                     size=1)[0]
# Use the input argument as the random state
else:
    random_state = p.random_state

random.seed(random_state)
torch.manual_seed(random_state)
np.random.seed(random_state)

# Set up device
if p.device == 'tpu':
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
else:
    device = p.device  # Just keep it as string for convenience

# ===========================================
# 1.3. Define Path
# ===========================================
# Get the time
time_now = time.ctime().split()[1:4]

# Configure the network string for path
net_str = f'{p.net_name}'
if p.net_name == 'mlp': net_str += f'-{p.hidden_act}-{p.hidden_dims}'
if p.net_name == 'resnet': net_str += f'-{p.depth}-{p.widen_factor}'
if p.net_name == 'densenet': net_str += f'-{p.dropRate}-{p.growthRate}-{p.compressionRate}'

# Configure other optimization specifics for the folder name
epoch_str = f'epochs_{p.n_epochs}_'
batch_str = f'batch_{p.batch_size}_'
optim_str = f'{p.optimizer_name}_'
optim_str += f'lr_{p.lr}-{p.lr_schedule}-{p.lr_milestones}-{p.lr_gamma}_'
optim_str += f'mm_{p.momentum}_w_decay_{p.weight_decay}_init_{p.init_method}_'
seed_str = f'seed_{random_state}'
time_str = f'time_{time_now[0]}-{time_now[1]}-{time_now[2]}'

# Set the folder name
folder_name = f'{epoch_str}{batch_str}{optim_str}{seed_str}'

# Append the folder name with the regularizer type
if p.reg_type != 'none':
    folder_name += f'_reg_{p.reg_type}-{p.reg_weight}'

# Uncomment the following if you want the folder name with time string
# folder_name += f'_{time_str}'

# Set the final path
final_path = Path(p.results_root) / f'{p.loader_name}' / net_str / folder_name

# Set the individual path for files inside final path
log_path = final_path / 'training.log'
model_path = final_path / 'model.tar'
results_path = final_path / 'results.pkl'
config_path = final_path / 'config.json'
state_dicts_path = final_path / 'state_dicts'
resume_path = state_dicts_path / f'epoch_{p.resume_epoch}.pkl'
performance_plot_path = final_path / f'performance.pdf'

# Create the path if not exist
if not os.path.exists(final_path): os.makedirs(final_path)
if not os.path.exists(state_dicts_path): os.makedirs(state_dicts_path)

# ===========================================
# 1.4. Setup Logger
# ===========================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler(log_path)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.info(f'Random state: {random_state}')
logger.info(final_path)
logger.info(time_str)


# ############################################
# 2. Model Training
# ############################################
# ---------------------- Configure dataset ---------------------- #
dataset = load_dataset(p.loader_name,
                       p.root,
                       p.filename,
                       p.test_size,
                       random_state,
                       p.download)

# ------------------ Set up model with trainer ------------------ #
model = Model()
model.set_network(p.net_name, p.in_dim, p.out_dim, p.hidden_act,
                  p.out_act, p.hidden_dims, p.depth, p.widen_factor,
                  p.dropRate, p.growthRate, p.compressionRate)

# ---------------------- Set training mode (resuming) ---------------------- #
if p.resume_epoch:
    logger.info(f'Resuming training from {resume_path}.')
    model.load_net_dict(resume_path, device)

# -------------------- Set training mode (non-resuming) -------------------- #
if not p.resume_epoch:
    # Initialize the network
    model.init_network(p.init_method)

# ---------------------- Training the model --------------------- #
model.train(dataset, p.optimizer_name, p.momentum, p.lr, p.lr_schedule,
            p.lr_milestones, p.lr_gamma, p.n_epochs, p.batch_size,
            p.weight_decay, device, p.n_jobs_dataloader,
            p.reg_type, p.reg_weight, final_path, p.resume_epoch)


# ############################################
# 3. Save Statistics
# ############################################
# Save net dicts, results and configs
model.save_net_dict(model_path, p.save_best)
model.save_config(config_path)

# Handle the net dicts saved by TPU
if p.device == 'tpu':
    utils.save_net_dicts_tpu(model.net, final_path, p.n_epochs, device)

# ############################################
# 4. Plot for gradients
# ############################################
# Notice on the plotting stage
logger.info('Stay tuned: your little Van Gogh is drawing...')

# Save the performance plot
plotter.plot_performance(final_path, performance_plot_path)

logger.info('All done. Good luck!')
logger.info(str(final_path))
