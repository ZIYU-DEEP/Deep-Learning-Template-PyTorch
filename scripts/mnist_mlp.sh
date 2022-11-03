#!/bin/bash

results_root=./results

# Arguments for dataset
loader_name=mnist
filename=MNIST
out_dim=10
in_dim=784

# Arguments for network
net_name=mlp
hidden_act=relu
hidden_dims=64-64-32-32-18-18
out_act=softmax

# Arguments for [dense training]
lr_pre=0.01
lr_schedule_pre=stepwise
lr_milestones_pre=10
lr_gamma_pre=0.5
n_epochs_pre=20
batch_size_pre=32

# Arguments for saving things
save_best=1

# Other unimportant Arguments
optimizer_name=sgd
momentum=0.9
weight_decay=0.0002
device=cpu
random_state=42
init_method=kaiming
n_jobs_dataloader=0

# Automatically generate the pretrain path based on the parameters
# You have to run dense in order to have model stored in this path
NET=${net_name}-${hidden_act}-${hidden_dims}
EPOCH=epochs_${n_epochs_pre}
BATCH=batch_${batch_size_pre}
OPTIM=${optimizer_name}_
OPTIM+=lr_${lr_pre}-${lr_schedule_pre}-${lr_milestones_pre}-${lr_gamma_pre}_
OPTIM+=mm_${momentum}_w_decay_${weight_decay}_init_${init_method}
SEED=seed_${random_state}

PRE_PATH=${results_root}/${loader_name}/${NET}/${EPOCH}_${BATCH}_${OPTIM}_${SEED}/model.tar

# 1. Dense Model
python run.py -ln ${loader_name} -fn ${filename} -rs ${random_state} -nt ${net_name} -in ${in_dim} -ot ${out_dim} -im ${init_method} -on ${optimizer_name} -mm ${momentum} -lr ${lr_pre} -ls ${lr_schedule_pre} -lm ${lr_milestones_pre} -lg ${lr_gamma_pre} -ne ${n_epochs_pre} -bs ${batch_size_pre} -wt ${weight_decay} -dv ${device} -nj ${n_jobs_dataloader} -sb ${save_best} -hd ${hidden_dims} -ha ${hidden_act}
