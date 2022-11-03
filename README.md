# 😈 PyTorch Template for Deep Learning
<!-- *This file is last updated by Ziyu on June, 2022.* -->

[![Python 3.8](https://img.shields.io/badge/python-3.8-blueviolet.svg)](https://www.python.org/downloads/release/python-380/) [![Pytorch](https://img.shields.io/badge/Pytorch-1.12.1-critical.svg)](https://github.com/pytorch/pytorch/releases/tag/v1.12.0) [![License](https://img.shields.io/badge/License-Apache%202.0-ff69b4.svg)](https://opensource.org/licenses/Apache-2.0) [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-success.svg)](https://github.com/ZIYU-DEEP/Generalization-and-Memorization-in-Sparse-Training)
## 1. Intro
This repository is organized as follows:
```bash
|== loader
│   └── loader_cifar100.py
│   └── main.py
|== network
│   └── mlp.py
│   └── resnet.py
│   └── main.py
|== optim
│   └── trainer.py
│   └── model.py
|== helper
│   └── plotter.py
│   └── utils.py
|== scripts
│   └── cifar100_resnet.sh
|== args.py
|== run.py
```

## 2. Requirements
### Working with CPU/GPU
To install necessary pakacges, check the list in `./requirements.txt` or lazily run the following in the designated environment for the project:
```bash
python3 -m pip install -r requirements.txt
```

### Working with TPU
[July 2022] If you are using TPUs on Google Cloud platform, please make sure you have also run the following (more information can be found [here](https://cloud.google.com/tpu/docs/run-calculation-pytorch#tpu-vm)).
```bash
# Config the TPU
echo "export XRT_TPU_CONFIG='localservice;0;localhost:51011'" >> ~/.bashrc
source ~/.bashrc

# Install torch_xla; you may install a previous version if the following does not work
pip install https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.9-cp37-cp37m-linux_x86_64.whl

# Install cloud client for tpu
pip install cloud-tpu-client
```
When running experiments with TPUs, you should set `--device tpu` in the arguments of `args.py`.

## 3. Example Commands
```shell
. scripts/mnist_mlp.sh
```
Please check the parameter part of `./args.py` for more detailed instructions on datasets, network, and optimization options. I have written help strings for every argument.
