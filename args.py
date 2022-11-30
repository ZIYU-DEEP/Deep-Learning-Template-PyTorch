"""
[Title] arguments.py
[Usage] The file to feed in arguments.
"""

import argparse


# Initialize the parser
parser = argparse.ArgumentParser()

# Arguments for general training mechanism
parser.add_argument('-rr', '--results_root', type=str, default='./results',
                    help='The directory to save results.')

# Arguments for loading datasets
parser.add_argument('-ln', '--loader_name', type=str, default='toy',
                    help='The name for your dataset.',
                    choices=['toy', 'mnist', 'cifar10', 'cifar100',
                             'fmnist', 'cifar100_tpu', 'image',
                             'cifar10_noisy', 'fmnist_noisy', 'tiny_imagenet'])
parser.add_argument('-rt', '--root', type=str, default='./data/',
                    help='The root path for stored data.')
parser.add_argument('-fn', '--filename', type=str, default='toy',
                    help='The filename for your data, e.g., toy, MNIST.')
parser.add_argument('-ts', '--test_size', type=float, default=0.2,
                    help='The test split ratio for the toy dataset.')
parser.add_argument('-rs', '--random_state', type=int, default=42,
                    help='Use 0 to set a random random seed for you.')
parser.add_argument('-dl', '--download', type=int, default=0,
                    help='1 if you need download the dataset, otherwise 0.')

# Arguments for setting network
parser.add_argument('-nt', '--net_name', type=str, default='mlp',
                    help='The name for your network',
                    choices=['mlp', 'alexnet', 'preresnet', 'resnet',
                             'densenet', 'vgg11', 'vgg11_bn', 'vgg13',
                             'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19',
                             'vgg19_bn', 'mnist_lenet', 'mnist_alexnet'])
parser.add_argument('-in', '--in_dim', type=int, default=12,
                    help='The feature dimension for the input data X.')
parser.add_argument('-ot', '--out_dim', type=int, default=2,
                    help='The number of classes of the output data y.')
parser.add_argument('-ha', '--hidden_act', type=str, default='tanh',
                    help='e.g., tanh, relu, softmax, sigmoid.')
parser.add_argument('-oa', '--out_act', type=str, default='softmax',
                    help='The activation for the output layer, e.g., softmax=.')
parser.add_argument('-hd', '--hidden_dims', type=str, default='10-7-5-4-3',
                    help='Hidden dimensions for MLP; using hypen to connect.')
parser.add_argument('-dp', '--depth', type=int, default=32,
                    help='The depth for preresnet or densenet.')
parser.add_argument('-wf', '--widen_factor', type=int, default=4,
                    help='The widen factor for resnet.')
parser.add_argument('-dr', '--dropRate', type=int, default=0,
                    help='The drop rate for dense net.')
parser.add_argument('-gr', '--growthRate', type=int, default=12,
                    help='The growth rate for dense net.')
parser.add_argument('-cr', '--compressionRate', type=int, default=1,
                    help='The compression rate for densenet.')

# Arguments for the model
parser.add_argument('-im', '--init_method', type=str, default='kaiming',
                    help='Currently support kaiming, xavier, and none.')
parser.add_argument('-on', '--optimizer_name', type=str, default='adam',
                    help='Currently support sgd or adam.')
parser.add_argument('-mm', '--momentum', type=float, default=0.9,
                    help='The momentum of the optimizer.')
parser.add_argument('-lr', '--lr', type=float, default=0.0004,
                    help='The learning rate for optimization.')
parser.add_argument('-ls', '--lr_schedule', type=str, default='stepwise',
                    help='Currently support stepwise or exponential.')
parser.add_argument('-lm', '--lr_milestones', type=str, default='50-100',
                    help='The milestones for stepwise lr scheduler.')
parser.add_argument('-lg', '--lr_gamma', type=float, default=0.1,
                    help='The decay rate for learning rate.')
parser.add_argument('-ne', '--n_epochs', type=int, default=500,
                    help='The number of training epochs.')
parser.add_argument('-bs', '--batch_size', type=int, default=256,
                    help='The batch size for training.')
parser.add_argument('-wt', '--weight_decay', type=float, default=1e-6,
                    help='The decay rate for weight.')
parser.add_argument('-dv', '--device', type=str, default='cpu',
                    choices=['cpu', 'cuda', 'tpu'])
parser.add_argument('-nj', '--n_jobs_dataloader', type=int, default=0,
                    help='The number of workers; default is just 0.')

# Arguments for the loss regularizer
parser.add_argument('-rp', '--reg_type', type=str, default='none',
                    choices=['none', 'jacobian'],
                    help='The type of loss regularization.')
parser.add_argument('-rw', '--reg_weight', type=float, default='0.01',
                    help='The coefficient for the loss regularizer.')

# Arguments for the indicators of saving things or training config
parser.add_argument('-sb', '--save_best', type=int, default=1,
                    help='1 if save the best model, else 0.')

# Arguments for resume training
# Some clusters have a 4-hour GPU usage limit
# If your training is interrupted, you may just resume it
# by telling the code the epoch to resume
# Notice no other parameter needed to change
parser.add_argument('-ree', '--resume_epoch', type=int, default=0,
                    help='The epoch to resume the training')

p = parser.parse_args()
