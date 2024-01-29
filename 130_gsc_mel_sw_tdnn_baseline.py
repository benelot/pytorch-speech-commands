#!/usr/bin/env python

"""Train a CNN for Google speech commands."""

# adapted from: https://github.com/tugstugi/pytorch-speech-commands

import argparse
import time

import sys
sys.path.append("..")


import torch
import torch.nn as nn

from tqdm import *

from datasets import CLASSES, get_gsc_dataloaders

from gsc_training import bens_run
from memory import SlidingWindow

class MLPBase(nn.Module):
  def __init__(self, input_size, output_size, hidden_size=100):
    super(MLPBase, self).__init__()
    self.input_size = input_size
    self.linear1 = nn.Linear(input_size, hidden_size)
    self.linear2 = nn.Linear(hidden_size, hidden_size)
    self.linear3 = nn.Linear(hidden_size, output_size)
    print("Initialized MLPBase model with {} parameters".format(self.count_params()))

  def count_params(self):
    return sum([p.view(-1).shape[0] for p in self.parameters()])

  def forward(self, x):
    x = x.reshape(x.shape[0], -1) # flatten the input
    h = self.linear1(x).relu()
    h = self.linear2(h).relu() # no residual connection # TODO: the results are better without the residual connection :D
    return self.linear3(h)

import argparse

if __name__ == '__main__':

    params = {
        # NN parameters
        "model_type": "tdnn",
        # gsc parameters
        # other parameters
        "use_cuda": True,
        "with_neptune": True,
    }

    # parse parameters from command line which often change
    parser = argparse.ArgumentParser(description='Train a GLE network on the MNIST1D dataset.')
    parser.add_argument('--no_neptune', action='store_true', default=False, help='Do not use neptune.')
    parser.add_argument('--seed', type=int, default=12, help='Random seed.')
    parser.add_argument("--train-dataset", type=str, default='datasets/speech_commands/train', help='path of train dataset')
    parser.add_argument("--valid-dataset", type=str, default='datasets/speech_commands/valid', help='path of validation dataset')
    parser.add_argument("--background-noise", type=str, default='datasets/speech_commands/train/_background_noise_', help='path of background noise')
    parser.add_argument("--batch-size", type=int, default=128, help='batch size')
    parser.add_argument("--dataload-workers-nums", type=int, default=6, help='number of workers for dataloader')
    parser.add_argument("--weight-decay", type=float, default=0, help='weight decay')
    parser.add_argument("--optim", choices=['sgd', 'adam'], default='adam', help='choices of optimization algorithms')
    parser.add_argument("--learning-rate", type=float, default=5e-5, help='learning rate for optimization')
    parser.add_argument("--lr-scheduler", choices=['plateau', 'step'], default='plateau', help='method to adjust learning rate')
    parser.add_argument("--lr-scheduler-patience", type=int, default=5, help='lr scheduler plateau: Number of epochs with no improvement after which learning rate will be reduced')
    parser.add_argument("--lr-scheduler-step-size", type=int, default=50, help='lr scheduler step: number of epochs of learning rate decay.')
    parser.add_argument("--lr-scheduler-gamma", type=float, default=0.5, help='learning rate is multiplied by the gamma to decrease it')
    parser.add_argument("--max-epochs", type=int, default=60, help='max number of epochs')
    parser.add_argument("--n-mels", choices=[32, 40], default=32, help='input of NN')
    parser.add_argument("--input-size", type=int, default=8, help='temporal kernel/input size of NN')
    parser.add_argument("--use-le", action='store_true', default=False, help='Use LE instead of ANN')
    args = parser.parse_known_args()[0]

    params['with_neptune'] = not args.no_neptune
    print("Using neptune: {}".format(params['with_neptune']))

    params['seed'] = args.seed
    print("Using seed: {}".format(params['seed']))

    params['epochs'] = args.max_epochs
    print("Using max epochs: {}".format(params['epochs']))

    params['batch_size'] = args.batch_size
    print("Using batch size: {}".format(params['batch_size']))

    params['lr'] = args.learning_rate
    print("Using learning rate: {}".format(params['lr']))

    params['lr_scheduler'] = args.lr_scheduler
    print("Using lr scheduler: {}".format(params['lr_scheduler']))

    params['input_size'] = args.input_size
    print("Using input size: {}".format(params['input_size']))

    params['use_le'] = args.use_le
    print("Using LE: {}".format(params['use_le']))

    # import neptune and connect to neptune.ai
    name = "130-gsc-mel-classification"

    torch.manual_seed(params["seed"])

    if params["use_le"]:
        raise NotImplementedError("LE not implemented for this model")
    else:
        model = MLPBase(args.n_mels * params['input_size'], len(CLASSES), hidden_size=400)
        model.input_size = params['input_size']

    if torch.cuda.is_available() and params['use_cuda']:
        model.cuda()
    else:
        params['use_cuda'] = False

    if params['use_cuda']:
        torch.backends.cudnn.benchmark = True # TODO: Will change acc, remove at some point


    memory = SlidingWindow
    memory.kwargs = {'window_size': model.input_size}
    print("Using Sliding Window: window_size={}".format(params['input_size']))

    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    if args.lr_scheduler == 'plateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.lr_scheduler_patience, factor=args.lr_scheduler_gamma, eps=1e-20)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_scheduler_step_size, gamma=args.lr_scheduler_gamma, last_epoch=-1)

    criterion = torch.nn.CrossEntropyLoss()

    # TODO: This will change the acc, enable later
    # torch.manual_seed(params["seed"])
    bens_run(args, params, name, memory, model, *get_gsc_dataloaders(args.n_mels, args, params['use_cuda']), criterion, optimizer, lr_scheduler)

