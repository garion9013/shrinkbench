from abc import ABC
import os, sys
import numpy as np
from matplotlib import pyplot as plt

import torch
import torchvision

from shrinkbench.experiment import PruningExperiment

os.environ['DATAPATH'] = '/home/younghwan/workspace/shrinkbench/data'
os.environ['WEIGHTSPATH'] = '/home/younghwan/workspace/shrinkbench/pretrained'


alexnet = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)

def debugger(function):
    def wrapper(self, step):
        sparsity = function(self, step)
        print("({}) ({}) ({}) ({})".format(self.begin_step, self.end_step, step, sparsity))
        return sparsity
    return wrapper
    

# Zhu et al.,ICLR '18 and keras implementation
# @debugger
# Stateless generator
def polynomial_decay_const_freq(self, step):
    # Gradually increasing pruning rate
    p = min(1.0, max(0.0, (step - self.begin_step) / (self.end_step - self.begin_step)))
    sparsity = (self.initial_sparsity - self.final_sparsity) * pow(1 - p, 3) + self.final_sparsity

    # Constant frequency
    waiting_steps = 100
    return sparsity, waiting_steps

# Stateful generator
class const_sp_const_freq(ABC):
    def __init__(self, pruning):
        target_sparsity = 0.95

        # Constant sparsity w/ n-steps
        # Constant state
        self.n = 10
        self.step_sparsity = 1-(1-target_sparsity)**(1.0/self.n)
        self.waiting_steps = np.floor(float(pruning.end_step - pruning.begin_step) / self.n)

        # Variable state
        self.sparsity = self.step_sparsity

    def next(self, step):
        # Constant sparsity w/ n-steps
        prev_sparsity = self.sparsity
        self.sparsity = 1 - (1-self.step_sparsity)*(1-self.sparsity)

        # Constant frequency
        return prev_sparsity, waiting_steps

def const_sp_lraware_freq(self, step):
    n = 2
    # Constant sparsity w/ n-steps
    target_sparsity = 0.98
    sparsity = 1-(1-target_sparsity)**(1.0/(n+1))

    # Constant frequency
    waiting_steps = np.ceil(float(self.end_step - self.begin_step) / n)
    return sparsity, waiting_steps

def nosparse(self, step):
    sparsity = 0

    # Constant frequency
    waiting_steps = sys.maxsize
    return sparsity, waiting_steps


# for strategy in ['GlobalMagWeightInclusive']:
for strategy in ['GlobalMagWeight']:
    exp = PruningExperiment(
                # dataset='ImageNet', 
                # model=alexnet,
                dataset='CIFAR10', 
                model='resnet20',
                # dataset='CIFAR100', 
                # model='resnet20_100',
                # dataset='MNIST', 
                # model='MnistNet',
                pruning_kwargs={
                    'initial_sparsity': 0.5,
                    'final_sparsity': 0.98,
                    'begin_epoch': 0,
                    'end_epoch': 5,
                    'strategy': strategy,
                    'weight_reset_epoch': 0,
                    # 'schedule': polynomial_decay_const_freq
                    'schedule': const_sp_const_freq
                    # 'schedule': nosparse
                },
                train_kwargs={
                    'epochs': 6,
                },
                dl_kwargs={
                    'batch_size': 128,
                },
                pretrained=True,
                save_freq=10
    )
    exp.run()
