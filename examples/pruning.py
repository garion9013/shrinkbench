from shrinkbench.experiment import PruningExperiment
from matplotlib import pyplot as plt
import numpy as np

import os
os.environ['DATAPATH'] = '/home/younghwan/workspace/shrinkbench/data'
os.environ['WEIGHTSPATH'] = '/home/younghwan/workspace/shrinkbench/pretrained'

import torch
import torchvision

alexnet = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)

def debugger(function):
    def wrapper(self, step):
        sparsity = function(self, step)
        print("({}) ({}) ({}) ({})".format(self.begin_step, self.end_step, step, sparsity))
        return sparsity
    return wrapper
    

# Zhu et al.,ICLR '18 and keras implementation
# @debugger
def polynomial_decay_const_freq(self, step):
    # Gradually increasing pruning rate
    p = min(1.0, max(0.0, (step - self.begin_step) / (self.end_step - self.begin_step)))
    sparsity = (self.initial_sparsity - self.final_sparsity) * pow(1 - p, 3) + self.final_sparsity

    # Constant frequency
    waiting_steps = 100
    return sparsity, waiting_steps

def const_sp_const_freq(self, step):
    n = 10
    # Constant sparsity w/ n-steps
    target_sparsity = 0.95
    sparsity = 1-(1-target_sparsity)**(1.0/n)

    # Constant frequency
    waiting_steps = np.floor(float(self.end_step - self.begin_step) / n)
    return sparsity, waiting_steps

def const_sp_lraware_freq(self, step):
    n = 2
    # Constant sparsity w/ n-steps
    target_sparsity = 0.98
    sparsity = 1-(1-target_sparsity)**(1.0/(n+1))

    # Constant frequency
    waiting_steps = np.ceil(float(self.end_step - self.begin_step) / n)
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
                    'end_epoch': 10,
                    'strategy': strategy,
                    'schedule': polynomial_decay_const_freq
                    # 'schedule': const_sp_const_freq
                },
                train_kwargs={
                    'epochs': 10,
                },
                dl_kwargs={
                    'batch_size': 128,
                },
                pretrained=True,
                save_freq=1
    )
    exp.run()
