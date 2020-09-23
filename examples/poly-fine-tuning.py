from abc import ABC
import os, sys
import pathlib
import numpy as np
from matplotlib import pyplot as plt

import torch
import torchvision

from shrinkbench.experiment import PruningExperiment

os.environ['DATAPATH'] = '/home/younghwan/workspace/shrinkbench/data'
os.environ['WEIGHTSPATH'] = '/home/younghwan/workspace/shrinkbench/pretrained'


def debugger(function):
    def wrapper(self, step):
        sparsity = function(self, step)
        print("({}) ({}) ({}) ({})".format(self.begin_step, self.end_step, step, sparsity))
        return sparsity
    return wrapper
    

# Zhu et al.,ICLR '18 and keras implementation
# @debugger
# Stateless generator
def polynomial_decay_const_freq(ctxt, step=0, initial_sparsity=0.5, final_sparsity=0.95, waiting_step=100):
    # Gradually increasing pruning rate
    p = min(1.0, max(0.0, (step - ctxt.begin_step) / (ctxt.end_step - ctxt.begin_step)))
    sparsity = (initial_sparsity - final_sparsity) * pow(1 - p, 3) + final_sparsity

    return sparsity, waiting_step

# Stateful generator
class const_sp_const_freq(ABC):
    def __init__(self, pruning, n=10):
        target_sparsity = 0.95

        # Constant sparsity w/ n-steps
        # Constant state
        self.n = n
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

def nosparse(ctxt, step):
    sparsity = 0

    # Constant frequency
    waiting_steps = sys.maxsize
    return sparsity, waiting_steps

exploration = [
    # {"initial_sparsity":0, 'final_sparsity':0.98, 'waiting_step':10},
    # {"initial_sparsity":0, 'final_sparsity':0.95, 'waiting_step':10},
    # {"initial_sparsity":0, 'final_sparsity':0.90, 'waiting_step':10},
    # {"initial_sparsity":0, 'final_sparsity':0.98, 'waiting_step':50},
    # {"initial_sparsity":0, 'final_sparsity':0.95, 'waiting_step':50},
    # {"initial_sparsity":0, 'final_sparsity':0.90, 'waiting_step':50},
    # {"initial_sparsity":0, 'final_sparsity':0.98, 'waiting_step':100},
    # {"initial_sparsity":0, 'final_sparsity':0.95, 'waiting_step':100},
    # {"initial_sparsity":0, 'final_sparsity':0.90, 'waiting_step':100},
    # {"initial_sparsity":0, 'final_sparsity':0.98, 'waiting_step':200},
    # {"initial_sparsity":0, 'final_sparsity':0.95, 'waiting_step':200},
    # {"initial_sparsity":0, 'final_sparsity':0.90, 'waiting_step':200},
    # {"initial_sparsity":0, 'final_sparsity':0.98, 'waiting_step':400},
    # {"initial_sparsity":0, 'final_sparsity':0.95, 'waiting_step':400},
    # {"initial_sparsity":0, 'final_sparsity':0.90, 'waiting_step':400},
    # {"initial_sparsity":0, 'final_sparsity':0.98, 'waiting_step':800},
    # {"initial_sparsity":0, 'final_sparsity':0.95, 'waiting_step':800},
    # {"initial_sparsity":0, 'final_sparsity':0.90, 'waiting_step':800},
    # {"initial_sparsity":0, 'final_sparsity':0.98, 'waiting_step':1600},
    # {"initial_sparsity":0, 'final_sparsity':0.95, 'waiting_step':1600},
    # {"initial_sparsity":0, 'final_sparsity':0.90, 'waiting_step':1600},
    # {"initial_sparsity":0, 'final_sparsity':0.98, 'waiting_step':3200},
    # {"initial_sparsity":0, 'final_sparsity':0.95, 'waiting_step':3200},
    # {"initial_sparsity":0, 'final_sparsity':0.90, 'waiting_step':3200},
    {"initial_sparsity":0, 'final_sparsity':0.98, 'waiting_step':s} for s in [10,50,100,200,400,800,1600,3200]
]

train_epoch = 20
# for strategy in ['GlobalMagWeightInclusive']:
for strategy in ['GlobalMagWeight']:
    for scheduler_args in exploration:
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
                        'begin_epoch': 0,
                        'end_epoch': train_epoch-1,
                        'strategy': strategy,
                        'weight_reset_epoch': 0,
                        # ==================================================================
                        # Pruning rate, iteration scheduler
                        # ==================================================================
                        'scheduler': polynomial_decay_const_freq,
                        'scheduler_args': scheduler_args 
                        # 'scheduler': const_sp_const_freq,
                        # 'scheduler_args': scheduler_args
                        # 'scheduler': nosparse,
                    },
                    train_kwargs={
                        # ==================================================================
                        # Learning rate scheduler
                        # ==================================================================
                        'optim': 'Adam',
                        # 'lr': 1e-3,
                        'lr_scheduler': False,
                        'epochs': train_epoch,
                    },
                    dl_kwargs={
                        'batch_size': 128,
                        'pin_memory': True,
                    },
                    pretrained=True,
                    save_freq=10,
                    path=pathlib.Path(f"./results/epoch-20/adam-{scheduler_args['final_sparsity']}-step-{scheduler_args['waiting_step']}")
        )
        exp.run()
