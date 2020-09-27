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
   

# Zhu et al.,ICLR '18 and keras implementation
# Stateless generator
def polynomial_decay_const_freq(ctxt, step=0, initial_sparsity=0.5, final_sparsity=0.95, waiting_step=100):
    # Gradually increasing pruning rate
    p = min(1.0, max(0.0, (step - ctxt.begin_step) / (ctxt.end_step - ctxt.begin_step)))
    sparsity = (initial_sparsity - final_sparsity) * pow(1 - p, 3) + final_sparsity

    return sparsity, waiting_step

# Stateful generator
class const_sp_const_freq(ABC):
    def __init__(self, ctxt, n=10, final_sparsity=0.95):

        # Constant sparsity w/ n-steps
        # Constant state
        self.step_sparsity = 1-(1-final_sparsity)**(1.0/(n+1))
        self.waiting_steps = np.floor(float(ctxt.end_step - ctxt.begin_step) / n)

        # Variable state
        self.sparsity = self.step_sparsity

    def next(self, step):
        # Constant sparsity w/ n-steps
        prev_sparsity = self.sparsity
        self.sparsity = 1 - (1-self.step_sparsity)*(1-self.sparsity)

        # Constant frequency
        return prev_sparsity, self.waiting_steps

def nosparse(ctxt, step):
    sparsity = 0

    # Constant frequency
    waiting_steps = sys.maxsize
    return sparsity, waiting_steps

# exploration = [
#     {"initial_sparsity":0, 'final_sparsity':0.98, 'waiting_step':s} for s in [10,50,100,200,400,800,1600,3200]
# ]
# exploration = [ {"n": i, "final_sparsity":0.98} for i in [8,16] ]
# exploration = [ {"n": i, "final_sparsity":0.98} for i in [256,512] ]
exploration = [ {"n": i, "final_sparsity":0.95} for i in [2,4] ]

train_epoch = 300
for strategy in ['GlobalMagWeightInclusive']:
# for strategy in ['GlobalMagWeight']:
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
                        'begin_epoch': 5,
                        'end_epoch': 200,
                        'strategy': strategy,
                        'capture_epoch_for_reset': 5,
                        'weight_reset': True,
                        # ==================================================================
                        # Pruning rate, iteration scheduler
                        # ==================================================================
                        # 'scheduler': polynomial_decay_const_freq,
                        # 'scheduler_args': scheduler_args 
                        'scheduler': const_sp_const_freq,
                        'scheduler_args': scheduler_args
                        # 'scheduler': nosparse,
                    },
                    train_kwargs={
                        # ==================================================================
                        # Learning rate scheduler
                        # ==================================================================
                        # 'optim': "Adam",
                        'optim': "SGD",
                        'lr': 1e-3,
                        'lr_scheduler': True,
                        'epochs': train_epoch,
                        'earlystop_args': {'patience': 10} # earlystop checking starts after end_epoch
                    },
                    dl_kwargs={
                        'batch_size': 128,
                        'pin_memory': True,
                        'num_workers': 0,
                    },
                    pretrained=False,
                    save_freq=9999,
                    gpu_number=0,
                    # path=pathlib.Path(f"./results/polynomial-{scheduler_args['final_sparsity']}-step-{scheduler_args['waiting_step']}")
                    path=pathlib.Path(f"./results/scratch/resnet20/const-0.95-{scheduler_args['n']}-sgd-inclusive")
        )
        exp.run()
