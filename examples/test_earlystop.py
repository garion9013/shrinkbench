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

# Stateful generator
class const_sp_const_freq(ABC):
    def __init__(self, ctxt, n=10):
        target_sparsity = 0.98

        # Constant sparsity w/ n-steps
        # Constant state
        self.step_sparsity = 1-(1-target_sparsity)**(1.0/(n+1))
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

train_epoch = 1000
exp = PruningExperiment(
            dataset='MNIST', 
            model='MnistNet',
            train_kwargs={
                # ==================================================================
                # Learning rate scheduler
                # ==================================================================
                'optim': 'SGD',
                'lr': 1e-3,
                'lr_scheduler': False,
                'epochs': train_epoch,
                'earlystop_args': {'patience': 20}
            },
            dl_kwargs={
                'batch_size': 128,
                'pin_memory': True,
                'num_workers': 8,
            },
            pruning_kwargs={
                "begin_epoch": 0,
                "end_epoch": 20,
                "strategy": "GlobalMagWeight",
                "scheduler": const_sp_const_freq,
                "scheduler_args": {"n": 32} 
            },
            pretrained=False,
            save_freq=10,
            gpu_number=3,
            path=pathlib.Path(f"./results/naive-prune-test/")
)
exp.run()
