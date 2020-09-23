""" Early stop implementations

https://github.com/Bjarten/early-stopping-pytorch

"""

import numpy as np

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.loss_min_step = 0
        self.delta = delta
        self.trace_func = trace_func

    def __call__(self, step, loss, acc1, acc5):

        score = -loss

        if self.best_score is None:
            self.best_score = score
            self.save_step(step, loss, acc1, acc5)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_step(step, loss, acc1, acc5)
            self.counter = 0

    def save_step(self, step, loss, acc1, acc5):
        '''Saves step when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving step ...')
        self.val_loss_min = loss
        self.corr_val_acc1 = acc1
        self.corr_val_acc5 = acc5
        self.loss_min_step = step
