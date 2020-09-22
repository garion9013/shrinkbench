from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy as np
import pandas as pd

from .mask import mask_module
from .modules import MaskedModule
from .utils import get_params


class Pruning(ABC):

    """Base class for Pruning operations
    """

    def __init__(self, model, inputs=None, outputs=None, **pruning_params):
        """Construct Pruning class

        Passed params are set as attributes for convienence and
        saved internally for __repr__

        Arguments:
            model {torch.nn.Module} -- Model for which to compute masks
            inputs {torch.nn.Tensor} -- Sample inputs to estimate activation &| gradients
            outputs {torch.nn.Tensor} -- Sample outputs to estimate activation &| gradients
        Keyword Arguments:
            **pruning_params {dict} -- [description]
        """
        self.model = model
        self.inputs = inputs
        self.outputs = outputs
        self.pruning_params = list(pruning_params.keys())
        for k, v in pruning_params.items():
            setattr(self, k, v)

        if isinstance(self.scheduler, type):
            self.scheduler_gen = self.scheduler(self, **self.scheduler_args)

    @abstractmethod
    def model_masks(self, prunable=None):
        """Compute masks for a given model

        """
        # TODO Also accept a dataloader
        pass
        # return masks

    def update_context(self, step):
        # Update prunable parameters after backward pass
        if hasattr(self, "scheduler_gen"):
            # from generator class (stateful)
            sparsity, next_waiting_steps = self.scheduler_gen.next(step)
        else:
            # from generator fn (stateless)
            sparsity, next_waiting_steps = self.scheduler(self, step=step, **self.scheduler_args)

        self.compression = 1/(1-sparsity)
        assert self.compression >= 1, "Unacceptable compression rate"
        self.init(self.compression)
        return next_waiting_steps

    def apply(self, step):
        next_waiting_steps = self.update_context(step)
        masks = self.model_masks()
        mask_module(self.model, masks)
        return next_waiting_steps

    @abstractmethod
    def can_prune(self, module):
        pass

    def prunable_modules(self):
        prunable = [module for module in self.model.modules() if self.can_prune(module)]
        return prunable

    def __repr__(self):
        s = f"{self.__class__.__name__}("
        for k in self.pruning_params:
            s += f"{k}={repr(getattr(self, k))}, "
        s = s[:-2] + ')'
        return s

    def __str__(self):
        return repr(self)

    def module_params(self, module):
        return get_params(module)

    def params(self, only_prunable=True):
        if only_prunable:
            return {module: get_params(module) for module in self.prunable}
        else:
            return {module: get_params(module) for module in self.model.modules()}

    def summary(self):
        rows = []
        for name, module in self.model.named_modules():
            for pname, param in module.named_parameters(recurse=False):
                if isinstance(module, MaskedModule):
                    compression = 1/getattr(module, pname+'_mask').detach().cpu().numpy().mean()
                else:
                    compression = 1
                shape = param.detach().cpu().numpy().shape
                rows.append([name, pname, compression, np.prod(shape), shape, self.can_prune(module)])
        columns = ['module', 'param', 'comp', 'size', 'shape', 'prunable']
        return pd.DataFrame(rows, columns=columns)


class LayerPruning(Pruning):

    @abstractmethod
    def layer_masks(self, module):
        """Instead of implementing masks for the entire model at once
        User needs to specify a layer_masks fn that can be applied layerwise

        Should return None is layer can't be masked
        """
        pass
        # return masks

    def model_masks(self, prunable=None):
        """Compute masks using the said strategy for every module
        This is a straight forward implementation that supports
        strategies that prune each module independently
        """
        masks = OrderedDict()
        if prunable is None:
            prunable = self.prunable_modules()

        for module in prunable:
            masks_ = self.layer_masks(module)
            if masks_ is not None:
                masks[module] = masks_

        return masks
