import json

from .train import TrainingExperiment

from .. import strategies
from ..metrics import model_size, flops
from ..util import printc, OnlineStats
import time
from tqdm import tqdm
import torch
from ..metrics import correct


class PruningExperiment(TrainingExperiment):

    default_pruning_kwargs = {
        'begin_epoch': 0,
        'end_epoch': 10,
        'strategy': "GlobalMagWeight",
        'scheduler': None,
        'scheduler_args': {},
    }

    def __init__(self,
                 dataset,
                 model,
                 seed=42,
                 path=None,
                 dl_kwargs=dict(),
                 train_kwargs=dict(),
                 pruning_kwargs=dict(),
                 debug=False,
                 pretrained=True,
                 resume=None,
                 resume_optim=False,
                 save_freq=10):

        super(PruningExperiment, self).__init__(dataset, model, seed, path, dl_kwargs, train_kwargs, debug, pretrained, resume, resume_optim, save_freq)
        pruning_kwargs = {**self.default_pruning_kwargs, **pruning_kwargs}

        params = locals()
        params['pruning_kwargs'] = pruning_kwargs
        self.add_params(**params)
        # Save params

        # Build pruning context
        self.build_pruning(**pruning_kwargs)
        self.steps_after_pruning = 0
        self.steps = 0

        self.path = path
        self.save_freq = save_freq
        self.metrics = []

    def build_pruning(self, **kwargs):
        constructor = getattr(strategies, kwargs['strategy'])
        data_iter = iter(self.train_dl)
        x, y = next(data_iter)

        # Inferred pruning parameters
        kwargs["compression"] = None
        kwargs["begin_step"] = kwargs["begin_epoch"] * len(data_iter)
        kwargs["end_step"] = kwargs["end_epoch"] * len(data_iter)

        # Firstly apply pruning and update context
        self.pruning = constructor(self.model, x, y, **kwargs)
        self.waiting_steps = kwargs["begin_step"]

    def run(self):
        self.freeze()
        printc(f"Running {repr(self)}", color='YELLOW')
        self.to_device()
        self.build_logging(self.train_metrics, self.path)

        self.save_metrics()

        self.run_epochs()

    def run_epoch(self, train, epoch=0, after_pruning=False):
        if train:
            self.model.train()
            prefix = 'train'
            dl = self.train_dl
        else:
            prefix = 'val'
            dl = self.val_dl
            self.model.eval()

        total_loss = OnlineStats()
        acc1 = OnlineStats()
        acc5 = OnlineStats()

        epoch_iter = tqdm(dl)
        desc = f"{prefix.capitalize()} Epoch {epoch}/{self.epochs}"
        if after_pruning:
            desc = f"{prefix.capitalize()} Steps {self.steps}"
        epoch_iter.set_description(desc)

        with torch.set_grad_enabled(train):
            for i, (x, y) in enumerate(epoch_iter, start=1):
                x, y = x.to(self.device), y.to(self.device)

                # Conditionally triggers pruning:
                # self.steps_after_pruning: how many mini-batch steps are performed 
                # self.steps: total steps during training/pruning process
                # Note that waiting_steps can be dynamically changed by user-specified schedule fn
                # if train and self.steps_after_pruning >= self.waiting_steps:
                #     self.waiting_steps = self.pruning.apply(self.steps)
                #     self.save_metrics(steps=self.steps)
                #     self.steps_after_pruning = 0

                # Forward
                yhat = self.model(x)
                loss = self.loss_func(yhat, y)

                # Backward
                if train:
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()
                    self.steps_after_pruning += 1
                    self.steps += 1

                    
                c1, c5 = correct(yhat, y, (1, 5))
                total_loss.add(loss.item() / dl.batch_size)
                acc1.add(c1 / dl.batch_size)
                acc5.add(c5 / dl.batch_size)

                epoch_iter.set_postfix(loss=total_loss.mean, top1=acc1.mean, top5=acc5.mean)

        self.log(**{
            f'{prefix}_loss': total_loss.mean,
            f'{prefix}_acc1': acc1.mean,
            f'{prefix}_acc5': acc5.mean,
        })

        return total_loss.mean, acc1.mean, acc5.mean

    def run_epochs(self):
        since = time.time()
        try:
            for epoch in range(self.epochs):
                current_lr = self.optim.param_groups[0]['lr']
                printc(f"Start epoch {epoch}, {current_lr:.5e}", color='YELLOW')
                self.train(epoch)
                self.eval(epoch)
                self.lr_scheduler.step()

                # Checkpoint epochs
                # TODO Model checkpointing based on best val loss/acc
                if epoch % self.save_freq == 0:
                    self.checkpoint()
                # TODO Early stopping
                # TODO ReduceLR on plateau?
                self.log(timestamp=time.time()-since, steps=self.steps, lr=current_lr)
                self.log_epoch(epoch)


        except KeyboardInterrupt:
            printc(f"\nInterrupted at epoch {epoch}. Tearing Down", color='RED')

    def save_metrics(self, steps=0):
        metric = self.pruning_metrics()
        metric["steps"] = steps
        self.metrics.append(metric)
        with open(self.path / 'metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=4)
        # printc(json.dumps(self.metrics, indent=4), color='GRASS')

        summary = self.pruning.summary()
        summary_path = self.path / 'masks_summary.csv'
        summary.to_csv(summary_path)
        # print(summary)

    def pruning_metrics(self):

        metrics = {}
        # Model Size
        size, size_nz = model_size(self.model)
        metrics['size'] = size
        metrics['size_nz'] = size_nz
        metrics['compression_ratio'] = size / size_nz

        x, y = next(iter(self.val_dl))
        x, y = x.to(self.device), y.to(self.device)

        # FLOPS
        ops, ops_nz = flops(self.model, x)
        metrics['flops'] = ops
        metrics['flops_nz'] = ops_nz
        metrics['theoretical_speedup'] = ops / ops_nz

        # Accuracy
        loss, acc1, acc5 = self.run_epoch(False, -1, after_pruning=True)
        self.log(steps=self.steps)
        self.log_epoch(-1)

        metrics['loss'] = loss
        metrics['val_acc1'] = acc1
        metrics['val_acc5'] = acc5

        return metrics
