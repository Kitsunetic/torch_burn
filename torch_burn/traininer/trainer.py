import math
from collections import defaultdict
import time
from multiprocessing import cpu_count
from typing import Iterable, Union, Tuple

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from torch_burn.callbacks import Callback, EarlyStopping
from torch_burn.data.utils import kfold
from torch_burn.metrics import Metric
from torch_burn.utils import seed_everything


class Trainer:
    def __init__(self,
                 model: nn.Module,
                 criterion: nn.Module,
                 optim: Optimizer,
                 metrics: Union[Metric, Iterable[Metric]] = None,
                 callbacks: Union[Callback, Iterable[Callback]] = None,
                 desc: str = '[{epoch:04d}/{num_epochs:04d}]',
                 data_parallel: bool = False,
                 gpus: int = torch.cuda.device_count(),
                 cpus: int = cpu_count(),
                 ncols: int = 100,
                 verbose: bool = True):
        self.model = model
        self.criterion = criterion
        self.optim = optim
        self.metrics = self.easy_list(metrics)
        self.callbacks = self.easy_list(callbacks)
        self.desc = desc
        self.data_parallel = data_parallel
        self.gpus = gpus
        self.cpus = cpus
        self.verbose = verbose
        self.ncols = ncols

        # sort callbacks along whose priorities
        if self.callbacks:
            self.callbacks = sorted(self.callbacks, key=lambda cb: cb.priority, reverse=True)

        # data parallel
        if self.data_parallel:
            self.model = nn.DataParallel(model)

    def fit(self,
            train_dataset: Dataset,
            valid_dataset: Dataset = None,
            train_valid_split: float = None,
            num_folds: int = None,
            fold: int = None,
            num_epochs: int = 1,
            start_epoch: int = 1,
            batch_size=32,
            valid_batch_size=None,
            shuffle=True,
            pin_memory=False,
            drop_last=False,
            seed=0):
        seed_everything(seed)
        train_ds, valid_ds = self._init_dataset(train_dataset, valid_dataset, train_valid_split, num_folds, fold, seed)
        train_dl, valid_dl = self._init_dataloader(train_ds, valid_ds,
                                                   batch_size, valid_batch_size,
                                                   shuffle, valid_shuffle=False,
                                                   pin_memory=pin_memory, drop_last=drop_last)

        # logs - average metric value of each epochs
        self.stop_loop = False
        for epoch in range(start_epoch, num_epochs + 1):
            if self.stop_loop:
                break
            logs = self._init_logs()

            # losses - metric value of each batches
            losses = defaultdict(float)

            # train callbacks
            with torch.no_grad():
                for cb in self.callbacks:
                    cb.on_train_epoch_begin(epoch)
                for m in self.metrics:
                    m.on_train_epoch_begin(epoch)

            # train loop
            self.model.train()

            if self.verbose:
                t = tqdm(total=len(train_dl), ncols=self.ncols,
                         desc=self.desc.format(epoch=epoch, num_epochs=num_epochs) + ' Train')

            for batch_idx, data in enumerate(train_dl):
                # train batch callbacks
                with torch.no_grad():
                    for cb in self.callbacks:
                        cb.on_train_batch_begin(epoch, batch_idx)

                # forward / backward
                x, pred, y = self.loop(batch_idx, data, is_train=True, losses=losses, logs=logs)

                # Calculate metrics
                with torch.no_grad():
                    self.calculate_metrics(pred, y, losses, logs, batch_idx, is_train=True)

                # Update progressbar
                if self.verbose:
                    msgs = []
                    for k, v in losses.items():
                        msgs.append(f'{k} {v:.4f}')
                    """
                    for m in self.metrics:
                        if m.visible:
                            msgs.append(f'{m.name} {logs[m.name]:.4f}')
                    """
                    msg = ' '.join(msgs)
                    t.set_postfix_str(msg, refresh=False)
                    t.update()

                # Train batch callbacks
                with torch.no_grad():
                    for cb in self.callbacks:
                        cb.on_train_batch_end(epoch, batch_idx, losses, x, y, pred)

            if self.verbose:
                t.close()
                time.sleep(0.01)  # wait for tqdm closed

            # train epoch callbacks
            with torch.no_grad():
                for cb in self.callbacks:
                    cb.on_train_epoch_end(epoch, logs)
                    cb.on_train_epoch_end_with_data(epoch, logs, x, y, pred)
                for m in self.metrics:
                    m.on_train_epoch_end(epoch, logs)

            # Without valid dataset, there is no valid loop.
            if valid_ds is None:
                continue

            # valid loop
            losses = defaultdict(float)
            self.model.eval()
            with torch.no_grad():
                # valid callbacks
                for cb in self.callbacks:
                    cb.on_valid_epoch_begin(epoch)
                for m in self.metrics:
                    m.on_valid_epoch_begin(epoch)

                if self.verbose:
                    t = tqdm(total=len(valid_dl), ncols=self.ncols,
                             desc=self.desc.format(epoch=epoch, num_epochs=num_epochs) + ' Validation')

                for batch_idx, data in enumerate(valid_dl):
                    # train batch callbacks
                    for cb in self.callbacks:
                        cb.on_valid_batch_begin(epoch, batch_idx)

                    # forward
                    x, pred, y = self.loop(batch_idx, data, is_train=False, losses=losses, logs=logs)

                    # Calculate metrics
                    self.calculate_metrics(pred, y, losses, logs, batch_idx, is_train=False)

                    # Update progressbar
                    if self.verbose:
                        msgs = []
                        for k, v in losses.items():
                            msgs.append(f'{k} {v:.4f}')
                        """
                        for m in self.metrics:
                            if m.visible:
                                name = 'val_' + m.name
                                msgs.append(f'{name} {logs[name]:.4f}')
                        """
                        msg = ' '.join(msgs)
                        t.set_postfix_str(msg, refresh=False)
                        t.update()

                    # train batch callbacks
                    for cb in self.callbacks:
                        cb.on_valid_batch_end(epoch, batch_idx, losses, x, y, pred)

                # Wait for tqdm closed
                if self.verbose:
                    t.close()
                    time.sleep(0.01)  # Wait for tqdm closed

                # Validation epoch callbacks
                for cb in self.callbacks:
                    cb.on_valid_epoch_end(epoch, logs)
                    cb.on_valid_epoch_end_with_data(epoch, logs, x, y, pred)
                    if isinstance(cb, EarlyStopping):
                        if cb.stopped:
                            self.stop_loop = True

                for m in self.metrics:
                    m.on_valid_epoch_end(epoch, logs)

    def _init_dataset(self,
                      train_dataset: Dataset,
                      valid_dataset: Dataset = None,
                      train_valid_split: float = None,
                      num_folds: int = None,
                      fold: int = None,
                      seed: int = 0):
        if valid_dataset is not None:
            return train_dataset, valid_dataset
        elif (train_valid_split is not None) and (0 < train_valid_split < 1):
            # train-valid split
            s = len(train_dataset)
            v = int(s * train_valid_split)
            t = s - v
            return random_split(train_dataset, (t, v))
        elif num_folds is not None and fold is not None:
            # k-fold
            return kfold(train_dataset, num_folds, fold, seed)
        elif num_folds is not None or fold is not None:
            raise NotImplementedError('Both num_folds and fold must be specified')
        else:
            # no validation
            return train_dataset, None

    def _init_dataloader(self,
                         train_dataset: Dataset,
                         valid_dataset: Dataset = None,
                         batch_size=32,
                         valid_batch_size=None,
                         shuffle=True,
                         valid_shuffle=False,
                         pin_memory=True,
                         drop_last=False):
        valid_batch_size = valid_batch_size or batch_size

        train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
                              num_workers=self.cpus, pin_memory=pin_memory, drop_last=drop_last)

        if valid_dataset is not None:
            valid_dl = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=valid_shuffle,
                                  num_workers=self.cpus, pin_memory=pin_memory, drop_last=drop_last)
        else:
            valid_dl = None

        return train_dl, valid_dl

    def _init_logs(self):
        logs = {}
        for m in self.metrics:
            if m.visible:
                logs[m.name] = math.inf if m.mode == 'min' else -math.inf
                logs['val_' + m.name] = math.inf if m.mode == 'min' else -math.inf
        return logs

    def forward(self, data, is_train: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = data[:2]
        x, y = self.cuda(x, y)
        return x, self.model(x), y

    def backward(self, x: torch.Tensor, pred: torch.Tensor, y: torch.Tensor, is_train: bool) -> float:
        if is_train:
            self.optim.zero_grad()
            loss = self.criterion(pred, y)
            loss.backward()
            self.optim.step()
        else:
            loss = self.criterion(pred, y)

        return loss.item()

    def loop(self, batch_idx, data, is_train: bool, losses: dict, logs: dict):
        x, pred, y = self.forward(data, is_train=is_train)
        loss = self.backward(x, pred, y, is_train=is_train)

        name = 'loss' if is_train else 'val_loss'
        losses[name] = loss
        logs[name] = self._ignition_mean(logs[name], loss, batch_idx)

        return x, pred, y

    def calculate_metrics(self, pred, y, losses, logs, batch_idx, is_train: bool):
        for m in self.metrics:
            v = m.get_value(pred, y, is_train=is_train)
            if m.visible:
                if isinstance(v, torch.Tensor):
                    v = v.item()
                name = 'val_' + m.name
                losses[name] = v
                logs[name] = self._ignition_mean(logs[name], v, batch_idx)

    def cuda(self, *tensors):
        if self.gpus > 0:
            if len(tensors) == 1:
                return tensors[0].cuda()
            else:
                res = []
                for t in tensors:
                    if isinstance(t, torch.Tensor):
                        res.append(t.cuda())
                return res
        else:
            return tensors

    @staticmethod
    def easy_list(items):
        if items is None:
            return []
        elif isinstance(items, Iterable):
            return list(items)
        else:
            return [items]

    @staticmethod
    def _ignition_mean(a, b, i):
        if math.isinf(a):
            return b
        else:
            return (a * i + b) / (i + 1)
