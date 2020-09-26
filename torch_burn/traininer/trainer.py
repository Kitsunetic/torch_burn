import time
from multiprocessing import cpu_count
from typing import Iterable, Union

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from torch_burn.callbacks import Callback
from torch_burn.metrics import Metric


class Trainer:
    def __init__(self,
                 model: nn.Module,
                 criterion: nn.Module,
                 optim: Union[Optimizer, Iterable[Optimizer]],
                 metrics: Iterable[Metric],
                 callbacks: Iterable[Callback] = None,
                 train_dataset: Dataset = None,
                 valid_dataset: Dataset = None,
                 batch_size: int = 1,
                 num_epochs: int = 1,
                 start_epoch: int = 1,
                 train_valid_split: float = None,
                 shuffle: bool = False,
                 num_cpus: int = cpu_count(),
                 drop_last: bool = False,
                 verbose: bool = True,
                 desc: str = '[{epoch:04d}/{num_epochs:04d}]',
                 ncols: int = 100):
        self.model = model
        self.criterion = criterion
        self.optim = list(optim) if isinstance(optim, Iterable) else [optim]
        self.metrics = list(metrics)
        self.callbacks = list(callbacks)
        self.num_epochs = num_epochs
        self.start_epoch = start_epoch
        self.verbose = verbose
        self.desc = desc
        self.ncols = ncols

        assert valid_dataset is None or train_valid_split is None
        if valid_dataset is not None:
            self.train_dataset = train_dataset
            self.valid_dataset = valid_dataset
        elif train_valid_split is not None:
            assert 0 < train_valid_split < 1

            dssize = len(train_dataset)
            valid_size = int(dssize * train_valid_split)
            train_size = dssize - valid_size
            self.train_dataset, self.valid_dataset = random_split(train_dataset, (train_size, valid_size))
        else:
            self.train_dataset = train_dataset
            self.valid_dataset = None

        self.train_dl = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=shuffle,
                                   num_workers=num_cpus, drop_last=drop_last)
        self.valid_dl = None
        if self.valid_dataset is not None:
            self.valid_dl = DataLoader(self.train_dataset, batch_size=batch_size,
                                       num_workers=num_cpus, drop_last=drop_last)

        self._device = next(iter(model.parameters()))[0].device
        self.logs = {metric.name: [] for metric in self.metrics}

    def forward(self, data):
        x, y = data
        return self.model(x), y

    def preprocessing(self, data, map_location: torch.device = None):
        """Preprocessing input data for both train and validation"""
        if map_location is None:
            map_location = self._device

        if isinstance(data, torch.Tensor):
            data = data.to(map_location)
            return data
        elif isinstance(data, Iterable):
            new_data = []
            for d in data:
                if isinstance(d, torch.Tensor):
                    new_data.append(d.to(map_location))
                else:
                    new_data.append(d)
            return new_data
        else:
            return data

    def train_preprocessing(self, data, map_location: torch.device = None):
        """Specify preprocessing pipeline only for train"""
        return self.preprocessing(data, map_location=map_location)

    def valid_preprocessing(self, data, map_location: torch.device = None):
        """Specify preprocessing pipeline only for validation"""
        return self.preprocessing(data, map_location=map_location)

    def train(self):
        for epoch in range(self.start_epoch, self.num_epochs):
            pass

    def _train_loop(self, epoch: int):
        for callback in self.callbacks:
            callback.on_epoch_begin(True, epoch, self.logs)

        losses = []
        desc = self.desc.format(epoch=epoch, num_epochs=self.num_epochs) + ' Train'
        with tqdm(total=len(self.train_dl), ncols=self.ncols, desc=desc) as t:
            for data in self.train_dl:
                data = self.train_preprocessing(data)

                self.model.train()
                output, target = self.forward(data)

                loss = self.criterion(output, target)
                for optim in self.optim:
                    optim.zero_grad()
                loss.backward()
                for optim in self.optim:
                    optim.step()

                losses.append(loss.item())
                mean_loss = sum(losses[-100:]) / len(loss[-100:])
                t.set_postfix_str(f'loss {mean_loss:.4f}', refresh=False)
                t.update()
        time.sleep(0.001)  # for tqdm timing problem

    def _valid_loop(self, epoch: int):
        with torch.no_grad():
            self.model.eval()
            desc = self.desc.format(epoch=epoch, num_epochs=self.num_epochs) + ' Validation'
            with tqdm(total=len(self.valid_dl), ncols=self.ncols, desc=desc) as t:
                pass

            for data in dl:
                self.train_preprocessing(*data)

            time.sleep(0.001)

        pass
