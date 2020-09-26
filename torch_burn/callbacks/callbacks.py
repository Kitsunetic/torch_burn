import math
from pathlib import Path
from typing import AnyStr

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from torch_burn.metrics import Metric


class CallBack:
    def __init__(self):
        pass

    def on_epoch_begin(self, is_train: bool, epoch: int, logs: dict):
        pass

    def on_epoch_end(self, is_train: bool, epoch: int, logs: dict):
        pass

    def on_batch_begin(self, is_train: bool, epoch: int, logs: dict, inputs: torch.tensor):
        pass

    def on_batch_end(self, is_train: bool, epoch: int, logs: dict, inputs: torch.tensor, outputs: torch.tensor):
        pass


class MetricImprovingCallback(CallBack):
    def __init__(self, monitor: Metric):
        super(MetricImprovingCallback, self).__init__()

        self.monitor = monitor

        self.best_metric_value = math.inf
        if self.monitor.mode == 'max':
            self.best_metric_value *= -1

    def on_epoch_end(self, is_train: bool, epoch: int, logs: dict):
        if not is_train:
            metric_name, metric_value = self.get_metric_info(logs)
            condition1 = (self.monitor.mode == 'max' and self.best_metric_value < metric_value)
            condition2 = (self.monitor.mode == 'min' and self.best_metric_value > metric_value)
            if condition1 or condition2:
                self.on_metric_improved(is_train, epoch, logs, metric_name, metric_value)
                self.best_metric_value = metric_value
            else:
                self.on_metric_not_improved(is_train, epoch, logs, metric_name, metric_value)

    def get_metric_info(self, logs: dict):
        metric_name = 'val_' + self.monitor.name
        assert metric_name in logs, f'There is no metric value in logs: {metric_name}'
        metric_value = logs[metric_name]
        return metric_name, metric_value

    def on_metric_improved(self, is_train: bool, epoch: int, logs: dict, metric_name: str, metric_value: float):
        pass

    def on_metric_not_improved(self, is_train: bool, epoch: int, logs: dict, metric_name: str, metric_value: float):
        pass


class SaveCheckpoint(MetricImprovingCallback):
    def __init__(self, checkpoint_spec: dict,
                 filepath: AnyStr, monitor: Metric,
                 save_best_only=True, verbose=True):
        super(SaveCheckpoint, self).__init__(monitor)

        self.checkpoint_spec = checkpoint_spec
        self.filepath = Path(filepath)
        self.save_best_only = save_best_only
        self.verbose = verbose

    def on_epoch_end(self, is_train: bool, epoch: int, logs: dict):
        if not is_train:
            if self.save_best_only:
                super().on_epoch_end(is_train, epoch, logs)
            else:
                metric_name, metric_value = self.get_metric_info(logs)
                self.on_metric_improved(is_train, epoch, logs, metric_name, metric_value)

    def on_metric_improved(self, is_train: bool, epoch: int, logs: dict, metric_name: str, metric_value: float):
        if not is_train:
            if self.verbose:
                text = 'Save checkpoint: '
                text += metric_name
                text += ' decreased ' if self.monitor.mode == 'min' else ' increased '
                text += f'from {self.best_metric_value} to {metric_value}'
                print(text)
            self.best_metric_value = metric_value

            filepath = str(self.filepath).format(epoch=epoch, **logs)
            data = {}
            for k, v in self.checkpoint_spec.items():
                vtype = type(self.checkpoint_spec[k])
                if issubclass(vtype, nn.DataParallel):
                    data[k] = v.module.state_dict()
                elif issubclass(vtype, nn.Module) or issubclass(vtype, Optimizer):
                    data[k] = v.state_dict()
                else:
                    data[k] = v
            torch.save(data, filepath)


class SaveSampleBase(CallBack):
    """
    Save one sample per a epoch
    Must be specified how to save sample data through `save_data` overriding function.
    """

    def __init__(self, model: nn.Module, sample_input: torch.Tensor, filepath: AnyStr, verbose=False):
        """

        :param model: model which already uploaded on GPU if you are tending to use GPU
        :param sample_input: an input tensor data which could be directly fed into the model
        :param filepath: adaptive filepath string
        :param verbose:
        """
        super(SaveSampleBase, self).__init__()
        self.model = model
        self.sample_input = sample_input
        self.filepath = Path(filepath)
        self.verbose = verbose

    def on_epoch_end(self, is_train: bool, epoch: int, logs: dict):
        if not is_train:
            with torch.no_grad():
                filepath = str(self.filepath).format(epoch=epoch, **logs)
                device = next(self.model.parameters()).device
                x = self.sample_input.to(device)
                out = self.model(x)

                if self.verbose:
                    print('Write sample', Path(filepath).name)
                self.save_data(out, filepath)

    def save_data(self, output: torch.tensor, filepath: str):
        raise NotImplementedError('SaveSampleBase is abstract class which must be inherited')


class EarlyStopping(MetricImprovingCallback):
    def __init__(self, monitor: Metric, patience=10, verbose=False):
        super(EarlyStopping, self).__init__(monitor)

        self.patience = patience
        self.verbose = verbose

        self.stopping_cnt = 0

    def on_metric_not_improved(self, is_train: bool, epoch: int, logs: dict, metric_name: str, metric_value: float):
        if not is_train:
            self.stopping_cnt += 1
            if self.stopping_cnt >= self.patience:
                if self.verbose:
                    metric_name, metric_value = super().get_metric_info(logs)
                    print('Stop training because', metric_name, 'did not improved for', self.patience, 'epochs')

    def on_metric_improved(self, is_train: bool, epoch: int, logs: dict, metric_name: str, metric_value: float):
        self.stopping_cnt = 0


class LRDecaying(MetricImprovingCallback):
    def __init__(self, optim: Optimizer, monitor: Metric, patience=5, decay_rate=0.5, verbose=False):
        super(LRDecaying, self).__init__(monitor)

        self.optim = optim
        self.patience = patience
        self.decay_rate = decay_rate
        self.verbose = verbose

        # self.lr = next(iter(self.optim.param_groups.values()))['lr']
        self.lr = self.optim.param_groups[0]['lr']

        self.decaying_cnt = 0

    def on_metric_not_improved(self, is_train: bool, epoch: int, logs: dict, metric_name: str, metric_value: float):
        if not is_train:
            self.decaying_cnt += 1
            if self.decaying_cnt >= self.patience:
                self.decaying_cnt = 0
                new_lr = self.lr * self.decay_rate
                if self.verbose:
                    metric_name, metric_value = super().get_metric_info(logs)
                    print('Decaying lr from', self.lr, 'to', new_lr,
                          'because', metric_name, 'did not improved for', self.patience, 'epochs')

                for p in self.optim.param_groups:
                    p['lr'] = new_lr
                self.lr = new_lr

    def on_metric_improved(self, is_train: bool, epoch: int, logs: dict, metric_name: str, metric_value: float):
        self.decaying_cnt = 0
