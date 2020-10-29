import math
import random
from pathlib import Path
from typing import AnyStr
from typing import Iterable, List, Union
from typing import Tuple

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from torch_burn.metrics import Metric


class Callback:
    # multiple callbacks are executed along the priority
    priority = 100

    def on_train_epoch_begin(self, epoch: int):
        """Event when train epoch begin"""
        pass

    def on_train_epoch_end(self, epoch: int, logs: dict):
        """Event when train epoch end"""
        pass

    def on_train_epoch_end_with_data(self, epoch: int, logs: dict,
                                     inputs: torch.Tensor, targets: torch.Tensor, preds: torch.Tensor):
        """Event when validation epoch end"""
        pass

    def on_valid_epoch_begin(self, epoch: int):
        """Event when validation epoch begin"""
        pass

    def on_valid_epoch_end(self, epoch: int, logs: dict):
        """Event when validation epoch end"""
        pass

    def on_valid_epoch_end_with_data(self, epoch: int, logs: dict,
                                     inputs: torch.Tensor, targets: torch.Tensor, preds: torch.Tensor):
        """Event when validation epoch end"""
        pass

    def on_train_batch_begin(self, epoch: int, batch_idx: int):
        """Event when train batch begin"""
        pass

    def on_train_batch_end(self, epoch: int, batch_idx: int, losses: dict,
                           inputs: torch.Tensor, targets: torch.Tensor, preds: torch.Tensor):
        """Event when train batch end"""
        pass

    def on_valid_batch_begin(self, epoch: int, batch_idx: int):
        """Event when validation batch begin"""
        pass

    def on_valid_batch_end(self, epoch: int, batch_idx: int, losses: dict,
                           inputs: torch.Tensor, targets: torch.Tensor, preds: torch.Tensor):
        """Event when validation batch end"""
        pass

    def on_fit_begin(self):
        """Event when training started"""
        pass

    def on_fit_end(self, epoch: int):
        """Event when training finished"""
        pass


class MetricImprovingCallback(Callback):
    def __init__(self, monitor: Metric, minimum_difference=0):
        self.monitor = monitor
        self.minimum_difference = minimum_difference

        self.best_metric_value = math.inf
        if self.monitor.mode == 'max':
            self.best_metric_value *= -1

    def on_valid_epoch_end(self, epoch: int, logs: dict):
        metric_name, metric_value = self.get_metric_info(logs)
        condition1 = (self.monitor.mode == 'max' and self.best_metric_value - metric_value < self.minimum_difference)
        condition2 = (self.monitor.mode == 'min' and self.best_metric_value - metric_value > self.minimum_difference)
        if condition1 or condition2:
            self.on_metric_improved(epoch, logs, metric_name, metric_value)
            self.best_metric_value = metric_value
        else:
            self.on_metric_not_improved(epoch, logs, metric_name, metric_value)

    def get_metric_info(self, logs: dict):
        metric_name = 'val_' + self.monitor.name
        assert metric_name in logs, f'There is no metric value in logs: {metric_name}'
        metric_value = logs[metric_name]
        return metric_name, metric_value

    def on_metric_improved(self, epoch: int, logs: dict, metric_name: str, metric_value: float):
        pass

    def on_metric_not_improved(self, epoch: int, logs: dict, metric_name: str, metric_value: float):
        pass


class SaveCheckpoint(MetricImprovingCallback):
    def __init__(self, checkpoint_spec: dict,
                 monitor: Metric,
                 save_dir: AnyStr,
                 filepath: AnyStr = 'ckpt-epoch{epoch:04d}-val_loss{val_loss:.4f}.pth',
                 save_best_only=True, verbose=True,
                 minimum_difference=0):
        super(SaveCheckpoint, self).__init__(monitor, minimum_difference)

        self.checkpoint_spec = checkpoint_spec
        self.filepath = Path(save_dir) / filepath
        self.save_best_only = save_best_only
        self.verbose = verbose

        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def on_valid_epoch_end(self, epoch: int, logs: dict):
        if self.save_best_only:
            super().on_valid_epoch_end(epoch, logs)
        else:
            metric_name, metric_value = self.get_metric_info(logs)
            self.on_metric_improved(epoch, logs, metric_name, metric_value)

    def on_metric_improved(self, epoch: int, logs: dict, metric_name: str, metric_value: float):
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
            if isinstance(self.checkpoint_spec[k], nn.DataParallel):
                data[k] = v.module.state_dict()
            elif isinstance(self.checkpoint_spec[k], nn.Module) or isinstance(self.checkpoint_spec[k], Optimizer):
                data[k] = v.state_dict()
            else:
                data[k] = v
        torch.save(data, filepath)


class EarlyStopping(MetricImprovingCallback):
    priority = 1
    stopped = False

    def __init__(self, monitor: Union[Metric, List[Metric]], patience=10, verbose=True, minimum_difference=0):
        if isinstance(monitor, Iterable):
            monitor = tuple(monitor)[0]

        super(EarlyStopping, self).__init__(monitor, minimum_difference)

        self.patience = patience
        self.verbose = verbose

        self.stopping_cnt = 0

    def on_metric_not_improved(self, epoch: int, logs: dict, metric_name: str, metric_value: float):
        self.stopping_cnt += 1
        print('val_loss is not improved for', self.stopping_cnt, 'epochs')
        if self.stopping_cnt >= self.patience:
            if self.verbose:
                metric_name, metric_value = super().get_metric_info(logs)
                print('Stop training because', metric_name, 'did not improved for', self.patience, 'epochs')
            self.stopped = True

    def on_metric_improved(self, epoch: int, logs: dict, metric_name: str, metric_value: float):
        self.stopping_cnt = 0


class LRDecaying(MetricImprovingCallback):
    def __init__(self, optim: Optimizer, monitor: Metric,
                 patience=5, decay_rate=0.5, verbose=True, minimum_difference=0):
        super(LRDecaying, self).__init__(monitor, minimum_difference)

        self.optim = optim
        self.patience = patience
        self.decay_rate = decay_rate
        self.verbose = verbose

        # self.lr = next(iter(self.optim.param_groups.values()))['lr']
        self.lr = self.optim.param_groups[0]['lr']

        self.decaying_cnt = 0

    def on_metric_not_improved(self, epoch: int, logs: dict, metric_name: str, metric_value: float):
        self.decaying_cnt += 1
        if self.decaying_cnt >= self.patience:
            self.decaying_cnt = 0
            new_lr = self.lr * self.decay_rate
            if self.verbose:
                metric_name, metric_value = self.get_metric_info(logs)
                print('Decaying lr from', self.lr, 'to', new_lr,
                      'because', metric_name, 'did not improved for', self.patience, 'epochs')

            for p in self.optim.param_groups:
                p['lr'] = new_lr
            self.lr = new_lr

    def on_metric_improved(self, epoch: int, logs: dict, metric_name: str, metric_value: float):
        self.decaying_cnt = 0


class Tensorboard(Callback):
    def __init__(self,
                 logdir: AnyStr,
                 model: nn.Module = None,
                 sample_input: torch.Tensor = None,
                 comment: str = '',
                 gpus=torch.cuda.device_count()):
        self.writer = SummaryWriter(logdir, comment=comment)

        if model is not None and sample_input is not None:
            with torch.no_grad():
                model.eval()

                sample_input = sample_input.unsqueeze(0)
                if gpus > 0:
                    sample_input = sample_input.cuda()
                self.writer.add_graph(model, sample_input)

    def on_batch_end(self, epoch: int, losses: dict, logs: dict):
        for k, v in losses.items():
            self.writer.add_scalar(k, v, epoch)


class SaveSample(Callback):
    def __init__(self,
                 save_dir: AnyStr,
                 output_filename: AnyStr = 'sample-epoch{epoch:04d}-val_loss{val_loss:.4f}.png',
                 input_filename: AnyStr = None,
                 target_filename: AnyStr = None,
                 verbose=True):
        super(SaveSample, self).__init__()

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.output_filename = str(save_dir / output_filename)
        self.input_filename = None
        self.target_filename = None
        self.verbose = verbose

        if input_filename is not None:
            self.input_filename = str(save_dir / input_filename)
        if target_filename is not None:
            self.target_filename = str(save_dir / target_filename)

        self.last_epoch = -1

    def on_valid_epoch_end_with_data(self, epoch: int, logs: dict,
                                     inputs: torch.Tensor, targets: torch.Tensor, preds: torch.Tensor):
        x = inputs[0].detach().cpu()
        y = targets[0].detach().cpu()
        p = preds[0].detach().cpu()

        if self.last_epoch != epoch: # Is it first callback of the epoch
            self.last_epoch = epoch

            self.save_output(p, self.output_filename.format(epoch=epoch, **logs))
            if self.input_filename is not None:
                self.save_input(x, self.output_filename.format(epoch=epoch, **logs))
            if self.target_filename is not None:
                self.save_target(y, self.target_filename.format(epoch=epoch, **logs))

    def save_output(self, data: torch.Tensor, path: str):
        raise NotImplementedError()

    def save_input(self, data: torch.Tensor, path: str):
        raise NotImplementedError()

    def save_target(self, data: torch.Tensor, path: str):
        raise NotImplementedError()
