import torch
import torch.nn as nn


class Metric:
    def __init__(self, name: str, mode='min', visible=True):
        mode = mode.lower()
        assert mode in ['min', 'max']

        self.name = name
        self.mode = mode
        self.visible = visible

    def on_train_epoch_begin(self, epoch: int):
        pass

    def on_train_epoch_end(self, epoch: int, logs: dict):
        pass

    def on_valid_epoch_begin(self, epoch: int):
        pass

    def on_valid_epoch_end(self, epoch: int, logs: dict):
        pass

    def get_value(self, outputs: torch.Tensor, targets: torch.Tensor, is_train: bool):
        pass


class InvisibleMetric(Metric):
    def __init__(self, name: str, mode='min'):
        super(InvisibleMetric, self).__init__(name=name, mode=mode, visible=False)


class ModuleMetric(Metric):
    def __init__(self, module: nn.Module, name: str, mode='min', visible=True):
        super(ModuleMetric, self).__init__(name, mode)
        self.module = module

    def get_value(self, outputs: torch.Tensor, targets: torch.Tensor, is_train):
        loss = self.module(outputs, targets)
        return loss
