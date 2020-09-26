import torch
import torch.nn as nn


class Metric:
    def __init__(self, name: str, mode='min'):
        mode = mode.lower()
        assert mode in ['min', 'max']

        self.name = name
        self.mode = mode

    def __call__(self, outputs: torch.tensor, targets: torch.tensor):
        pass


class ModuleMetric(Metric):
    def __init__(self, name: str, module: nn.Module, mode='min'):
        super(ModuleMetric, self).__init__(name, mode)
        self.module = module

    def __call__(self, outputs: torch.tensor, targets: torch.tensor):
        loss = self.module(outputs, targets)
        return loss
