import time
from multiprocessing import cpu_count

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class Predictor:
    def __init__(self, model: nn.Module, cpus=cpu_count(), gpus=torch.cuda.device_count(), verbose=True):
        self.model = model
        self.cpus = cpus
        self.gpus = gpus
        self.verbose = verbose

    def forward(self, data):
        x = data
        if self.gpus:
            x = x.cuda()
        preds = self.model(x).detach().cpu()
        return preds

    def on_predict_begin(self):
        pass

    def on_predict_end(self):
        pass

    def predict(self, dataset: Dataset, batch_size=16, shuffle=False,
                verbose=None, desc='Prediction', ncols=100):
        verbose = verbose or self.verbose

        self.on_predict_begin()

        dl = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=self.cpus)

        t = None
        if verbose:
            t = tqdm(total=len(dl), ncols=ncols, desc=desc)

        rets = []
        for data in dl:
            ret = self.forward(data)
            rets.append(ret)

            if verbose:
                t.update()
        ret = torch.cat(rets)

        if verbose:
            t.close()
            time.sleep(0.001)

        self.on_predict_end()
        return ret
