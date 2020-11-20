from typing import Tuple

import numpy as np
from torch.utils.data import Dataset, Subset

from torch_burn.utils import seed_everything


def kfold(ds: Dataset, k: int, fold: int, seed: int) -> Tuple[Dataset, Dataset]:
    """
    데이터셋을 k개로 잘라서 많은 쪽을 train, 작은 쪽을 test 데이터셋으로 분할합니다.
    Parameters
    ----------
    ds : 입력 데이터셋
    k : 폴드의 개수
    fold : 몇 번째 폴드인지

    Returns
    -------
    나눠진 두 개의 train, test 데이터셋
    """
    L = len(ds)

    assert 0 <= fold < k
    assert k > 1
    assert L >= k

    seed_everything(seed)
    perm = np.random.permutation(L)
    idxlist = []
    P = L // k
    for i in range(k - 1):
        idxlist.append(perm[P * i:P * (i + 1)])
    idxlist.append(perm[P * (k - 1):])

    y = Subset(ds, idxlist[fold])
    del idxlist[fold]
    x = Subset(ds, np.concatenate(idxlist))

    return x, y


class ChainDataset(Dataset):
    def __init__(self, *ds_list: Dataset):
        """
        Combine multiple dataset into one.

        Parameters
        ----------
        ds_list: list of datasets
        """
        self.ds_list = ds_list
        self.len_list = [len(ds) for ds in self.ds_list]
        self.total_len = sum(self.len_list)

        self.idx_list = []
        for i, l in enumerate(self.len_list):
            self.idx_list.extend([(i, j) for j in range(l)])

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        didx, sidx = self.idx_list[idx]
        return self.ds_list[didx][sidx]
