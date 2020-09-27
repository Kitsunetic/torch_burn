from torch.utils.data import Dataset


def kfold(ds: Dataset, n: int, fold: int):
    assert 0 < fold < n
    assert n > 1
    assert len(ds) >= n

    idx1, idx2 = [], []
    for i in range(len(ds)):
        if i % n == fold:
            idx2.append(i)
        else:
            idx1.append(i)

    return ds[idx1], ds[idx2]
