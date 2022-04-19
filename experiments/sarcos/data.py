import logging
from pathlib import Path

import torch
import numpy as np
from scipy.io import loadmat
from torch.utils.data import TensorDataset


def get_data(dataroot):
    path = Path(dataroot)
    # https://github.com/Kaixhin/SARCOS/blob/master/data.py
    train_data = loadmat(path / "sarcos_inv.mat")["sarcos_inv"].astype(np.float32)
    val_data, train_data = train_data[:4448], train_data[4448:]
    test_data = loadmat(path / "sarcos_inv_test.mat")["sarcos_inv_test"].astype(
        np.float32
    )

    X_train, Y_train = train_data[:, :21], train_data[:, 21:]
    X_val, Y_val = val_data[:, :21], val_data[:, 21:]
    X_test, Y_test = test_data[:, :21], test_data[:, 21:]

    quant = np.quantile(Y_train, q=0.9, axis=0)
    Y_train /= quant
    Y_val /= quant
    Y_test /= quant
    logging.info(
        f"training examples: {len(X_train)}, validation {len(X_val)}, test {len(X_test)}"
    )

    return (
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train)),
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val)),
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(Y_test)),
    )
