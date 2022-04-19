import pickle

import torch
from sklearn.model_selection import train_test_split


class Dataset:
    def __init__(self, path, val_size=0):
        self.path = path
        self.val_size = val_size

    def get_datasets(self):
        with open(self.path, "rb") as f:
            trainX, trainLabel, testX, testLabel = pickle.load(f)

        n_train = len(trainX)
        if self.val_size > 0:
            trainX, valX, trainLabel, valLabel = train_test_split(
                trainX, trainLabel, test_size=self.val_size, random_state=42
            )
            n_train = len(trainX)
            n_val = len(valX)

        trainX = torch.from_numpy(trainX.reshape(n_train, 1, 36, 36)).float()
        trainLabel = torch.from_numpy(trainLabel).long()
        testX = torch.from_numpy(testX.reshape(20000, 1, 36, 36)).float()
        testLabel = torch.from_numpy(testLabel).long()

        train_set = torch.utils.data.TensorDataset(trainX, trainLabel)
        test_set = torch.utils.data.TensorDataset(testX, testLabel)

        if self.val_size > 0:
            valX = torch.from_numpy(valX.reshape(n_val, 1, 36, 36)).float()
            valLabel = torch.from_numpy(valLabel).long()
            val_set = torch.utils.data.TensorDataset(valX, valLabel)

            return train_set, val_set, test_set

        return train_set, test_set
