from torch import nn
import torch.nn.functional as F


class FCNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(21, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 7),
        )

    def forward(self, x):
        return self.fc(x)


class HyperFCNet(nn.Module):
    def __init__(self, ray_hidden_dim=100):
        super().__init__()
        self.ray_mlp = nn.Sequential(
            nn.Linear(7, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
        )

        self.in_dim = 21
        self.dims = [256, 256, 256, 7]

        prvs_dim = self.in_dim
        for i, dim in enumerate(self.dims):
            setattr(self, f"fc_{i}_weights", nn.Linear(ray_hidden_dim, prvs_dim * dim))
            setattr(self, f"fc_{i}_bias", nn.Linear(ray_hidden_dim, dim))
            prvs_dim = dim

    def forward(self, ray):
        out_dict = dict()
        features = self.ray_mlp(ray)

        prvs_dim = self.in_dim
        for i, dim in enumerate(self.dims):
            out_dict[f"fc_{i}_weights"] = self.__getattr__(f"fc_{i}_weights")(
                features
            ).reshape(dim, prvs_dim)
            out_dict[f"fc_{i}_bias"] = self.__getattr__(f"fc_{i}_bias")(
                features
            ).flatten()
            prvs_dim = dim

        return out_dict


class TargetFCNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weights):

        for i in range(int(len(weights) / 2)):
            x = F.linear(x, weights[f"fc_{i}_weights"], weights[f"fc_{i}_bias"])
            if i < int(len(weights) / 2) - 1:
                x = F.relu(x)
        return x
