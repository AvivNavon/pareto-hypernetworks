from typing import List

import torch
from torch import nn
import torch.nn.functional as F


class PHNHyper(nn.Module):
    """Modification of https://github.com/pytorch/examples/blob/master/mnist/main.py

    """

    def __init__(self, kernel_size: List[int], ray_hidden_dim=100, out_dim=10,
                 target_hidden_dim=50, n_kernels=10, n_conv_layers=2, n_hidden=1, n_tasks=2):
        super().__init__()
        self.n_conv_layers = n_conv_layers
        self.n_hidden = n_hidden
        self.n_tasks = n_tasks

        assert len(kernel_size) == n_conv_layers, "kernel_size is list with same dim as number of " \
                                                  "conv layers holding kernel size for each conv layer"

        self.ray_mlp = nn.Sequential(
            nn.Linear(2, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim)
        )

        self.conv_0_weights = nn.Linear(ray_hidden_dim, n_kernels * kernel_size[0] * kernel_size[0])
        self.conv_0_bias = nn.Linear(ray_hidden_dim, n_kernels)

        for i in range(1, n_conv_layers):
            # previous number of kernels
            p = 2 ** (i-1) * n_kernels
            # current number of kernels
            c = 2 ** i * n_kernels

            setattr(self, f"conv_{i}_weights", nn.Linear(ray_hidden_dim, c * p * kernel_size[i] * kernel_size[i]))
            setattr(self, f"conv_{i}_bias", nn.Linear(ray_hidden_dim,  c))

        latent = 25
        self.hidden_0_weights = nn.Linear(ray_hidden_dim, target_hidden_dim * 2 ** i * n_kernels * latent)
        self.hidden_0_bias = nn.Linear(ray_hidden_dim, target_hidden_dim)

        for j in range(n_tasks):
            setattr(self, f"task_{j}_weights", nn.Linear(ray_hidden_dim, target_hidden_dim * out_dim))
            setattr(self, f"task_{j}_bias", nn.Linear(ray_hidden_dim, out_dim))

    def shared_parameters(self):
        return list([p for n, p in self.named_parameters() if 'task' not in n])

    def forward(self, ray):
        features = self.ray_mlp(ray)

        out_dict = {}
        layer_types = ["conv", "hidden", "task"]

        for i in layer_types:
            if i == "conv":
                n_layers = self.n_conv_layers
            elif i == "hidden":
                n_layers = self.n_hidden
            elif i == "task":
                n_layers = self.n_tasks

            for j in range(n_layers):
                out_dict[f"{i}{j}.weights"] = getattr(self, f"{i}_{j}_weights")(features)
                out_dict[f"{i}{j}.bias"] = getattr(self, f"{i}_{j}_bias")(features).flatten()

        return out_dict


class PHNTarget(nn.Module):
    """Modification of https://github.com/pytorch/examples/blob/master/mnist/main.py

    """
    def __init__(self, kernel_size, n_kernels=10, out_dim=10, target_hidden_dim=50, n_conv_layers=2, n_tasks=2):
        super().__init__()
        assert len(kernel_size) == n_conv_layers, "kernel_size is list with same dim as number of " \
                                                  "conv layers holding kernel size for each conv layer"
        self.n_kernels = n_kernels
        self.kernel_size = kernel_size
        self.out_dim = out_dim
        self.n_conv_layers = n_conv_layers
        self.n_tasks = n_tasks
        self.target_hidden_dim = target_hidden_dim

    def forward(self, x, weights=None):
        x = F.conv2d(
            x, weight=weights['conv0.weights'].reshape(self.n_kernels, 1, self.kernel_size[0],
                                                       self.kernel_size[0]),
            bias=weights['conv0.bias'], stride=1
        )
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        for i in range(1, self.n_conv_layers):
            x = F.conv2d(
                x,
                weight=weights[f'conv{i}.weights'].reshape(int(2 ** i * self.n_kernels),
                                                           int(2 ** (i-1) * self.n_kernels),
                                                           self.kernel_size[i],
                                                           self.kernel_size[i]),
                bias=weights[f'conv{i}.bias'], stride=1
            )
            x = F.relu(x)
            x = F.max_pool2d(x, 2)

        x = torch.flatten(x, 1)

        x = F.linear(
            x,
            weight=weights["hidden0.weights"].reshape(self.target_hidden_dim, x.shape[-1]),
            bias=weights["hidden0.bias"]
        )

        logits = []
        for j in range(self.n_tasks):
            logits.append(
                F.linear(
                    x, weight=weights[f'task{j}.weights'].reshape(self.out_dim, self.target_hidden_dim),
                    bias=weights[f'task{j}.bias']
                )
            )
        return logits
