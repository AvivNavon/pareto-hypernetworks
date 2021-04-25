from typing import List

import torch
import torch.nn.functional as F
from torch import nn
import torchvision.models as models


class LeNetHyper(nn.Module):
    """LeNet Hypernetwork

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


class LeNetTarget(nn.Module):
    """LeNet target network

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


class ResnetHyper(nn.Module):

    def __init__(
            self, preference_dim=2, preference_embedding_dim=32, hidden_dim=100,
            num_chunks=105, chunk_embedding_dim=64, num_ws=11, w_dim=10000
    ):
        """

        :param preference_dim: preference vector dimension
        :param preference_embedding_dim: preference embedding dimension
        :param hidden_dim: hidden dimension
        :param num_chunks: number of chunks
        :param chunk_embedding_dim: chunks embedding dimension
        :param num_ws: number of W matrices (see paper for details)
        :param w_dim: row dimension of the W matrices
        """
        super().__init__()
        self.preference_embedding_dim = preference_embedding_dim
        self.num_chunks = num_chunks
        self.chunk_embedding_matrix = nn.Embedding(num_embeddings=num_chunks, embedding_dim=chunk_embedding_dim)
        self.preference_embedding_matrix = nn.Embedding(
            num_embeddings=preference_dim, embedding_dim=preference_embedding_dim
        )

        self.fc = nn.Sequential(
            nn.Linear(preference_embedding_dim + chunk_embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        list_ws = [self._init_w((w_dim, hidden_dim)) for _ in range(num_ws)]
        self.ws = nn.ParameterList(list_ws)

        # initialization
        torch.nn.init.normal_(self.preference_embedding_matrix.weight, mean=0., std=0.1)
        torch.nn.init.normal_(self.chunk_embedding_matrix.weight, mean=0., std=0.1)
        for w in self.ws:
            torch.nn.init.normal_(w, mean=0., std=0.1)

        self.layer_to_shape = {
            'resnet.conv1.weight': torch.Size([64, 1, 7, 7]),  # torch.Size([64, 3, 7, 7]),
            'resnet.bn1.weight': torch.Size([64]),
            'resnet.bn1.bias': torch.Size([64]),
            'resnet.layer1.0.conv1.weight': torch.Size([64, 64, 3, 3]),
            'resnet.layer1.0.bn1.weight': torch.Size([64]),
            'resnet.layer1.0.bn1.bias': torch.Size([64]),
            'resnet.layer1.0.conv2.weight': torch.Size([64, 64, 3, 3]),
            'resnet.layer1.0.bn2.weight': torch.Size([64]),
            'resnet.layer1.0.bn2.bias': torch.Size([64]),
            'resnet.layer1.1.conv1.weight': torch.Size([64, 64, 3, 3]),
            'resnet.layer1.1.bn1.weight': torch.Size([64]),
            'resnet.layer1.1.bn1.bias': torch.Size([64]),
            'resnet.layer1.1.conv2.weight': torch.Size([64, 64, 3, 3]),
            'resnet.layer1.1.bn2.weight': torch.Size([64]),
            'resnet.layer1.1.bn2.bias': torch.Size([64]),
            'resnet.layer2.0.conv1.weight': torch.Size([128, 64, 3, 3]),
            'resnet.layer2.0.bn1.weight': torch.Size([128]),
            'resnet.layer2.0.bn1.bias': torch.Size([128]),
            'resnet.layer2.0.conv2.weight': torch.Size([128, 128, 3, 3]),
            'resnet.layer2.0.bn2.weight': torch.Size([128]),
            'resnet.layer2.0.bn2.bias': torch.Size([128]),
            'resnet.layer2.0.downsample.0.weight': torch.Size([128, 64, 1, 1]),
            'resnet.layer2.0.downsample.1.weight': torch.Size([128]),
            'resnet.layer2.0.downsample.1.bias': torch.Size([128]),
            'resnet.layer2.1.conv1.weight': torch.Size([128, 128, 3, 3]),
            'resnet.layer2.1.bn1.weight': torch.Size([128]),
            'resnet.layer2.1.bn1.bias': torch.Size([128]),
            'resnet.layer2.1.conv2.weight': torch.Size([128, 128, 3, 3]),
            'resnet.layer2.1.bn2.weight': torch.Size([128]),
            'resnet.layer2.1.bn2.bias': torch.Size([128]),
            'resnet.layer3.0.conv1.weight': torch.Size([256, 128, 3, 3]),
            'resnet.layer3.0.bn1.weight': torch.Size([256]),
            'resnet.layer3.0.bn1.bias': torch.Size([256]),
            'resnet.layer3.0.conv2.weight': torch.Size([256, 256, 3, 3]),
            'resnet.layer3.0.bn2.weight': torch.Size([256]),
            'resnet.layer3.0.bn2.bias': torch.Size([256]),
            'resnet.layer3.0.downsample.0.weight': torch.Size([256, 128, 1, 1]),
            'resnet.layer3.0.downsample.1.weight': torch.Size([256]),
            'resnet.layer3.0.downsample.1.bias': torch.Size([256]),
            'resnet.layer3.1.conv1.weight': torch.Size([256, 256, 3, 3]),
            'resnet.layer3.1.bn1.weight': torch.Size([256]),
            'resnet.layer3.1.bn1.bias': torch.Size([256]),
            'resnet.layer3.1.conv2.weight': torch.Size([256, 256, 3, 3]),
            'resnet.layer3.1.bn2.weight': torch.Size([256]),
            'resnet.layer3.1.bn2.bias': torch.Size([256]),
            'resnet.layer4.0.conv1.weight': torch.Size([512, 256, 3, 3]),
            'resnet.layer4.0.bn1.weight': torch.Size([512]),
            'resnet.layer4.0.bn1.bias': torch.Size([512]),
            'resnet.layer4.0.conv2.weight': torch.Size([512, 512, 3, 3]),
            'resnet.layer4.0.bn2.weight': torch.Size([512]),
            'resnet.layer4.0.bn2.bias': torch.Size([512]),
            'resnet.layer4.0.downsample.0.weight': torch.Size([512, 256, 1, 1]),
            'resnet.layer4.0.downsample.1.weight': torch.Size([512]),
            'resnet.layer4.0.downsample.1.bias': torch.Size([512]),
            'resnet.layer4.1.conv1.weight': torch.Size([512, 512, 3, 3]),
            'resnet.layer4.1.bn1.weight': torch.Size([512]),
            'resnet.layer4.1.bn1.bias': torch.Size([512]),
            'resnet.layer4.1.conv2.weight': torch.Size([512, 512, 3, 3]),
            'resnet.layer4.1.bn2.weight': torch.Size([512]),
            'resnet.layer4.1.bn2.bias': torch.Size([512]),
            'resnet.fc.weight': torch.Size([512, 512]),
            'resnet.fc.bias': torch.Size([512]),
            'task1.weight': torch.Size([10, 512]),
            'task1.bias': torch.Size([10]),
            'task2.weight': torch.Size([10, 512]),
            'task2.bias': torch.Size([10]),
        }

    def _init_w(self, shapes):
        return nn.Parameter(torch.randn(shapes), requires_grad=True)

    def forward(self, preference):
        # preference embedding
        pref_embedding = torch.zeros((self.preference_embedding_dim, ), device=preference.device)
        for i, pref in enumerate(preference):
            pref_embedding += self.preference_embedding_matrix(
                torch.tensor([i], device=preference.device)
            ).squeeze(0) * pref
        # chunk embedding
        weights = []
        for chunk_id in range(self.num_chunks):
            chunk_embedding = self.chunk_embedding_matrix(torch.tensor([chunk_id], device=preference.device)).squeeze(0)
            # input to fc
            input_embedding = torch.cat((pref_embedding, chunk_embedding)).unsqueeze(0)
            # hidden representation
            rep = self.fc(input_embedding)

            weights.append(torch.cat([F.linear(rep, weight=w) for w in self.ws], dim=1))

        weight_vector = torch.cat(weights, dim=1).squeeze(0)

        out_dict = dict()
        position = 0
        for name, shapes in self.layer_to_shape.items():
            out_dict[name] = weight_vector[position:position+shapes.numel()].reshape(shapes)
            position += shapes.numel()
        return out_dict


class ResNetTarget(nn.Module):
    def __init__(self, pretrained=False, progress=True, **kwargs):
        super().__init__()
        self.resnet = models.resnet18(pretrained=pretrained, progress=progress, num_classes=512, **kwargs)

        self.resnet.conv1.weight.data = torch.randn((64, 1, 7, 7))

        self.task1 = nn.Linear(512, 10)
        self.task2 = nn.Linear(512, 10)

    def forward(self, x, weights=None):
        # pad input
        x = F.pad(input=x, pad=[0, 2, 0, 2], mode='constant', value=0.)

        if weights is None:
            x = self.resnet(x)
            x = F.relu(x)
            p1, p2 = self.task1(x), self.task2(x)
            return p1, p2

        else:
            x = self.forward_init(x, weights)
            x = self.forward_layer(x, weights, 1)
            x = self.forward_layer(x, weights, 2)
            x = self.forward_layer(x, weights, 3)
            x = self.forward_layer(x, weights, 4)
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
            x = self.forward_linear(x, weights)
            x = F.relu(x)
            p1 = self.forward_clf(x, weights, 1)
            p2 = self.forward_clf(x, weights, 2)

            return p1, p2

    @staticmethod
    def forward_init(x, weights):
        """Before blocks
        """
        device = x.device
        x = F.conv2d(x, weights['resnet.conv1.weight'], stride=2, padding=3)
        x = F.batch_norm(
            x,
            torch.zeros(x.data.size()[1]).to(device),
            torch.ones(x.data.size()[1]).to(device),
            weights['resnet.bn1.weight'],
            weights['resnet.bn1.bias'],
            training=True
        )
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1, dilation=1)

        return x

    def forward_block(self, x, weights, layer, index):
        if layer == 1:
            stride = 1
        else:
            stride = 2 if index == 0 else 1

        device = x.device
        identity = x

        # conv
        out = F.conv2d(x, weights[f'resnet.layer{layer}.{index}.conv1.weight'], stride=stride, padding=1)
        # bn
        out = F.batch_norm(
            out,
            torch.zeros(out.data.size()[1]).to(device),
            torch.ones(out.data.size()[1]).to(device),
            weights[f'resnet.layer{layer}.{index}.bn1.weight'],
            weights[f'resnet.layer{layer}.{index}.bn1.bias'],
            training=True
        )
        out = F.relu(out, inplace=True)
        # conv
        out = F.conv2d(out, weights[f'resnet.layer{layer}.{index}.conv2.weight'], stride=1, padding=1)
        # bn
        out = F.batch_norm(
            out,
            torch.zeros(out.data.size()[1]).to(device),
            torch.ones(out.data.size()[1]).to(device),
            weights[f'resnet.layer{layer}.{index}.bn2.weight'],
            weights[f'resnet.layer{layer}.{index}.bn2.bias'],
            training=True
        )

        if layer > 1 and index == 0:
            identity = self.forward_dowmsample(x, weights, layer)

        out += identity

        out = F.relu(out)

        return out

    @staticmethod
    def forward_dowmsample(x, weights, layer):
        device = x.device

        out = F.conv2d(x, weights[f'resnet.layer{layer}.0.downsample.0.weight'], stride=2)

        out = F.batch_norm(
            out,
            torch.zeros(out.data.size()[1]).to(device),
            torch.ones(out.data.size()[1]).to(device),
            weights[f'resnet.layer{layer}.0.downsample.1.weight'],
            weights[f'resnet.layer{layer}.0.downsample.1.bias'],
            training=True
        )
        return out

    def forward_layer(self, x, weights, layer):
        x = self.forward_block(x, weights, layer, 0)
        x = self.forward_block(x, weights, layer, 1)
        return x

    @staticmethod
    def forward_linear(x, weights):
        return F.linear(x, weights['resnet.fc.weight'], weights['resnet.fc.bias'])

    @staticmethod
    def forward_clf(x, weights, index):
        return F.linear(x, weights[f'task{index}.weight'], weights[f'task{index}.bias'])
