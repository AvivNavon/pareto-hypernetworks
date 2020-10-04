import json
import logging
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def set_logger():
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )


def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(no_cuda=False, gpus='0'):
    return torch.device(f"cuda:{gpus}" if torch.cuda.is_available() and not no_cuda else "cpu")


def init_model_weights(model_obj):
    for l in model_obj.children():
        if isinstance(l, nn.Sequential):
            for sub_l in l.children():
                if isinstance(sub_l, (nn.Linear, nn.Conv2d)):
                    nn.init.xavier_normal_(sub_l.weight.data, gain=2**0.5)
                    # nn.init.xavier_normal_(sub_l.weight.data)
        else:
            if isinstance(sub_l, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_normal_(l.weight.data, gain=2**0.5)
                # nn.init.xavier_normal_(l.weight.data)


def save_args(folder, args, name='config.json'):
    set_logger()
    path = Path(folder)
    if path.exists():
        logging.warning(f"folder {folder} already exists! old files might be lost.")
    path.mkdir(parents=True, exist_ok=True)

    json.dump(
        vars(args),
        open(path / name, "w")
    )


def circle_points(K, min_angle=None, max_angle=None):
    # generate evenly distributed preference vector
    ang0 = 1e-6 if min_angle is None else min_angle
    ang1 = np.pi / 2 - ang0 if max_angle is None else max_angle
    angles = np.linspace(ang0, ang1, K, endpoint=True)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.c_[x, y]

def plot_multimnist(results, epoch, fig_dir=None):
    m = 2
    rs = np.array(results['ray'])

    colors = plt.cm.rainbow(np.linspace(0, 1, len(rs)))
    fig, axs = plt.subplots(ncols=2, figsize=(15, 6))

    for k, r in enumerate(rs):
        r_inv = 1. / r
        ep_ray = 1.1 * r_inv / np.linalg.norm(r_inv)
        ep_ray_line = np.stack([np.zeros(m), ep_ray])
        label = r'$r^{-1}$ ray' if k == 0 else ''

        axs[0].plot(ep_ray_line[:, 0], ep_ray_line[:, 1], lw=1, color=colors[k], ls='--', dashes=(15, 5), label=label)
        axs[0].arrow(
            .95 * ep_ray[0], .95 * ep_ray[1], .05 * ep_ray[0], .05 * ep_ray[1], color=colors[k], lw=1, head_width=.02
        )

        axs[0].scatter(results['task1_loss'][k], results['task2_loss'][k], color=colors[k])
        axs[1].scatter(results['task1_acc'][k], results['task2_acc'][k], color=colors[k])

    axs[0].set_xlabel(r'$\ell_1$', size=15)
    axs[0].set_ylabel(r'$\ell_2$', size=15)
    axs[0].xaxis.set_label_coords(1.015, -0.03)
    axs[0].yaxis.set_label_coords(-0.01, 1.01)
    axs[0].spines['right'].set_color('none')
    axs[0].spines['top'].set_color('none')
    axs[0].set_title("Loss")

    axs[1].set_xlabel('Acc. 1', size=15)
    axs[1].set_ylabel('Acc. 2', size=15)
    axs[1].xaxis.set_label_coords(1.015, -0.03)
    axs[1].yaxis.set_label_coords(-0.01, 1.01)
    axs[1].spines['right'].set_color('none')
    axs[1].spines['top'].set_color('none')
    axs[1].set_title("Accuracy")

    plt.suptitle(f"Epoch {epoch}", size=20)

    if fig_dir is not None:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(fig_dir / f"epoch_{epoch}.png")

    else:
        plt.show()

    plt.close(fig)

    return results
