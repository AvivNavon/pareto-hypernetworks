import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt


def plot_multimnist(results, epoch, fig_dir=None):
    m = 2
    rs = np.array(results['ray'])

    colors = plt.cm.rainbow(np.linspace(0, 1, len(rs)))
    fig, axs = plt.subplots(ncols=2, figsize=(15, 6))

    for k, r in enumerate(rs):
        plot_rays(r=r, k=k, axs=axs, colors=colors)

        axs[0].scatter(results['task1_loss'][k], results['task2_loss'][k], color=colors[k])
        axs[1].scatter(results['task1_acc'][k], results['task2_acc'][k], color=colors[k])

    set_axes_style(axes=axs)

    plt.suptitle(f"Epoch {epoch}", size=20)

    if fig_dir is not None:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(fig_dir / f"epoch_{epoch}.png")

    else:
        plt.show()

    plt.close(fig)

    return results


def set_axes_style(axes):
    axes[0].set_xlabel(r'$\ell_1$', size=15)
    axes[0].set_ylabel(r'$\ell_2$', size=15)
    axes[0].xaxis.set_label_coords(1.015, -0.03)
    axes[0].yaxis.set_label_coords(-0.01, 1.01)
    axes[0].spines['right'].set_color('none')
    axes[0].spines['top'].set_color('none')
    axes[0].set_title("Loss")

    axes[1].set_xlabel('Acc. 1', size=15)
    axes[1].set_ylabel('Acc. 2', size=15)
    axes[1].xaxis.set_label_coords(1.015, -0.03)
    axes[1].yaxis.set_label_coords(-0.01, 1.01)
    axes[1].spines['right'].set_color('none')
    axes[1].spines['top'].set_color('none')
    axes[1].set_title("Accuracy")


def plot_rays(r, k, axs, colors, m=2):
    r_inv = 1. / r
    ep_ray = 1.1 * r_inv / np.linalg.norm(r_inv)
    ep_ray_line = np.stack([np.zeros(m), ep_ray])
    label = r'$r^{-1}$ ray' if k == 0 else ''

    axs[0].plot(ep_ray_line[:, 0], ep_ray_line[:, 1], lw=1, color=colors[k], ls='--', dashes=(15, 5), label=label)
    axs[0].arrow(
        .95 * ep_ray[0], .95 * ep_ray[1], .05 * ep_ray[0], .05 * ep_ray[1], color=colors[k], lw=1, head_width=.02
    )
