import json
import argparse
from collections import defaultdict
from pathlib import Path

from tqdm import trange

import numpy as np
import torch
from torch import nn
from experiments.utils import set_logger, set_seed, circle_points, count_parameters, get_device, save_args
from phn import LinearScalarizationSolver, EPOSolver

from experiments.multimnist.models import PHNHyper, PHNTarget
from experiments.multimnist.data import Dataset
from experiments.multimnist.plot_utils import plot_multimnist


def evaluate(hypernet, targetnet, loader, rays, device):
    hypernet.eval()
    loss1 = nn.CrossEntropyLoss()
    loss2 = nn.CrossEntropyLoss()

    results = defaultdict(list)

    for ray in rays:
        total = 0.
        task1_correct, task2_correct = 0., 0.
        l1, l2 = 0., 0.
        ray = torch.from_numpy(ray.astype(np.float32)).to(device)

        if not args.circle_rays:
            ray /= ray.sum()

        for batch in loader:
            hypernet.zero_grad()

            batch = (t.to(device) for t in batch)
            img, ys = batch
            bs = len(ys)

            weights = hypernet(ray)
            logit1, logit2 = targetnet(img, weights)

            # loss
            curr_l1 = loss1(logit1, ys[:, 0])
            curr_l2 = loss2(logit2, ys[:, 1])
            l1 += curr_l1 * bs
            l2 += curr_l2 * bs

            # acc
            pred1 = logit1.data.max(1)[1]  # first column has actual prob.
            pred2 = logit2.data.max(1)[1]  # first column has actual prob.
            task1_correct += pred1.eq(ys[:, 0]).sum()
            task2_correct += pred2.eq(ys[:, 1]).sum()

            total += bs

        results['ray'].append(ray.squeeze(0).cpu().numpy().tolist())
        results['task1_acc'].append(task1_correct.cpu().item() / total)
        results['task2_acc'].append(task2_correct.cpu().item() / total)
        results['task1_loss'].append(l1.cpu().item() / total)
        results['task2_loss'].append(l2.cpu().item() / total)

    return results


def train(
        path,
        solver_type: str,
        epochs: int,
        ray_hidden: int,
        lr: float,
        wd: float,
        bs: int,
        val_size: float,
        n_rays: int,
        alpha: float,
        no_val_eval: bool,
        out_dir: str,
        device,
        eval_every: int,
) -> None:
    # ----
    # Nets
    # ----
    hnet: nn.Module = PHNHyper([9, 5], ray_hidden_dim=ray_hidden)
    net: nn.Module = PHNTarget([9, 5])

    hnet = hnet.to(device)
    net = net.to(device)

    # ---------
    # Task loss
    # ---------
    loss1 = nn.CrossEntropyLoss()
    loss2 = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(hnet.parameters(), lr=lr, weight_decay=wd)

    # ------
    # solver
    # ------
    solvers = dict(ls=LinearScalarizationSolver, epo=EPOSolver)

    solver_method = solvers[solver_type]
    if solver_type == 'epo':
        solver = solver_method(n_tasks=2, n_params=count_parameters(hnet))
    else:
        # ls
        solver = solver_method(n_tasks=2)

    # ----
    # data
    # ----
    assert val_size > 0, "please use validation by providing val_size > 0"
    data = Dataset(path, val_size=val_size)
    train_set, val_set, test_set = data.get_datasets()

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=bs,
        shuffle=True,
        num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=bs,
        shuffle=True,
        num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=bs,
        shuffle=False,
        num_workers=4
    )

    min_angle = 0.1
    max_angle = np.pi / 2 - 0.1
    test_rays = circle_points(n_rays, min_angle=min_angle, max_angle=max_angle)

    # ----------
    # Train loop
    # ----------
    last_eval = -1
    epoch_iter = trange(epochs)

    val_results = dict()
    test_results = dict()

    for epoch in epoch_iter:

        for i, batch in enumerate(train_loader):
            hnet.train()
            optimizer.zero_grad()
            img, ys = batch
            img = img.to(device)
            ys = ys.to(device)

            if alpha > 0:
                ray = torch.from_numpy(
                    np.random.dirichlet((alpha, alpha), 1).astype(np.float32).flatten()
                ).to(device)
            else:
                alpha = torch.empty(1, ).uniform_(0., 1.)
                ray = torch.tensor([alpha.item(), 1 - alpha.item()]).to(device)

            weights = hnet(ray)
            logit1, logit2 = net(img, weights)

            l1 = loss1(logit1, ys[:, 0])
            l2 = loss2(logit2, ys[:, 1])
            losses = torch.stack((l1, l2))

            ray = ray.squeeze(0)
            loss = solver(losses, ray, list(hnet.parameters()))

            loss.backward()

            epoch_iter.set_description(
                f"total weighted loss: {loss.item():.3f}, loss 1: {l1.item():.3f}, loss 2: {l2.item():.3f},"
                f"ray {ray.cpu().numpy().tolist()}"
            )

            optimizer.step()

        if (epoch + 1) % eval_every == 0:
            last_eval = epoch
            if not no_val_eval:
                epoch_results = evaluate(hypernet=hnet, targetnet=net, loader=val_loader, rays=test_rays, device=device)
                val_results[f'epoch_{epoch + 1}'] = epoch_results

            test_epoch_results = evaluate(
                hypernet=hnet, targetnet=net, loader=test_loader, rays=test_rays, device=device
            )
            test_results[f'epoch_{epoch + 1}'] = test_epoch_results

            plot_multimnist(test_epoch_results, epoch=epoch+1, fig_dir=out_dir)

    if epoch != last_eval:
        if not no_val_eval:
            epoch_results = evaluate(hypernet=hnet, targetnet=net, loader=val_loader, rays=test_rays, device=device)
            val_results[f'epoch_{epoch + 1}'] = epoch_results

        test_epoch_results = evaluate(hypernet=hnet, targetnet=net, loader=test_loader, rays=test_rays, device=device)
        test_results[f'epoch_{epoch + 1}'] = test_epoch_results

        plot_multimnist(test_epoch_results, epoch=epoch + 1, fig_dir=out_dir)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(Path(out_dir) / "val_results.json", "w") as file:
        json.dump(val_results, file)
    with open(Path(out_dir) / "test_results.json", "w") as file:
        json.dump(test_results, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MultiMNIST')
    parser.add_argument('--datapath', type=str, default='data/multi_fashion_and_mnist.pickle', help='path to data')
    parser.add_argument('--n-epochs', type=int, default=150, help='num. epochs')
    parser.add_argument('--ray-hidden', type=int, default=100, help='lower range for ray')
    parser.add_argument('--alpha', type=float, default=.2, help='alpha for dirichlet')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='train on gpu')
    parser.add_argument('--gpus', type=str, default='0', help='gpu device')
    parser.add_argument('--batch-size', type=int, default=256, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--wd', type=float, default=0., help='weight decay')
    parser.add_argument('--val-size', type=float, default=.1, help='validation size')
    parser.add_argument('--no-val-eval', action='store_true', default=False, help='evaluate on validation')
    parser.add_argument('--solver', type=str, choices=['ls', 'epo'], default='epo', help='solver')
    parser.add_argument('--eval-every', type=int, default=10, help='number of epochs between evaluations')
    parser.add_argument('--out-dir', type=str, default='outputs', help='outputs dir')
    parser.add_argument('--n-rays', type=int, default=25, help='num. rays')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    args = parser.parse_args()

    set_seed(args.seed)
    set_logger()

    train(
        path=args.datapath,
        solver_type=args.solver,
        epochs=args.n_epochs,
        ray_hidden=args.ray_hidden,
        lr=args.lr,
        wd=args.wd,
        bs=args.batch_size,
        device=get_device(no_cuda=args.no_cuda, gpus=args.gpus),
        eval_every=args.eval_every,
        no_val_eval=args.no_val_eval,
        val_size=args.val_size,
        n_rays=args.n_rays,
        alpha=args.alpha,
        out_dir=args.out_dir
    )

    save_args(folder=args.out_dir, args=args)
