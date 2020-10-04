import json
import logging
import argparse
from pathlib import Path
from tqdm import trange
from collections import defaultdict

import numpy as np
import torch
from torch import nn

from experiments.utils import get_device, set_logger, set_seed, circle_points, save_args, count_parameters
from experiments.multimnist.data import Dataset
from experiments.multimnist.models import LeNet
from experiments.metrics import min_norm, non_uniformity
from solvers.baseline_solvers import LinearScalarizationBaseline, PMTLBaseline, EPOBaseline, MGDABaseline


def evaluate_baseline(net, loader, device, ray):
    net.eval()
    loss1 = nn.CrossEntropyLoss()
    loss2 = nn.CrossEntropyLoss()

    eval_results = defaultdict()

    total, task1_correct, task2_correct, task1_loss, task2_loss = 0., 0., 0., 0., 0.
    min_norm_val, non_unif = 0., 0.
    ray = ray.to(device)
    # ray /= ray.sum()
    for i, batch in enumerate(loader):
        net.zero_grad()
        img, ys = batch
        img = img.to(device)
        ys = ys.to(device)
        bs = len(ys)

        outputs = net(img)

        logit1 = outputs[0]
        logit2 = outputs[1]

        pred1 = logit1.data.max(1)[1]  # first column has actual prob.
        pred2 = logit2.data.max(1)[1]  # first column has actual prob.
        task1_correct += pred1.eq(ys[:, 0]).sum()
        task2_correct += pred2.eq(ys[:, 1]).sum()
        total += bs

        l1 = loss1(logit1, ys[:, 0])
        l2 = loss2(logit2, ys[:, 1])
        losses = torch.stack((l1, l2))

        # metrics
        ray = ray.squeeze(0)
        min_norm_val += min_norm(losses, ray, net.shared_parameters()) * bs
        non_unif += non_uniformity(losses, ray) * bs

        task1_loss = task1_loss + l1 * bs
        task2_loss = task2_loss + l2 * bs

    task1_acc, task2_acc = task1_correct / total, task2_correct / total
    task1_loss, task2_loss = task1_loss / total, task2_loss / total
    min_norm_val, non_unif = min_norm_val / total, non_unif / total

    eval_results['ray'] = ray.cpu().numpy().tolist()
    eval_results['task1_loss'] = task1_loss.cpu().item()
    eval_results['task2_loss'] = task2_loss.cpu().item()
    eval_results['task1_acc'] = task1_acc.cpu().item()
    eval_results['task2_acc'] = task2_acc.cpu().item()
    eval_results['min_norm_val'] = min_norm_val
    eval_results['non_unif'] = non_unif.cpu().item()

    return eval_results


def train(
        path,
        solver_type: str,
        epochs: int,
        lr: float,
        bs: int,
        device: torch.device,
        eval_every: int,
        n_rays: int = 9
) -> None:

    # ---------
    # Task loss
    # ---------
    loss1 = nn.CrossEntropyLoss()
    loss2 = nn.CrossEntropyLoss()

    # ----
    # rays
    # ----
    if not args.hpo_mode:
        min_angle = 0.1
        max_angle = np.pi / 2 - 0.1
        rays = torch.tensor(circle_points(n_rays, min_angle=min_angle, max_angle=max_angle), dtype=torch.float32)
        rays = rays.to(device)
    else:
        logging.info("HPO mode, training on [.5, .5] only.")
        rays = torch.tensor([[.5, .5]], dtype=torch.float32).to(device)

    # ------
    # solver
    # ------
    solvers = dict(ls=LinearScalarizationBaseline, epo=EPOBaseline, pmtl=PMTLBaseline, mgda=MGDABaseline)

    # ----
    # data
    # ----
    assert args.val_size > 0, "please use validation by providing val_size > 0"
    data = Dataset(path, val_size=args.val_size)
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

    # ----------
    # Train loop
    # ----------
    val_results = defaultdict(lambda: defaultdict(list))
    test_results = defaultdict(lambda: defaultdict(list))

    out_dir = Path(args.out_dir) / f"{solver_type}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i_ray, ray in enumerate(rays):
        # ----
        # Nets
        # ----
        last_eval = -1
        epoch_iter = trange(epochs)

        net: nn.Module = LeNet(2)
        net = net.to(device)

        # ------
        # solver
        # ------
        solver_method = solvers[solver_type]
        if solver_type == 'epo':
            solver = solver_method(n_tasks=2, n_params=count_parameters(net))
        elif solver_type in ['ls', 'mgda']:
            solver = solver_method(n_tasks=2)
        else:
            # pmtl
            # norm_rays = rays / rays.sum(1).reshape(-1, 1)
            solver = solver_method(
                n_tasks=2,
                rays=rays.clone().detach().to(device),
                max_init_steps=2
            )

        scheduler = None
        if args.optim == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0., weight_decay=args.wd)
        else:
            optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=args.wd)
            # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15], gamma=0.5)

        for epoch in epoch_iter:
            net.train()
            for i, batch in enumerate(train_loader):
                optimizer.zero_grad()
                img, ys = batch
                img = img.to(device)
                ys = ys.to(device)

                # ray /= ray.sum()  # F.normalize(ray, p=1, dim=0)  # so that each ray sums to 1

                outputs = net(img)

                logit1 = outputs[0]
                logit2 = outputs[1]

                l1 = loss1(logit1, ys[:, 0])
                l2 = loss2(logit2, ys[:, 1])
                losses = torch.stack((l1, l2))

                ray = ray.squeeze(0)
                loss = solver(losses, ray, list(net.parameters()))

                loss.backward()
                epoch_iter.set_description(
                    f"total weighted loss: {loss.item():.3f}, loss 1: {l1.item():.3f}, loss 2: {l2.item():.3f}"
                )

                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            if (epoch + 1) % eval_every == 0:
                curr = f'epoch_{epoch + 1}'
                last_eval = epoch

                logging.info(f"epoch {epoch + 1}, ray [{ray[0].item():.2f}, {ray[1].item():.2f}]")
                if not args.no_val_eval:
                    eval_results = evaluate_baseline(net=net, loader=val_loader, device=device, ray=ray)
                    val_results[curr]['ray'].append(eval_results['ray'])
                    val_results[curr]['task1_loss'].append(eval_results['task1_loss'])
                    val_results[curr]['task2_loss'].append(eval_results['task2_loss'])
                    val_results[curr]['task1_acc'].append(eval_results['task1_acc'])
                    val_results[curr]['task2_acc'].append(eval_results['task2_acc'])
                    val_results[curr]['min_norm_val'].append(eval_results['min_norm_val'])
                    val_results[curr]['non_unif'].append(eval_results['non_unif'])

                eval_results = evaluate_baseline(net=net, loader=test_loader, device=device, ray=ray)
                test_results[curr]['ray'].append(eval_results['ray'])
                test_results[curr]['task1_loss'].append(eval_results['task1_loss'])
                test_results[curr]['task2_loss'].append(eval_results['task2_loss'])
                test_results[curr]['task1_acc'].append(eval_results['task1_acc'])
                test_results[curr]['task2_acc'].append(eval_results['task2_acc'])
                test_results[curr]['min_norm_val'].append(eval_results['min_norm_val'])
                test_results[curr]['non_unif'].append(eval_results['non_unif'])

        if epoch != last_eval:
            curr = f'epoch_{epoch + 1}'
            logging.info(f"epoch {epoch + 1}, ray [{ray[0].item():.2f}, {ray[1].item():.2f}]")

            if not args.no_val_eval:
                eval_results = evaluate_baseline(net=net, loader=test_loader, device=device, ray=ray)
                val_results[curr]['ray'].append(eval_results['ray'])
                val_results[curr]['task1_loss'].append(eval_results['task1_loss'])
                val_results[curr]['task2_loss'].append(eval_results['task2_loss'])
                val_results[curr]['task1_acc'].append(eval_results['task1_acc'])
                val_results[curr]['task2_acc'].append(eval_results['task2_acc'])
                val_results[curr]['min_norm_val'].append(eval_results['min_norm_val'])
                val_results[curr]['non_unif'].append(eval_results['non_unif'])

            eval_results = evaluate_baseline(net=net, loader=test_loader, device=device, ray=ray)
            test_results[curr]['ray'].append(eval_results['ray'])
            test_results[curr]['task1_loss'].append(eval_results['task1_loss'])
            test_results[curr]['task2_loss'].append(eval_results['task2_loss'])
            test_results[curr]['task1_acc'].append(eval_results['task1_acc'])
            test_results[curr]['task2_acc'].append(eval_results['task2_acc'])
            test_results[curr]['min_norm_val'].append(eval_results['min_norm_val'])
            test_results[curr]['non_unif'].append(eval_results['non_unif'])

        # torch.save(net.state_dict(), Path(out_dir) / f"net_{i_ray}.ckpt")

    with open(Path(out_dir) / "val_results.json", "w") as file:
        json.dump(val_results, file)
    with open(Path(out_dir) / "test_results.json", "w") as file:
        json.dump(test_results, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MultiMNIST - Baselines')
    parser.add_argument(
        '--datapath',
        type=str,
        default='/cortex/data/images/paretoHN/data/multi_mnist.pickle',
        help='path to data'
    )
    parser.add_argument('--n-epochs', type=int, default=50, help='num. epochs')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='train on gpu')
    parser.add_argument('--gpus', type=str, default='1', help='gpu device')
    parser.add_argument('--optim', type=str, default='adam', choices=['adam', 'sgd'], help='optimizer type')
    parser.add_argument('--batch-size', type=int, default=256, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--wd', type=float, default=0., help='weight decay')
    parser.add_argument('--val-size', type=float, default=.1, help='validation size')
    parser.add_argument('--no-val-eval', action='store_true', default=False, help='evaluate on validation')
    parser.add_argument('--hpo-mode', action='store_true', default=False, help='for hpo')
    parser.add_argument('--solver', type=str, choices=['ls', 'epo', 'pmtl', 'mgda'], default='epo', help='solver')
    parser.add_argument('--eval-every', type=int, default=2, help='number of epochs between evaluations')
    parser.add_argument(
        '--out-dir', type=str,
        default='/cortex/data/images/paretoHN/outputs/',
        help='outputs dir'
    )
    parser.add_argument('--n-rays', type=int, default=5, help='num. rays')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    args = parser.parse_args()

    set_seed(args.seed)
    set_logger()

    train(
        path=args.datapath,
        solver_type=args.solver,
        epochs=args.n_epochs,
        lr=args.lr,
        bs=args.batch_size,
        device=get_device(no_cuda=args.no_cuda, gpus=args.gpus),
        eval_every=args.eval_every,
        n_rays=args.n_rays
    )

    save_args(folder=Path(args.out_dir) / f"{args.solver}", args=args)
