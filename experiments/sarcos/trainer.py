import argparse
import json
from pathlib import Path

from tqdm import trange
from collections import defaultdict
import logging

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from experiments.utils import (
    set_seed,
    set_logger,
    count_parameters,
    get_device,
    save_args,
)
from experiments.sarcos.models import HyperFCNet, TargetFCNet
from experiments.sarcos.data import get_data
from pymoo.factory import get_reference_directions
from pymoo.factory import get_performance_indicator
from phn import EPOSolver, LinearScalarizationSolver


@torch.no_grad()
def evaluate(hypernet, targetnet, loader, rays, device):
    hypernet.eval()
    results = defaultdict(list)
    for ray in rays:
        ray = torch.from_numpy(ray.astype(np.float32)).to(device)

        ray /= ray.sum()

        total = 0.0
        full_losses = []
        for batch in loader:
            hypernet.zero_grad()

            batch = (t.to(device) for t in batch)
            xs, ys = batch
            bs = len(ys)

            weights = hypernet(ray)
            pred = targetnet(xs, weights)

            # loss
            curr_losses = get_losses(pred, ys)
            # metrics
            ray = ray.squeeze(0)

            # losses
            full_losses.append(curr_losses.detach().cpu().numpy())
            total += bs

        results["ray"].append(ray.cpu().numpy().tolist())
        results["loss"].append(np.array(full_losses).mean(0).tolist())

    hv = get_performance_indicator(
        "hv",
        ref_point=np.ones(
            7,
        ),
    )
    results["hv"] = hv.calc(np.array(results["loss"]))

    return results


# ---------
# Task loss
# ---------
def get_losses(pred, label):
    return F.mse_loss(pred, label, reduction="none").mean(0)


def get_test_rays():
    """Create 100 rays for evaluation. Not pretty but does the trick"""
    test_rays = get_reference_directions("das-dennis", 7, n_partitions=11).astype(
        np.float32
    )
    test_rays = test_rays[[(r > 0).all() for r in test_rays]][5:-5:2]
    logging.info(f"initialize {len(test_rays)} test rays")
    return test_rays


def train(
    path,
    solver_type: str,
    epochs: int,
    lr: float,
    bs: int,
    device,
    eval_every: int,
) -> None:
    # ----
    # Nets
    # ----
    hnet: nn.Module = HyperFCNet()
    net: nn.Module = TargetFCNet()

    hnet = hnet.to(device)
    net = net.to(device)

    optimizer = torch.optim.Adam(hnet.parameters(), lr=lr, weight_decay=args.wd)

    # ------
    # solver
    # ------
    solvers = dict(ls=LinearScalarizationSolver, epo=EPOSolver)

    solver_method = solvers[solver_type]
    if solver_type == "epo":
        solver = solver_method(n_tasks=7, n_params=count_parameters(hnet))
    else:
        # ls
        solver = solver_method(n_tasks=7)

    # ----
    # data
    # ----
    train_set, val_set, test_set = get_data(path)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=bs, shuffle=True, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set, batch_size=bs, shuffle=True, num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=bs, shuffle=False, num_workers=4
    )

    # ----------
    # Train loop
    # ----------
    last_eval = -1
    epoch_iter = trange(epochs)
    test_hv = -1.
    val_hv = -1.

    val_results = dict()
    test_results = dict()

    test_rays = get_test_rays()

    for epoch in epoch_iter:

        for i, batch in enumerate(train_loader):
            hnet.train()
            optimizer.zero_grad()
            batch = (t.to(device) for t in batch)
            xs, ys = batch

            alpha = args.alpha
            ray = torch.from_numpy(
                np.random.dirichlet([alpha] * 7, 1).astype(np.float32).flatten()
            ).to(device)

            weights = hnet(ray)
            pred = net(xs, weights)

            losses = get_losses(pred, ys)

            ray = ray.squeeze(0)
            loss = solver(losses, ray, list(hnet.parameters()))

            loss.backward()
            epoch_iter.set_description(
                f"total weighted loss: {loss.item():.3f}, MSE: {losses.mean().item():.3f}, "
                f"val. HV {val_hv:.4f}, test HV {test_hv:.4f}"
            )

            # grad clip
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(hnet.parameters(), args.clip)

            optimizer.step()

        if (epoch + 1) % eval_every == 0:
            last_eval = epoch
            if not args.no_val_eval:
                epoch_results = evaluate(
                    hypernet=hnet,
                    targetnet=net,
                    loader=val_loader,
                    rays=test_rays,
                    device=device,
                )
                val_results[f"epoch_{epoch + 1}"] = epoch_results
                val_hv = epoch_results["hv"]

            test_epoch_results = evaluate(
                hypernet=hnet,
                targetnet=net,
                loader=test_loader,
                rays=test_rays,
                device=device,
            )
            test_results[f"epoch_{epoch + 1}"] = test_epoch_results
            test_hv = test_epoch_results["hv"]

    if epoch != last_eval:
        if not args.no_val_eval:
            epoch_results = evaluate(
                hypernet=hnet,
                targetnet=net,
                loader=val_loader,
                rays=test_rays,
                device=device,
            )
            val_results[f"epoch_{epoch + 1}"] = epoch_results

        test_epoch_results = evaluate(
            hypernet=hnet,
            targetnet=net,
            loader=test_loader,
            rays=test_rays,
            device=device,
        )
        test_results[f"epoch_{epoch + 1}"] = test_epoch_results

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(Path(out_dir) / "val_results.json", "w") as file:
        json.dump(val_results, file)
    with open(Path(out_dir) / "test_results.json", "w") as file:
        json.dump(test_results, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SARCOS")

    parser.add_argument("--datapath", type=str, default="data", help="path to data")
    parser.add_argument("--n-epochs", type=int, default=1000, help="num. epochs")
    parser.add_argument(
        "--ray-hidden", type=int, default=25, help="lower range for ray"
    )
    parser.add_argument("--alpha", type=float, default=0.2, help="alpha for dirichlet")
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="train on gpu"
    )
    parser.add_argument("--gpus", type=str, default="0", help="gpu device")
    parser.add_argument(
        "--optim",
        type=str,
        default="adam",
        choices=["adam", "sgd"],
        help="optimizer type",
    )
    parser.add_argument("--batch-size", type=int, default=512, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--wd", type=float, default=0.0, help="weight decay")
    parser.add_argument("--val-size", type=float, default=0.1, help="validation size")
    parser.add_argument(
        "--no-val-eval",
        action="store_true",
        default=False,
        help="evaluate on validation",
    )
    parser.add_argument("--clip", type=float, default=-1, help="grad clipping")
    parser.add_argument(
        "--solver", type=str, choices=["ls", "epo"], default="epo", help="solver"
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=10,
        help="number of epochs between evaluations",
    )
    parser.add_argument("--out-dir", type=str, default="output", help="outputs dir")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
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
    )

    save_args(folder=args.out_dir, args=args)
