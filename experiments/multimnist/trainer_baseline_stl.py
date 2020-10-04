import json
import logging
import argparse
from pathlib import Path
from tqdm import trange
from collections import defaultdict
from torch import nn

from experiments.utils import get_device, set_logger, set_seed, save_args
from experiments.multimnist.data import Dataset
from experiments.multimnist.models import LeNet
from experiments.baselines import *


def evaluate_baseline(net, loader, device):
    net.eval()
    loss = nn.CrossEntropyLoss()

    eval_results = defaultdict()

    total, task_correct, task_loss = 0., 0., 0.
    for i, batch in enumerate(loader):
        net.zero_grad()
        img, ys = batch
        img = img.to(device)
        ys = ys.to(device)
        bs = len(ys)

        outputs = net(img)

        logit = outputs[args.task]

        pred = logit.data.max(1)[1]  # first column has actual prob.
        task_correct += pred.eq(ys[:, args.task]).sum()
        total += bs

        l = loss(logit, ys[:, args.task])
        task_loss = task_loss + l * bs

    task_acc, task_loss = task_correct / total, task_loss / total

    eval_results['task_loss'] = task_loss.cpu().item()
    eval_results['task_acc'] = task_acc.cpu().item()

    return eval_results


def train(
        path,
        epochs: int,
        lr: float,
        bs: int,
        device: torch.device,
        eval_every: int,
) -> None:

    # ---------
    # Task loss
    # ---------
    criterion = nn.CrossEntropyLoss()

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

    # ----
    # Nets
    # ----
    last_eval = -1
    epoch_iter = trange(epochs)

    net: nn.Module = LeNet(2)
    net = net.to(device)

    task = args.task

    scheduler = None
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0., weight_decay=args.wd)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 45, 60, 75, 90], gamma=0.5)
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

            outputs = net(img)

            logit = outputs[task]

            loss = criterion(logit, ys[:, task])

            loss.backward()
            epoch_iter.set_description(
                f"loss: {loss.item():.3f}"
            )

            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if (epoch + 1) % eval_every == 0:
            curr = f'epoch_{epoch + 1}'
            last_eval = epoch

            if not args.no_val_eval:
                eval_results = evaluate_baseline(net=net, loader=val_loader, device=device)
                val_results[curr]['task_loss'].append(eval_results['task_loss'])
                val_results[curr]['task_acc'].append(eval_results['task_acc'])

            eval_results = evaluate_baseline(net=net, loader=test_loader, device=device)
            test_results[curr]['task_loss'].append(eval_results['task_loss'])
            test_results[curr]['task_acc'].append(eval_results['task_acc'])

    if epoch != last_eval:
        curr = f'epoch_{epoch + 1}'
        if not args.no_val_eval:
            eval_results = evaluate_baseline(net=net, loader=val_loader, device=device)
            val_results[curr]['task_loss'].append(eval_results['task_loss'])
            val_results[curr]['task_acc'].append(eval_results['task_acc'])

        eval_results = evaluate_baseline(net=net, loader=test_loader, device=device)
        test_results[curr]['task_loss'].append(eval_results['task_loss'])
        test_results[curr]['task_acc'].append(eval_results['task_acc'])

    out_dir = Path(args.out_dir) / f"stl/task_{args.task}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(Path(out_dir) / "val_results.json", "w") as file:
        json.dump(val_results, file)
    with open(Path(out_dir) / "test_results.json", "w") as file:
        json.dump(test_results, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MultiMNIST - Baselines STL')
    parser.add_argument(
        '--datapath',
        type=str,
        default='/cortex/data/images/paretoHN/data/multi_mnist.pickle',
        help='path to data'
    )
    parser.add_argument('--task', type=int, default=0, choices=[0, 1], help='task')
    parser.add_argument('--n-epochs', type=int, default=50, help='num. epochs')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='train on gpu')
    parser.add_argument('--gpus', type=str, default='1', help='gpu device')
    parser.add_argument('--optim', type=str, default='adam', choices=['adam', 'sgd'], help='optimizer type')
    parser.add_argument('--batch-size', type=int, default=256, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--wd', type=float, default=0., help='weight decay')
    parser.add_argument('--val-size', type=float, default=.1, help='validation size')
    parser.add_argument('--no-val-eval', action='store_true', default=False, help='evaluate on validation')
    parser.add_argument('--eval-every', type=int, default=2, help='number of epochs between evaluations')
    parser.add_argument(
        '--out-dir', type=str,
        default='/cortex/data/images/paretoHN/outputs/',
        help='outputs dir'
    )
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    args = parser.parse_args()

    set_seed(args.seed)
    set_logger()

    train(
        path=args.datapath,
        epochs=args.n_epochs,
        lr=args.lr,
        bs=args.batch_size,
        device=get_device(no_cuda=args.no_cuda, gpus=args.gpus),
        eval_every=args.eval_every,
    )

    save_args(folder=Path(args.out_dir) / f"stl/task_{args.task}", args=args)
