import argparse
from collections import defaultdict

from tqdm import trange

from experiments.utils import *
from solvers.baseline_solvers import LinearScalarizationBaseline
from experiments.multimnist.models import PHNHyper, PHNTarget
from experiments.multimnist.data import Dataset
from experiments.multimnist.plot_utils import plot_multimnist
from experiments.metrics import non_uniformity, min_norm, calc_hypervolume
from solvers.epo import EPO


def evaluate(hypernet, targetnet, loader, rays, device):
    hypernet.eval()
    loss1 = nn.CrossEntropyLoss()
    loss2 = nn.CrossEntropyLoss()

    results = defaultdict(list)
    all_losses = []
    for ray in rays:
        total = 0.
        task1_correct, task2_correct = 0., 0.
        l1, l2 = 0., 0.
        min_norm_val, non_unif = 0., 0.
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
            losses = torch.stack((curr_l1, curr_l2))
            # metrics
            ray = ray.squeeze(0)
            min_norm_val += min_norm(losses, ray, hypernet.shared_parameters()) * bs
            non_unif += non_uniformity(losses, ray) * bs

            # losses
            l1 += curr_l1 * bs
            l2 += curr_l2 * bs

            # acc
            pred1 = logit1.data.max(1)[1]  # first column has actual prob.
            pred2 = logit2.data.max(1)[1]  # first column has actual prob.
            task1_correct += pred1.eq(ys[:, 0]).sum()
            task2_correct += pred2.eq(ys[:, 1]).sum()

            total += bs

        results['ray'].append(ray.cpu().numpy().tolist())
        results['task1_acc'].append(task1_correct.cpu().item() / total)
        results['task2_acc'].append(task2_correct.cpu().item() / total)
        results['task1_loss'].append(l1.cpu().item() / total)
        results['task2_loss'].append(l2.cpu().item() / total)
        results['min_norm'].append(min_norm_val / total)
        results['non_unif'].append(non_unif.cpu().item() / total)

        all_losses.append([results['task1_loss'][-1], results['task2_loss'][-1]])

    results['hv'] = calc_hypervolume(np.array(all_losses), ref_point=np.ones(2, ) * 3)

    return results


def train(
        path,
        solver_type: str,
        epochs: int,
        lr: float,
        bs: int,
        device,
        eval_every: int,
        fixed_rays: bool = False
) -> None:
    # ----
    # Nets
    # ----
    hnet: nn.Module = PHNHyper([9, 5], ray_hidden_dim=args.ray_hidden)
    net: nn.Module = PHNTarget([9, 5])

    hnet = hnet.to(device)
    net = net.to(device)

    # ---------
    # Task loss
    # ---------
    loss1 = nn.CrossEntropyLoss()
    loss2 = nn.CrossEntropyLoss()

    scheduler = None
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(hnet.parameters(), lr=lr, momentum=0., weight_decay=args.wd)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 45, 60, 75, 90], gamma=0.5)
    else:
        optimizer = torch.optim.Adam(hnet.parameters(), lr=lr, weight_decay=args.wd)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15], gamma=0.5)

    # ------
    # solver
    # ------
    solvers = dict(ls=LinearScalarizationBaseline, epo=EPO)

    solver_method = solvers[solver_type]
    if solver_type == 'epo':
        solver = solver_method(n_tasks=2, n_params=count_parameters(hnet))
    else:
        # ls
        solver = solver_method(n_tasks=2)

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

    # train on fixed test (!) rays
    min_angle = 0.1
    max_angle = np.pi / 2 - 0.1
    test_rays = circle_points(args.n_rays, min_angle=min_angle, max_angle=max_angle)
    if fixed_rays:
        rays = test_rays

    n_steps = epochs * len(train_loader)
    rays_to_sample = circle_points(n_steps, min_angle=1e-2, max_angle=np.pi / 2 - 1e-2)

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

            if args.circle_rays:
                idx = np.random.choice(range(n_steps))
                ray = torch.from_numpy(rays_to_sample[idx].astype(np.float32)).to(device)
            elif fixed_rays:
                idx = np.random.choice(range(args.n_rays))
                ray = torch.from_numpy(rays[idx].astype(np.float32)).to(device)
            else:
                if args.alpha > 0:
                    alpha = args.alpha
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

            # grad clip
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(hnet.parameters(), args.clip)

            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if (epoch + 1) % eval_every == 0:
            last_eval = epoch
            if not args.no_val_eval:
                epoch_results = evaluate(hypernet=hnet, targetnet=net, loader=val_loader, rays=test_rays, device=device)
                val_results[f'epoch_{epoch + 1}'] = epoch_results

            test_epoch_results = evaluate(hypernet=hnet, targetnet=net, loader=test_loader, rays=test_rays, device=device)
            test_results[f'epoch_{epoch + 1}'] = test_epoch_results

            plot_multimnist(test_epoch_results, epoch=epoch+1, fig_dir=args.out_dir)

    if epoch != last_eval:
        if not args.no_val_eval:
            epoch_results = evaluate(hypernet=hnet, targetnet=net, loader=val_loader, rays=test_rays, device=device)
            val_results[f'epoch_{epoch + 1}'] = epoch_results

        test_epoch_results = evaluate(hypernet=hnet, targetnet=net, loader=test_loader, rays=test_rays, device=device)
        test_results[f'epoch_{epoch + 1}'] = test_epoch_results

        plot_multimnist(test_epoch_results, epoch=epoch + 1, fig_dir=args.out_dir)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(Path(out_dir) / "val_results.json", "w") as file:
        json.dump(val_results, file)
    with open(Path(out_dir) / "test_results.json", "w") as file:
        json.dump(test_results, file)

    if args.save_model:
        torch.save(hnet.state_dict(), Path(out_dir) / f"hypernet.ckpt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MultiMNIST')

    parser.add_argument(
        '--datapath',
        type=str,
        default='/cortex/data/images/paretoHN/data/multi_fashion_and_mnist.pickle',
        help='path to data'
    )
    parser.add_argument('--circle-rays', action='store_true', default=False, help='rays from circle')
    parser.add_argument('--n-epochs', type=int, default=50, help='num. epochs')
    parser.add_argument('--ray-hidden', type=int, default=100, help='lower range for ray')
    parser.add_argument('--alpha', type=float, default=-1, help='alpha for dirichlet')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='train on gpu')
    parser.add_argument('--fixed-rays', action='store_true', default=False, help='train on test rays')
    parser.add_argument('--save-model', action='store_true', default=False, help='save model')
    parser.add_argument('--gpus', type=str, default='0', help='gpu device')
    parser.add_argument('--optim', type=str, default='adam', choices=['adam', 'sgd'], help='optimizer type')
    parser.add_argument('--batch-size', type=int, default=256, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--wd', type=float, default=0., help='weight decay')
    parser.add_argument('--val-size', type=float, default=.1, help='validation size')
    parser.add_argument('--no-val-eval', action='store_true', default=False, help='evaluate on validation')
    parser.add_argument('--clip', type=float, default=-1, help='grad clipping')
    parser.add_argument('--solver', type=str, choices=['ls', 'epo'], default='epo', help='solver')
    parser.add_argument('--eval-every', type=int, default=10, help='number of epochs between evaluations')
    parser.add_argument(
        '--out-dir', type=str, 
        default='/cortex/data/images/paretoHN/outputs/multi_mnist_fashion',
        help='outputs dir'
    )
    parser.add_argument('--n-rays', type=int, default=5, help='num. rays')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    args = parser.parse_args()

    set_seed(args.seed)
    set_logger()
    assert not (args.circle_rays and args.fixed_rays)
    train(
        path=args.datapath,
        solver_type=args.solver,
        epochs=args.n_epochs,
        lr=args.lr,
        bs=args.batch_size,
        device=get_device(no_cuda=args.no_cuda, gpus=args.gpus),
        eval_every=args.eval_every,
        fixed_rays=args.fixed_rays
    )

    save_args(folder=args.out_dir, args=args)
