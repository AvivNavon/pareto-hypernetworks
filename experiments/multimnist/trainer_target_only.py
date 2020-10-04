import argparse

from tqdm import trange
from collections import defaultdict

from experiments.utils import *
from solvers.baseline_solvers import LinearScalarizationBaseline
from experiments.multimnist.models import VanilaRayLeNet
from experiments.multimnist.data import Dataset
from experiments.multimnist.plot_utils import *
from solvers.epo import EPO


def evaluate(net, loader, device, ray=None):
    net.eval()
    loss1 = nn.CrossEntropyLoss()
    loss2 = nn.CrossEntropyLoss()

    ray = ray.to(device)

    eval_results = defaultdict(list)

    total, task1_correct, task2_correct, task1_loss, task2_loss = 0., 0., 0., 0., 0.
    with torch.no_grad():
        for i, batch in enumerate(loader):
            net.eval()
            img, ys = batch
            img = img.to(device)
            ys = ys.to(device)

            outputs = net(img, ray)

            logit1 = outputs[0]
            logit2 = outputs[1]

            pred1 = logit1.data.max(1)[1]  # first column has actual prob.
            pred2 = logit2.data.max(1)[1]  # first column has actual prob.
            task1_correct += pred1.eq(ys[:, 0]).sum()
            task2_correct += pred2.eq(ys[:, 1]).sum()
            total += len(ys)

            l1 = loss1(logit1, ys[:, 0])
            l2 = loss2(logit2, ys[:, 1])

            l1 *= len(ys)
            l2 *= len(ys)

            task1_loss += l1
            task2_loss += l2

    task1_acc, task2_acc = task1_correct / total, task2_correct / total
    task1_loss, task2_loss = task1_loss / total, task2_loss / total

    eval_results['ray'].append(ray.cpu().numpy().tolist())
    eval_results['task1_loss'].append(task1_loss.cpu().item())
    eval_results['task2_loss'].append(task2_loss.cpu().item())
    eval_results['task1_acc'].append(task1_acc.cpu().item())
    eval_results['task2_acc'].append(task2_acc.cpu().item())

    return eval_results


def train(
        path,
        solver_type: str,
        epochs: int,
        lr: float,
        bs: int,
        device,
        eval_every: int,
        ray_lower: float = .0,
        ray_upper: float = 1.,
        fixed_rays: bool = False
) -> None:
    # ----
    # Nets
    # ---
    net: nn.Module = VanilaRayLeNet(2)
    net = net.to(device)

    # ---------
    # Task loss
    # ---------
    loss1 = nn.CrossEntropyLoss()
    loss2 = nn.CrossEntropyLoss()

    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # ------
    # solver
    # ------
    solvers = dict(ls=LinearScalarizationBaseline, epo=EPO)

    solver_method = solvers[solver_type]
    if solver_type == 'epo':
        solver = solver_method(n_tasks=2, n_params=count_parameters(net))
    else:
        # ls
        solver = solver_method(n_tasks=2)

    # ----
    # data
    # ----
    data = Dataset(path)
    train_set, test_set = data.get_datasets()

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
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
    last_fig = -1
    epoch_iter = trange(epochs)

    # test rays
    min_angle = 0.01
    max_angle = np.pi / 2 - 0.01
    test_rays = circle_points(args.n_rays, min_angle=min_angle, max_angle=max_angle)

    results = defaultdict(list)

    for epoch in epoch_iter:

        for i, batch in enumerate(train_loader):
            net.train()
            optimizer.zero_grad()
            img, ys = batch
            img = img.to(device)
            ys = ys.to(device)

            alpha = torch.empty(1, ).uniform_(ray_lower, ray_upper)
            ray = torch.tensor([alpha.item(), 1 - alpha.item()]).to(device)

            logit1, logit2 = net(img, ray)

            l1 = loss1(logit1, ys[:, 0])
            l2 = loss2(logit2, ys[:, 1])
            losses = torch.stack((l1, l2))

            ray = ray.squeeze(0)
            loss = solver(losses, ray, list(net.parameters()))

            loss.backward()
            epoch_iter.set_description(
                f"total weighted loss: {loss.item():.3f}, loss 1: {l1.item():.3f}, loss 2: {l2.item():.3f},"
                f"ray {ray.cpu().numpy().tolist()}"
            )

            optimizer.step()

        if (epoch + 1) % args.fig_every == 0:
            # last_fig = epoch
            m = 2
            colors = plt.cm.rainbow(np.linspace(0, 1, len(test_rays)))
            fig, axs = plt.subplots(ncols=2, figsize=(15, 6))

            for k, r in enumerate(test_rays):
                eval_results = evaluate(
                    net, loader=test_loader, device=device, ray=torch.from_numpy(r.astype(np.float32))
                )
                results[f'epoch_{epoch + 1}'].append(eval_results)

                r_inv = 1. / r
                ep_ray = 1.1 * r_inv / np.linalg.norm(r_inv)
                ep_ray_line = np.stack([np.zeros(m), ep_ray])
                label = r'$r^{-1}$ ray' if k == 0 else ''

                axs[0].plot(ep_ray_line[:, 0], ep_ray_line[:, 1], lw=1, color=colors[k], ls='--', dashes=(15, 5),
                            label=label)
                axs[0].arrow(
                    .95 * ep_ray[0], .95 * ep_ray[1], .05 * ep_ray[0], .05 * ep_ray[1], color=colors[k], lw=1,
                    head_width=.02
                )

                axs[0].scatter(eval_results['task1_loss'], eval_results['task2_loss'], color=colors[k])
                axs[1].scatter(eval_results['task1_acc'], eval_results['task2_acc'], color=colors[k])

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

            if args.out_dir is not None:
                fig_dir = Path(args.out_dir)
                fig_dir.mkdir(exist_ok=True, parents=True)
                plt.savefig(fig_dir / f"epoch_{epoch}.png")

            else:
                plt.show()

            plt.close(fig)

    with open(Path(args.out_dir) / "results.json", "w") as file:
        json.dump(results, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MultiMNIST')

    parser.add_argument(
        '--datapath',
        type=str,
        default='/cortex/data/images/paretoHN/data/multi_fashion_and_mnist.pickle',
        help='path to data'
    )

    parser.add_argument('--n-epochs', type=int, default=50, help='num. epochs')
    parser.add_argument('--ray-lower', type=float, default=0., help='lower range for ray')
    parser.add_argument('--ray-upper', type=float, default=1., help='upper range for ray')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='train on gpu')
    parser.add_argument('--fixed-rays', action='store_true', default=False, help='train on test rays')
    parser.add_argument('--gpus', type=str, default='0', help='gpu device')
    parser.add_argument('--optim', type=str, default='adam', choices=['adam', 'sgd'], help='optimizer type')
    parser.add_argument('--batch-size', type=int, default=256, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--solver', type=str, choices=['ls', 'epo'], default='epo', help='solver')
    parser.add_argument('--eval-every', type=int, default=1, help='number of epochs between evaluations')
    parser.add_argument(
        '--out-dir', type=str,
        default='/cortex/data/images/paretoHN/outputs/multi_mnist_fashion',
        help='outputs dir'
    )

    parser.add_argument('--n-rays', type=int, default=5, help='num. rays')
    parser.add_argument('--fig-every', type=int, default=5, help='create fig. every fig_every epochs')
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
        ray_lower=args.ray_lower,
        ray_upper=args.ray_upper,
        fixed_rays=args.fixed_rays
    )

    save_args(folder=args.out_dir, args=args)
