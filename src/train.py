import os
import sys
from typing import List

import click
import numpy as np
import torch
import torchvision.transforms as T
from PIL import ImageFile
from torch import nn, optim
from torch.utils import data, tensorboard

from src import configs
from src import data as lc_data
from src import network
from src.l3c import timer


def plot_bpsp(
        plotter: tensorboard.SummaryWriter, bits: network.Bits,
        inp_size: int, train_iter: int
) -> None:
    """ Plot bpsps for all keys on tensorboard.
        bpsp: bits per subpixel/bits per dimension
        There are 2 bpsps per key:
        self_bpsp: bpsp based on dimension of log-likelihood tensor.
            Measures bits if log-likelihood tensor is final scale.
        scaled_bpsp: bpsp based on dimension of original image.
            Measures how many bits we contribute to the total bpsp.

        param plotter: tensorboard logger
        param bits: bpsp aggregator
        param inp_size: product of dims of original image
        param train_iter: current training iteration
        returns: None
    """
    for key in bits.get_keys():
        plotter.add_scalar(
            f"{key}_self_bpsp", bits.get_self_bpsp(key).item(), train_iter)
        plotter.add_scalar(
            f"{key}_scaled_bpsp",
            bits.get_scaled_bpsp(key, inp_size).item(), train_iter)


def train_loop(
        x: torch.Tensor, compressor: nn.Module,
        optimizer: optim.Optimizer,  # type: ignore
        train_iter: int, plotter: tensorboard.SummaryWriter,
        plot_iters: int, clip: float,
) -> None:
    """ Training loop for one batch. Computes grads and runs optimizer.
    """
    compressor.train()
    optimizer.zero_grad()
    inp_size = np.prod(x.size())
    x = x.cuda()
    bits = compressor(x)
    total_loss = bits.get_total_bpsp(inp_size)
    total_loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(compressor.parameters(), clip)
    optimizer.step()

    if train_iter % plot_iters == 0:
        plotter.add_scalar(
            "train/bpsp", total_loss.item(), train_iter)
        # Plots gradident norm pre-clipping.
        plotter.add_scalar("train/grad_norm", grad_norm, train_iter)
        plot_bpsp(plotter, bits, inp_size, train_iter)


def run_eval(
        eval_loader: data.DataLoader, compressor: nn.Module,
        train_iter: int, plotter: tensorboard.SummaryWriter,
        epoch: int,
) -> None:
    """ Runs entire eval epoch. """
    time_accumulator = timer.TimeAccumulator()
    compressor.eval()
    inp_size = 0

    with torch.no_grad():
        # BitsKeeper is used to aggregates bits from all eval iterations.
        bits_keeper = network.Bits()
        for _, x in eval_loader:
            inp_size += np.prod(x.size())
            with time_accumulator.execute():
                x = x.cuda()
                bits = compressor(x)
            bits_keeper.add_bits(bits)

        total_bpsp = bits_keeper.get_total_bpsp(inp_size)

        eval_bpsp = total_bpsp.item()
        print(f"Iteration {train_iter} bpsp: {total_bpsp}")
        plotter.add_scalar(
            "eval/bpsp", eval_bpsp, train_iter)
        plotter.add_scalar(
            "eval/batch_time", time_accumulator.mean_time_spent(), train_iter)
        plot_bpsp(plotter, bits_keeper, inp_size, train_iter)

        if configs.best_bpsp > eval_bpsp:
            configs.best_bpsp = eval_bpsp
            torch.save(
                {"nets": compressor.nets.state_dict(),  # type: ignore
                 "best_bpsp": configs.best_bpsp,
                 "epoch": epoch},
                os.path.join(configs.plot, "best.pth"))


def save(compressor: network.Compressor,
         sampler_indices: List[int],
         index: int,
         epoch: int,
         train_iter: int,
         plot: str,
         filename: str) -> None:
    """ Checkpoints training such that the entire training state
        can be restored. Reason we need this is because condor
        cluster can preempt jobs.

        param compressor: Contains all of our networks.
        param sampler_indices: Random indices of dataset
            produced by our Sampler, which prevents us from having
            unbalanced sampling of our dataset when restoring. Important 
            because our number of epochs is low.
        param index: Current index of indices in Sampler. Tells which 
            part of dataset is sampled and which part is not.
        param train_iter: Train iteration at which model is last trained.
        param epoch: Current training epoch. 
        param plot: Directory to store checkpoint.
        param filename: Checkpoint filename.
    """
    torch.save({
        "nets": compressor.nets.state_dict(),
        "sampler_indices": sampler_indices,
        "index": index,
        "epoch": epoch,
        "train_iter": train_iter,
        "best_bpsp": configs.best_bpsp,
    }, os.path.join(plot, filename))


@click.command()
@click.option("--train-path", type=click.Path(exists=True),
              help="path to directory of training images.")
@click.option("--eval-path", type=click.Path(exists=True),
              help="path to directory of eval images.")
@click.option("--train-file", type=click.File("r"),
              help="file for training image names.")
@click.option("--eval-file", type=click.File("r"),
              help="file for eval image names.")
@click.option("--batch", type=int, help="Batch size for training.")
@click.option("--workers", type=int, default=1,
              help="Number of worker threads to use in dataloader.")
@click.option("--plot", type=str,
              help="path to store tensorboard run data/plots.")
@click.option("--epochs", type=int, default=50, show_default=True,
              help="Number of epochs to run.")
@click.option("--resblocks", type=int, default=5, show_default=True,
              help="Number of resblocks to use.")
@click.option("--n-feats", type=int, default=64, show_default=True,
              help="Size of feature vector/channel width.")
@click.option("--scale", type=int, default=3, show_default=True,
              help="Scale of downsampling")
@click.option("--load", type=click.Path(exists=True), default="/dev/null",
              help="Path to load model")
@click.option("--lr", type=float, default=1e-4, help="Learning rate")
@click.option("--eval-iters", type=int, default=0,
              help="Number of train iterations per evaluation. "
                   "If 0, then evaluate at the end of every epoch.")
@click.option("--lr-epochs", type=int, default=1,
              help="Number of epochs before multiplying learning rate by 0.75")
@click.option("--plot-iters", type=int, default=1000,
              help="Number of train iterations before plotting data")
@click.option("--K", type=int, default=10,
              help="Number of clusters in logistic mixture model.")
@click.option("--clip", type=float, default=0.5,
              help="Norm to clip by for gradient clipping.")
@click.option("--crop", type=int, default=128,
              help="Size of image crops in training.")
@click.option("--gd", type=click.Choice(["sgd", "adam", "rmsprop"]), default="adam",
              help="Type of gd to use.")
def main(
        train_path: str, eval_path: str, train_file, eval_file,
        batch: int, workers: int, plot: str, epochs: int,
        resblocks: int, n_feats: int, scale: int, load: str,
        lr: float, eval_iters: int, lr_epochs: int,
        plot_iters: int, k: int, clip: float,
        crop: int, gd: str,
) -> None:
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    configs.n_feats = n_feats
    configs.scale = scale
    configs.resblocks = resblocks
    configs.K = k
    configs.plot = plot

    print(sys.argv)

    os.makedirs(plot, exist_ok=True)
    model_load = os.path.join(plot, "train.pth")
    if os.path.isfile(model_load):
        load = model_load
    if os.path.isfile(load) and load != "/dev/null":
        checkpoint = torch.load(load)
        print(f"Loaded model from {load}.")
        print("Epoch:", checkpoint["epoch"])
        if checkpoint.get("best_bpsp") is None:
            print("Warning: best_bpsp not found!")
        else:
            configs.best_bpsp = checkpoint["best_bpsp"]
            print("Best bpsp:", configs.best_bpsp)
    else:
        checkpoint = {}

    compressor = network.Compressor()
    if checkpoint:
        compressor.nets.load_state_dict(checkpoint["nets"])
    compressor = compressor.cuda()

    optimizer: optim.Optimizer  # type: ignore
    if gd == "adam":
        optimizer = optim.Adam(compressor.parameters(), lr=lr, weight_decay=0)
    elif gd == "sgd":
        optimizer = optim.SGD(compressor.parameters(), lr=lr,
                              momentum=0.9, nesterov=True)
    elif gd == "rmsprop":
        optimizer = optim.RMSprop(  # type: ignore
            compressor.parameters(), lr=lr)
    else:
        raise NotImplementedError(gd)

    starting_epoch = checkpoint.get("epoch") or 0

    print(compressor)

    train_dataset = lc_data.ImageFolder(
        train_path,
        [filename.strip() for filename in train_file],
        scale,
        T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(crop),
        ]),
    )
    dataset_index = checkpoint.get("index") or 0
    train_sampler = lc_data.PreemptiveRandomSampler(
        checkpoint.get("sampler_indices") or torch.randperm(
            len(train_dataset)).tolist(),
        dataset_index,
    )
    train_loader = data.DataLoader(
        train_dataset, batch_size=batch, sampler=train_sampler,
        num_workers=workers, drop_last=True,
    )
    print(f"Loaded training dataset with {len(train_loader)} batches "
          f"and {len(train_loader.dataset)} images")
    eval_dataset = lc_data.ImageFolder(
        eval_path, [filename.strip() for filename in eval_file],
        scale,
        T.Lambda(lambda x: x),
    )
    eval_loader = data.DataLoader(
        eval_dataset, batch_size=1, shuffle=False,
        num_workers=workers, drop_last=False,
    )
    print(f"Loaded eval dataset with {len(eval_loader)} batches "
          f"and {len(eval_dataset)} images")

    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer, lr_epochs, gamma=0.75)

    for _ in range(starting_epoch):
        lr_scheduler.step()  # type: ignore

    train_iter = checkpoint.get("train_iter") or 0
    if eval_iters == 0:
        eval_iters = len(train_loader)

    for epoch in range(starting_epoch, epochs):
        with tensorboard.SummaryWriter(plot) as plotter:
            # input: List[Tensor], downsampled images.
            # sizes: N scale 4
            for _, inputs in train_loader:
                train_iter += 1
                batch_size = inputs[0].shape[0]

                train_loop(inputs, compressor, optimizer, train_iter,
                           plotter, plot_iters, clip)
                # Increment dataset_index before checkpointing because
                # dataset_index is starting index of index of the FIRST
                # unseen piece of data.
                dataset_index += batch_size

                if train_iter % plot_iters == 0:
                    plotter.add_scalar(
                        "train/lr",
                        lr_scheduler.get_lr()[0],  # type: ignore
                        train_iter)
                    save(compressor, train_sampler.indices, dataset_index,
                         epoch, train_iter, plot, "train.pth")

                if train_iter % eval_iters == 0:
                    run_eval(
                        eval_loader, compressor, train_iter,
                        plotter, epoch)

            lr_scheduler.step()  # type: ignore
            dataset_index = 0

    with tensorboard.SummaryWriter(plot) as plotter:
        run_eval(eval_loader, compressor, train_iter,
                 plotter, epochs)
    save(compressor, train_sampler.indices, train_sampler.index,
         epochs, train_iter, plot, "train.pth")
    print("training done")


if __name__ == "__main__":
    main()
