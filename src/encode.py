import os
import sys

import click
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils import data

from src import configs
from src import data as lc_data
from src import network
from src.l3c import bitcoding, timer


@click.command()
@click.option("--path", type=click.Path(exists=True),
              help="Directory of images.")
@click.option("--file", type=click.File("r"),
              help="File for image names.")
@click.option("--resblocks", type=int, default=3, show_default=True,
              help="Number of resblocks to use.")
@click.option("--n-feats", type=int, default=64, show_default=True,
              help="Size of n_feats vector used.")
@click.option("--scale", type=int, default=3, show_default=True,
              help="Scale of downsampling")
@click.option("--load", type=click.Path(exists=True),
              help="Path to load model")
@click.option("--K", type=int, default=10,
              help="Number of clusters in logistic mixture model.")
@click.option("--crop", type=int, default=0,
              help="Size of image crops in training. 0 means no crop.")
@click.option("--log-likelihood", is_flag=True, default=False,
              help="Turn on log-likelihood calculations.")
@click.option("--decode", is_flag=True, default=False,
              help="Turn on decoding to verify coding correctness.")
@click.option("--save-path", type=str,
              help="Save directory for images.")
def main(
        path: str, file, resblocks: int, n_feats: int, scale: int,
        load: str, k: int, crop: int,
        log_likelihood: bool, decode: bool, save_path: str,
) -> None:

    configs.n_feats = n_feats
    configs.resblocks = resblocks
    configs.K = k
    configs.scale = scale
    configs.log_likelihood = log_likelihood
    configs.collect_probs = True

    print(sys.argv)

    checkpoint = torch.load(load)
    print(f"Loaded model from {load}.")
    print("Epoch:", checkpoint["epoch"])

    compressor = network.Compressor()
    compressor.nets.load_state_dict(checkpoint["nets"])
    compressor = compressor.cuda()

    print(compressor.nets)

    transforms = []  # type: ignore
    if crop > 0:
        transforms.insert(0, T.CenterCrop(crop))

    dataset = lc_data.ImageFolder(
        path, [filename.strip() for filename in file],
        scale, T.Compose(transforms)
    )
    loader = data.DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=0, drop_last=False,
    )
    print(f"Loaded directory with {len(dataset)} images")

    os.makedirs(save_path, exist_ok=True)

    coder = bitcoding.Bitcoding(compressor)
    encoder_time_accumulator = timer.TimeAccumulator()
    decoder_time_accumulator = timer.TimeAccumulator()
    total_file_bytes = 0
    total_num_subpixels = 0
    total_entropy_coding_bytes: np.ndarray = 0  # type: ignore
    total_log_likelihood_bits = network.Bits()

    for i, (filenames, x) in enumerate(loader):
        assert len(filenames) == 1, filenames
        filename = filenames[0]
        file_id = filename.split(".")[0]
        filepath = os.path.join(save_path, f"{file_id}.srec")

        with encoder_time_accumulator.execute():
            log_likelihood_bits, entropy_coding_bytes = coder.encode(
                x, filepath)

        total_file_bytes += os.path.getsize(filepath)
        total_entropy_coding_bytes += np.array(entropy_coding_bytes)
        total_num_subpixels += np.prod(x.size())
        if configs.log_likelihood:
            total_log_likelihood_bits.add_bits(log_likelihood_bits)

        if decode:
            with decoder_time_accumulator.execute():
                y = coder.decode(filepath)
                y = y.cpu()
            assert torch.all(x == y), (x[x != y], y[x != y])

        if configs.log_likelihood:
            theoretical_bpsp = total_log_likelihood_bits.get_total_bpsp(
                total_num_subpixels).item()
            print(
                f"Theoretical Bpsp: {theoretical_bpsp:.3f};\t", end="")
        print(
            f"Bpsp: {total_file_bytes*8/total_num_subpixels:.3f};\t"
            f"Images: {i + 1};\t"
            f"Comp: {encoder_time_accumulator.mean_time_spent():.3f};\t",
            end="")
        if decode:
            print(
                "Decomp: "
                f"{decoder_time_accumulator.mean_time_spent():.3f}",
                end="")
        print(end="\r")
    print()

    if decode:
        print("Decomp Time By Scale: ", end="")
        print(", ".join(
            f"{scale_time:.3f}"
            for scale_time in coder.decomp_scale_times()))
    else:
        print("Scale Bpsps: ", end="")
        print(", ".join(
            f"{scale_bpsp:.3f}"
            for scale_bpsp in total_entropy_coding_bytes*8/total_num_subpixels))


if __name__ == "__main__":
    main()
