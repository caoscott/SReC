import os
import sys

import click
import numpy as np
import torch
import torchvision.transforms as T
from PIL import ImageFile

from src import configs, network
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
@click.option("--save-path", type=str,
              help="Save directory for images.")
def main(
        path: str, file, resblocks: int, n_feats: int, scale: int,
        load: str, k: int, save_path: str,
) -> None:
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    configs.n_feats = n_feats
    configs.resblocks = resblocks
    configs.K = k
    configs.scale = scale
    configs.collect_probs = False

    print(sys.argv)

    checkpoint = torch.load(load)
    print(f"Loaded model from {load}.")
    print("Epoch:", checkpoint["epoch"])

    compressor = network.Compressor()
    compressor.nets.load_state_dict(checkpoint["nets"])
    compressor = compressor.cuda()
    print(compressor.nets)

    filenames = [filename.strip() for filename in file]
    print(f"Loaded directory with {len(filenames)} images")

    os.makedirs(save_path, exist_ok=True)

    coder = bitcoding.Bitcoding(compressor)
    decoder_time_accumulator = timer.TimeAccumulator()
    total_num_bytes = 0
    total_num_subpixels = 0

    for filename in filenames:
        assert filename.endswith(".srec"), (
            f"{filename} is not a .srec file")
        filepath = os.path.join(path, filename)
        with decoder_time_accumulator.execute():
            x = coder.decode(filepath)
            x = x.byte().squeeze(0).cpu()
        img = T.functional.to_pil_image(x)
        img.save(os.path.join(save_path, f"{filename[:-5]}.png"))
        print(
            "Decomp: "
            f"{decoder_time_accumulator.mean_time_spent():.3f};\t"
            "Decomp Time By Scale: ",
            end="")
        decomp_scale_times = coder.decomp_scale_times()
        print(
            ", ".join(f"{scale_time:.3f}" for scale_time in decomp_scale_times),
            end="; ")

        total_num_bytes += os.path.getsize(filepath)
        total_num_subpixels += np.prod(x.size())

        print(
            f"Bpsp: {total_num_bytes*8/total_num_subpixels:.3f}", end="\r")
    print()


if __name__ == "__main__":
    main()
