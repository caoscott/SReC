import io
import os
from typing import Iterable, List, Tuple

import numpy as np
import PIL.Image as Image
import torch
import torch.utils.data as data
from torch.nn import functional as F

from src import configs, util


class PreemptiveRandomSampler(data.Sampler):
    def __init__(self, indices: List[int], index: int) -> None:
        super().__init__("None")
        self.indices: List[int]
        self.indices = indices
        self.index = index

    def __iter__(self) -> Iterable[int]:  # type: ignore
        size = len(self.indices)
        assert 0 <= self.index < size, \
            f"0 <= {self.index} < {size} violated"
        while self.index < size:
            yield self.indices[self.index]
            self.index += 1
        self.index = 0
        self.indices = torch.randperm(size).tolist()

    def __len__(self) -> int:
        return len(self.indices)


def average_downsamples(x: torch.Tensor) -> List[torch.Tensor]:
    downsampled = []
    for _ in range(configs.scale):
        downsampled.append(x.detach())
        x = F.avg_pool2d(pad_to_even(util.tensor_round(x)), 2)
    downsampled.append(x.detach())
    return downsampled


class ImageFolder(data.Dataset):
    """ Generic Dataset class for a directory full of images given 
        a list of image filenames. Can be used for any unsupervised 
        learning task without labels.
    """

    def __init__(self,
                 dir_path: str,
                 filenames: List[str],
                 scale: int,
                 transforms,  # torchvision Transform
                 ) -> None:
        """ param dir_path: Path to image directory
            param filenames: List of image filenames in the directory.
            param transforms: a torchvision Transform that can be applied to 
                the image.
        """
        self.scale = scale
        self.dir_path = dir_path
        self.filenames = filenames
        assert filenames, f"{filenames} is empty"
        self.transforms = transforms

    def to_tensor_not_normalized(self, pic: Image) -> torch.Tensor:
        """ copied from PyTorch functional.to_tensor.
            removed final .div(255.) """
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        elif pic.mode == 'F':
            img = torch.from_numpy(np.array(pic, np.float32, copy=False))
        elif pic.mode == '1':
            img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
        else:
            img = torch.ByteTensor(  # type: ignore
                torch.ByteStorage.from_buffer(pic.tobytes()))  # type: ignore
        # PIL image mode: L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float()

    def load(self,
             fp: io.BytesIO
             ) -> torch.Tensor:
        # x: 3HW
        img = Image.open(fp)
        x = self.transforms(img)
        return self.to_tensor_not_normalized(x)

    def read(self, filename: str) -> bytes:
        with open(os.path.join(self.dir_path, filename), "rb") as f:
            return f.read()

    def __getitem__(
            self,
            idx: int
    ) -> Tuple[str, torch.Tensor]:  # type: ignore
        filename = self.filenames[idx]
        file_bytes = self.read(filename)
        img_data = self.load(io.BytesIO(file_bytes))
        return filename, img_data

    def __len__(self) -> int:
        return len(self.filenames)


def pad_to_even(x: torch.Tensor) -> torch.Tensor:
    _, _, h, w = x.size()
    pad_right = w % 2 == 1
    pad_bottom = h % 2 == 1
    padding = [0, 1 if pad_right else 0, 0, 1 if pad_bottom else 0]
    x = F.pad(x, padding, mode="replicate")
    return x


def pad(x: torch.Tensor, H: int, W: int) -> torch.Tensor:
    _, _, xH, xW = x.size()
    padding = [0, W - xW, 0, H - xH]
    return F.pad(x, padding, mode="replicate")


def join_2x2(padded_slices: List[torch.Tensor],
             shape: Tuple[int, int]
             ) -> torch.Tensor:
    assert len(padded_slices) == 4, len(padded_slices)
    # 4 N 3 H W
    x = torch.stack(padded_slices)
    # N 3 4 H W
    x = x.permute(1, 2, 0, 3, 4)
    N, _, _, H, W = x.size()
    # N 12 H W
    x = x.contiguous().view(N, -1, H, W)
    # N 3 2H 2W
    x = F.pixel_shuffle(x, upscale_factor=2)
    # return x[..., :unpad_h, :unpad_w]
    return x[..., :shape[-2], :shape[-1]]


def get_shapes(H: int, W: int) -> List[Tuple[int, int]]:
    shapes = [(H, W)]
    h = H
    w = W
    for _ in range(3):
        h = (h + 1) // 2
        w = (w + 1) // 2
        shapes.append((h, w))
    return shapes


def get_2x2_shapes(
        H: int, W: int
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    top_H = (H + 1) // 2
    left_W = (W + 1) // 2
    bottom_H = H - top_H
    right_W = W - left_W
    return (
        (top_H, left_W), (top_H, right_W),
        (bottom_H, left_W), (bottom_H, right_W))
