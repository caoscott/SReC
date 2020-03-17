from collections import defaultdict
from typing import (DefaultDict, Generator, KeysView, List, NamedTuple,
                    Optional, Tuple)

import numpy as np
import torch
from torch import nn

from src import configs, data, util
from src.l3c import edsr
from src.l3c import logistic_mixture as lm
from src.l3c import prob_clf, quantizer


class LogisticMixtureProbability(NamedTuple):
    name: str
    pixel_index: int
    probs: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor


Probs = Tuple[torch.Tensor, Optional[LogisticMixtureProbability], int]


class Bits:
    """
    Tracks bpsps from different parts of the pipeline for one forward pass.
    """

    def __init__(self) -> None:
        assert configs.collect_probs or configs.log_likelihood, (
            configs.collect_probs, configs.log_likelihood)
        self.key_to_bits: DefaultDict[
            str, torch.Tensor] = defaultdict(float)  # type: ignore
        self.key_to_sizes: DefaultDict[str, int] = defaultdict(int)
        self.probs: List[Probs] = []

    def add_with_size(
            self, key: str, nll_sum: torch.Tensor, size: int,
    ) -> None:
        if configs.log_likelihood:
            assert key not in self.key_to_bits, f"{key} already exists"
            # Divide by np.log(2) to convert from natural log to log base 2
            self.key_to_bits[key] = nll_sum / np.log(2)
            self.key_to_sizes[key] = size

    def add(self, key: str, nll: torch.Tensor) -> None:
        self.add_with_size(
            key, nll.sum(), np.prod(nll.size()))

    def add_lm(
            self, y_i: torch.Tensor,
            lm_probs: LogisticMixtureProbability,
            loss_fn: lm.DiscretizedMixLogisticLoss) -> None:
        assert lm_probs.probs.shape[-2:] == y_i.shape[-2:], (
            lm_probs.probs.shape, y_i.shape)
        if configs.log_likelihood:
            nll = loss_fn(y_i, lm_probs.probs)
            self.add(lm_probs.name, nll)
        if configs.collect_probs:
            self.probs.append((y_i, lm_probs, -1))

    def add_uniform(
            self,
            key: str,
            y_i: torch.Tensor,
            levels: int = 256) -> None:
        if configs.log_likelihood:
            size = np.prod(y_i.size())
            nll_sum = np.log(levels) * size
            self.add_with_size(key, nll_sum, size)
        if configs.collect_probs:
            self.probs.append((y_i, None, levels))

    def get_bits(self, key: str) -> torch.Tensor:
        return self.key_to_bits[key]

    def get_size(self, key: str) -> int:
        return self.key_to_sizes[key]

    def get_keys(self) -> KeysView:
        return self.key_to_bits.keys()

    def get_self_bpsp(self, key: str) -> torch.Tensor:
        return self.key_to_bits[key]/self.key_to_sizes[key]

    def get_scaled_bpsp(self, key: str, inp_size: int) -> torch.Tensor:
        return self.key_to_bits[key]/inp_size

    def get_total_bpsp(self, inp_size: int) -> torch.Tensor:
        return sum(self.key_to_bits.values()) / inp_size  # type: ignore

    def update(self, other: "Bits") -> "Bits":
        # Used by Compressor to aggregate bits from decoder.
        assert len(self.get_keys() & other.get_keys()) == 0, \
            f"{self.get_keys()} and {other.get_keys()} intersect."
        self.key_to_bits.update(other.key_to_bits)
        self.key_to_sizes.update(other.key_to_sizes)
        self.probs += other.probs
        return self

    def add_bits(self, other: "Bits") -> "Bits":
        keys = other.get_keys()
        assert keys == self.get_keys() or len(self.get_keys()) == 0, (
            f"{self.get_keys()} != {keys}")

        for key in keys:
            self.key_to_bits[key] += other.get_bits(key)
            self.key_to_sizes[key] += other.get_size(key)
            # Don't do anything with self.key_to_probs at the moment.
        return self


class PixDecoder(nn.Module):
    """ Super-resolution based decoder for pixel-based factorization. """

    def __init__(self, scale: int) -> None:
        super().__init__()
        self.loss_fn = lm.DiscretizedMixLogisticLoss(rgb_scale=True)
        self.scale = scale

    def forward_probs(
            self,
            x: torch.Tensor,
            ctx: torch.Tensor
    ) -> Generator[LogisticMixtureProbability, torch.Tensor,
                   Tuple[torch.Tensor, torch.Tensor]]:
        raise NotImplementedError

    def forward(self,  # type: ignore
                x: torch.Tensor,
                y: torch.Tensor,
                ctx: torch.Tensor,
                ) -> Tuple[Bits, torch.Tensor]:
        bits = Bits()

        # Check y are filled with integers.
        # y.long().float() == y
        if __debug__:
            not_int = y.long().float() != y
            assert not torch.any(not_int), y[not_int]

        # mode is used to key tensorboard loggings
        mode = "train" if self.training else "eval"
        deltas = x - util.tensor_round(x)
        bits.add_uniform(
            f"{mode}/{self.scale}_rounding",
            quantizer.to_sym(deltas, x_min=-0.25, x_max=0.5, L=4),
            levels=4)

        _, _, x_h, x_w = x.size()
        if not isinstance(ctx, float):
            ctx = ctx[..., :x_h, :x_w]

        # Divide pixels of y into 2x2 grids and slice y by pixels on
        # different util of grid.
        # y: N 3 H W -> N 4 3 H/2 W/2
        y_slices = group_2x2(y)

        gen = self.forward_probs(x, ctx)
        try:
            for i, y_slice in enumerate(y_slices):
                if i == 0:
                    lm_probs = next(gen)
                else:
                    lm_probs = gen.send(y_slices[i-1])
                _, _, h, w = y_slice.size()
                lm_probs = LogisticMixtureProbability(
                    name=lm_probs.name,
                    pixel_index=lm_probs.pixel_index,
                    probs=lm_probs.probs[..., :h, :w],
                    lower=lm_probs.lower[..., :h, :w],
                    upper=lm_probs.upper[..., :h, :w])
                bits.add_lm(y_slice, lm_probs, self.loss_fn)
        except StopIteration as e:
            last_pixels, ctx = e.value
            last_slice = y_slices[-1]
            _, _, last_h, last_w = last_slice.size()
            last_pixels = last_pixels[..., : last_h, : last_w]
            assert torch.all(last_pixels == last_slice), (
                last_pixels[last_pixels != last_slice],
                last_slice[last_pixels != last_slice])

        return bits, ctx


class StrongPixDecoder(PixDecoder):
    def __init__(self, scale: int) -> None:
        super().__init__(scale)
        # Input: N 3 H W
        # Output: N C H W
        self.rgb_decs = nn.ModuleList([
            edsr.EDSRDec(
                3*i, configs.n_feats,
                resblocks=configs.resblocks, tail="conv")
            for i in range(1, 4)
        ])
        self.mix_logits_prob_clf = nn.ModuleList([
            prob_clf.AtrousProbabilityClassifier(
                configs.n_feats, C=3, K=configs.K,
                num_params=self.loss_fn._num_params)
            for _ in range(1, 4)
        ])
        self.feat_convs = nn.ModuleList([
            util.conv(configs.n_feats, configs.n_feats, 3) for _ in range(1, 4)
        ])
        assert (len(self.rgb_decs) == len(self.mix_logits_prob_clf) ==
                len(self.feat_convs)), (
                    f"{len(self.rgb_decs)}, "
                    f"{len(self.mix_logits_prob_clf)}, {len(self.feat_convs)}"
        )

    def forward_probs(
            self,
            x: torch.Tensor,
            ctx: torch.Tensor,
    ) -> Generator[LogisticMixtureProbability, torch.Tensor,
                   Tuple[torch.Tensor, torch.Tensor]]:
        # mode is used to key tensorboard loggings
        mode = "train" if self.training else "eval"
        # x: N 3 H W, [0, 255]
        # pix_sum: N 3 H W, [0, 1020]
        pix_sum = x * 4
        xy_normalized = x / 127.5 - 1
        y_i = torch.tensor([], device=x.device)
        z: torch.Tensor = 0.  # type: ignore

        for i, (rgb_dec, clf, feat_conv) in enumerate(
                zip(self.rgb_decs,  # type: ignore
                    self.mix_logits_prob_clf, self.feat_convs)):
            xy_normalized = torch.cat((xy_normalized, y_i / 127.5 - 1), dim=1)
            z = rgb_dec(xy_normalized, ctx)
            ctx = feat_conv(z)

            probs = clf(z)
            lower = torch.max(
                pix_sum - (3 - i) * 255, torch.tensor(0., device=x.device))
            upper = torch.min(
                pix_sum, torch.tensor(255., device=x.device))

            y_i = yield LogisticMixtureProbability(
                f"{mode}/{self.scale}_{i}", i, probs, lower, upper)
            y_i = data.pad(y_i, x.shape[-2], x.shape[-1])
            pix_sum -= y_i

        # Last pixel in 2x2 grid should be <= 255 and >= 0
        return pix_sum, ctx


def group_2x2(
        x: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """ Group 2x2 patches of x on its own channel
        param x: N C H W
        returns: Tuple[N 4 C H/2 W/2]
    """
    _, _, h, w = x.size()
    # assert h % 2 == 0, f"{x.shape} does not satisfy h % 2 == 0"
    # assert w % 2 == 0, f"{x.shape} does not satisfy w % 2 == 0"
    x_even_height = x[:, :, 0:h:2, :]
    x_odd_height = x[:, :, 1:h:2, :]
    return (
        x_even_height[:, :, :, 0:w:2],
        x_even_height[:, :, :, 1:w:2],
        x_odd_height[:, :, :, 0:w:2],
        x_odd_height[:, :, :, 1:w:2])


class Compressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        assert configs.scale >= 0, configs.scale

        self.loss_fn = lm.DiscretizedMixLogisticLoss(rgb_scale=True)
        self.ctx_upsamplers = nn.ModuleList([
            nn.Identity(),  # type: ignore
            *[edsr.Upsampler(scale=2, n_feats=configs.n_feats)
              for _ in range(configs.scale-1)]
        ] if configs.scale > 0 else [])
        self.decs = nn.ModuleList([
            StrongPixDecoder(i) for i in range(configs.scale)
        ])
        assert len(self.ctx_upsamplers) == len(self.decs), \
            f"{len(self.ctx_upsamplers)}, {len(self.decs)}"
        self.nets = nn.ModuleList([
            self.ctx_upsamplers, self.decs,
        ])

    def forward(self,  # type: ignore
                x: torch.Tensor
                ) -> Bits:
        downsampled = data.average_downsamples(x)
        assert len(downsampled)-1 == len(self.decs), (
            f"{len(downsampled)-1}, {len(self.decs)}")

        mode = "train" if self.training else "eval"
        bits = Bits()
        bits.add_uniform(f"{mode}/codes_0", util.tensor_round(downsampled[-1]))

        ctx = 0.
        for dec, ctx_upsampler, x, y, in zip(  # type: ignore
                self.decs, self.ctx_upsamplers,
                downsampled[::-1], downsampled[-2::-1]):
            ctx = ctx_upsampler(ctx)
            dec_bits, ctx = dec(x, util.tensor_round(y), ctx)
            bits.update(dec_bits)
        return bits
