"""
Copyright 2019, ETH Zurich

This file is part of L3C-PyTorch.

L3C-PyTorch is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

L3C-PyTorch is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with L3C-PyTorch.  If not, see <https://www.gnu.org/licenses/>.
"""
import os
from typing import List, Tuple, Union

import numpy as np
import torch
from torch.nn import functional as F

from src import configs
from src import data as lc_data
from src import network
from src.l3c import coders, coders_helpers
from src.l3c import logistic_mixture as lm
from src.l3c import quantizer, timer


class Bitcoding(object):
    """
    Class to encode an image to a file and decode it. Saves timings of 
    individual steps to `times`.
    If `compare_with_theory = True`, also compares actual bitstream 
    size to size predicted by cross entropy. Note
    that this is slower because we need to evaluate the loss.
    """

    def __init__(
            self,
            compressor: network.Compressor,
    ) -> None:
        self.compressor = compressor
        self.total_num_bytes = 0
        self.total_num_subpixels = 0
        self.log_likelihood_bits = network.Bits()
        self.file_sizes: np.ndarray = 0.  # type: ignore
        self.scale_timers = [
            timer.TimeAccumulator() for _ in range(configs.scale + 1)]

    def encode(
            self, x: torch.Tensor, pout: str
    ) -> Tuple[network.Bits, List[int]]:
        """
        Encode image to disk at path `p`.
        :param img: uint8 tensor of shape CHW or 1CHW
        :param pout: path
        """
        assert not os.path.isfile(pout), f"{pout} already exists"
        assert x.dtype == torch.float32
        x = x.cuda()
        self.compressor.eval()

        with torch.no_grad():
            bits = self.compressor(x)

        entropy_coding_bytes = []  # bytes used by different scales

        with open(pout, "wb") as fout:
            write_shape(x.shape, fout)

            for _, (y_i, lm_probs, levels) in enumerate(bits.probs):
                # with self.times.prefix_scope(f"[{key}]"):
                if lm_probs is None:
                    assert levels != -1, levels
                    self.encode_uniform(y_i, levels, fout)
                else:
                    assert levels == -1, levels
                    y_i = y_i.contiguous()
                    entropy_coding_bytes.append(
                        self.encode_scale(
                            y_i,
                            lm_probs.probs.contiguous(),
                            y_i, self.compressor.loss_fn, fout))
        return bits, entropy_coding_bytes

    def decomp_scale_times(self) -> List[float]:
        return [acc.mean_time_spent() for acc in self.scale_timers]

    def decode(self, pin) -> torch.Tensor:
        """
        :param pin:  Path where image is stored
        :return: Decoded image, as 1CHW, long
        """
        with torch.no_grad(), open(pin, "rb") as fin:
            with self.scale_timers[0].execute():
                H, W = read_shapes(fin)
                shapes = lc_data.get_shapes(H, W)
                x = self.decode_uniform(256, fin, shapes[-1])

            ctx: torch.Tensor = 0.  # type: ignore
            for dec, ctx_upsampler, prev_shape, \
                shape, scale_timer in zip(  # type: ignore
                    self.compressor.decs, self.compressor.ctx_upsamplers,
                    shapes[::-1], shapes[-2::-1],
                    self.scale_timers[1:]):

                with scale_timer.execute():
                    deltas = self.decode_uniform(4, fin, prev_shape)
                    deltas = quantizer.to_bn(
                        deltas, x_min=-0.25, x_max=0.5, L=4)
                    y = x + deltas

                with scale_timer.execute():
                    ctx = ctx_upsampler(ctx)
                    if not isinstance(ctx, float):
                        ctx = ctx[..., :prev_shape[-2], :prev_shape[-1]]
                    gen = dec.forward_probs(y, ctx)
                    x_slices: List[torch.Tensor] = []
                    try:
                        for i, (h, w) in enumerate(
                                lc_data.get_2x2_shapes(*shape)):
                            if i == 0:
                                lm_probs = next(gen)
                            else:
                                lm_probs = gen.send(x_slices[i-1])
                            x_i = self.decode_scale(
                                self.compressor.loss_fn, lm_probs.probs,
                                fin, (h, w)).float()
                            x_slices.append(lc_data.pad(
                                x_i, y.shape[-2], y.shape[-1]))
                    except StopIteration as e:
                        last_pixels, ctx = e.value
                        x_slices.append(last_pixels)

                    x = lc_data.join_2x2(x_slices, shape)

        assert x is not None  # assert decoding worked
        return x

    def encode_uniform(self, S: torch.Tensor, levels: int, fout) -> int:
        """ encode coarsest scale, for which we assume a uniform prior. """
        # Because our model only stores RGB values and rounding bits as uniform,
        # levels is either 4 or 256.
        # This means we can apply simple bit manipulations to store these as uniform.
        assert levels == 4 or levels == 256, levels

        if levels == 4:
            S = S.reshape(-1)
            S = F.pad(S, [0, S.shape[0] % 4])
            assert S.shape[0] % 4 == 0
            N = S.shape[0]//4

        S = S.type(torch.cuda.ByteTensor)  # type: ignore
        N = S.shape[0]//4 if levels == 4 else np.prod(S.size())

        if levels == 4:
            S = (S[:N] * 64 +
                 S[N:2*N] * 16 +
                 S[2*N:3*N] * 4 +
                 S[3*N:])

        S_np = S.cpu().numpy()
        S_buffer = S_np.tobytes()
        fout.write(S_buffer)
        return N

    def decode_uniform(
            self, levels: int, fin, shape: Tuple[int, int]
    ) -> torch.Tensor:
        """ decode coarsest scale, for which we assume a uniform prior. """
        # Because our model only stores RGB values and rounding bits as uniform,
        # levels is either 4 or 256.
        assert levels == 4 or levels == 256, levels

        num_elements = np.prod(shape) * 3
        buffer_size = (num_elements+3)//4 if levels == 4 else num_elements
        buffer = fin.read(buffer_size)
        x_np = np.frombuffer(buffer, dtype=np.uint8)
        x = torch.from_numpy(x_np)
        if torch.cuda.is_available():
            x = x.cuda()

        if levels == 4:
            x = torch.stack([
                (x & 0b11000000) // 64,
                (x & 0b00110000) // 16,
                (x & 0b00001100) // 4,
                (x & 0b00000011)]).reshape(-1)
            padding = num_elements % 4
            assert x.shape[0] == num_elements + padding
            x = x[:num_elements]

        x = x.reshape(-1, 3, shape[0], shape[1])
        x = x.float()
        return x

    def encode_scale(self, S, l, bn, dmll, fout):
        """ Encode scale `scale`. """

        r = coders.ArithmeticCoder(dmll.L)

        # We encode channel by channel, because that's what's needed for the RGB scale. For s > 0, this could be done
        # in parallel for all channels
        def encoder(c, C_cur):
            S_c = S[:, c, ...].to(torch.int16)
            encoded = r.range_encode(S_c, cdf=C_cur)
            write_num_bytes_encoded(len(encoded), fout)
            fout.write(encoded)
            # yielding always bottleneck and extra_info
            return bn[:, c, ...], len(encoded)

        _, entropy_coding_bytes_per_c = self.code_with_cdf(
            l, bn.shape, encoder, dmll)

        return sum(entropy_coding_bytes_per_c)

    def decode_scale(self,
                     dmll: lm.DiscretizedMixLogisticLoss,
                     l: torch.Tensor,
                     fin,
                     shape: Tuple[int, int],
                     ) -> torch.Tensor:
        H, W = shape
        C = 3
        l = l[..., :H, :W].contiguous()
        r = coders.ArithmeticCoder(dmll.L)

        # We decode channel by channel, see `encode_scale`.
        def decoder(_, C_cur):
            num_bytes = read_num_bytes_encoded(fin)
            encoded = fin.read(num_bytes)
            S_c = r.range_decode(
                encoded, cdf=C_cur).reshape(1, H, W)
            # TODO: do directly in the extension
            S_c = S_c.to(l.device, non_blocking=True)
            bn_c = dmll.to_bn(S_c)
            # yielding always bottleneck and extra_info (=None here)
            return bn_c, None

        bn, _ = self.code_with_cdf(l, (1, C, H, W), decoder, dmll)

        return bn

    def code_with_cdf(self, l, bn_shape, bn_coder, dmll):
        """
        :param l: predicted distribution, i.e., NKpHW, see DiscretizedMixLogisticLoss
        :param bn_shape: shape of the bottleneck to encode/decode
        :param bn_coder: function with signature (c: int, C_cur: CDFOut) -> (bottleneck[c], extra_info_c). This is
        called for every channel of the bottleneck, with C_cur == CDF to use to encode/decode the channel. It shoud
        return the bottleneck[c].
        :param dmll: instance of DiscretizedMixLogisticLoss
        :return: decoded bottleneck, list of all extra info produced by `bn_coder`.
        """
        N, C, H, W = bn_shape
        coding = coders_helpers.CodingCDFNonshared(
            l, total_C=C, dmll=dmll)

        # needed also while encoding to get next C
        decoded_bn = torch.zeros(N, C, H, W, dtype=torch.float32).to(l.device)
        extra_info = []

        for c in range(C):
            C_cond_cur = coding.get_next_C(decoded_bn)
            decoded_bn[:, c, ...], extra_info_c = bn_coder(
                c, C_cond_cur)
            extra_info.append(extra_info_c)

        return decoded_bn, extra_info


def write_shape(
        shape: Union[torch.Size, Tuple[int, int, int, int]],
        fout
) -> int:
    """
    Write tuple (C,H,W) to file, given shape 1CHW.
    :return number of bytes written
    """
    assert len(shape) == 4 and shape[0] == 1 and shape[1] == 3, shape
    assert shape
    assert shape[2] < 2**16, shape
    assert shape[3] < 2**16, shape
    write_bytes(fout, [np.uint16, np.uint16], shape[2:])
    return 4


def read_shapes(fin) -> Tuple[int, int]:
    shape = tuple(map(int, read_bytes(fin, [np.uint16, np.uint16])))
    assert len(shape) == 2, shape
    return shape  # type: ignore


def write_num_bytes_encoded(num_bytes, fout):
    assert num_bytes < 2**32
    write_bytes(fout, [np.uint32], [num_bytes])
    return 2  # number of bytes written


def read_num_bytes_encoded(fin):
    return int(list(read_bytes(fin, [np.uint32]))[0])


def write_bytes(f, ts, xs):
    for t, x in zip(ts, xs):
        f.write(t(x).tobytes())


def read_bytes(f, ts):
    for t in ts:
        num_bytes_to_read = t().itemsize
        yield np.frombuffer(f.read(num_bytes_to_read), t, count=1)
