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

--------------------------------------------------------------------------------

Very thin wrapper around torchac, for arithmetic coding.

"""
import torch

from src.l3c import logistic_mixture as lm
from src.torchac import torchac


class ArithmeticCoder(object):
    def __init__(self, L):
        self.L = L
        self._cached_cdf = None

    def range_encode(self, data: torch.Tensor, cdf):
        """
        :param data: data to encode
        :param cdf: cdf to use, either a NHWLp matrix or instance of CDFOut
        :return: data encode to a bytes string
        """
        assert len(data.shape) == 3, data.shape

        data = data.to('cpu', non_blocking=True)
        assert data.dtype == torch.int16, 'Wrong dtype: {}'.format(data.dtype)
        data = data.reshape(-1).contiguous()

        if isinstance(cdf, lm.CDFOut):
            logit_probs_c_sm, means_c, log_scales_c, _, targets = cdf
            out_bytes = torchac.encode_logistic_mixture(
                targets, means_c, log_scales_c, logit_probs_c_sm, data)
        else:
            _, _, _, Lp = cdf.shape
            assert Lp == self.L + 1, (Lp, self.L)
            out_bytes = torchac.encode_cdf(cdf, data)

        return out_bytes

    def range_decode(self, encoded_bytes, cdf):
        """
        :param encoded_bytes: bytes encoded by range_encode
        :param cdf: cdf to use, either a NHWLp matrix or instance of CDFOut
        :return: decoded matrix as np.int16, NHW
        """
        if isinstance(cdf, lm.CDFOut):
            logit_probs_c_sm, means_c, log_scales_c, _, targets = cdf

            N, _, H, W = means_c.shape

            decoded = torchac.decode_logistic_mixture(
                targets, means_c, log_scales_c, logit_probs_c_sm,
                encoded_bytes)

        else:
            N, H, W, Lp = cdf.shape
            assert Lp == self.L + 1, (Lp, self.L)
            decoded = torchac.decode_cdf(cdf, encoded_bytes)

        return decoded.reshape(N, H, W)
