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

This class is based on the TensorFlow code of PixelCNN++:
    https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py
In contrast to that code, we predict mixture weights pi for each channel, i.e., mixture weights are "non-shared".
Also, x_min, x_max and L are parameters, and we implement a function to get the CDF of a channel.

# ------
# Naming
# ------

Note that we use the following names through the code, following the code PixelCNN++:
    - x: targets, e.g., the RGB image for scale 0
    - l: for the output of the network;
      In Fig. 2 in our paper, l is the final output, denoted with p(z^(s-1) | f^(s)), i.e., it contains the parameters
      for the mixture weights.
"""

# from collections import namedtuple
from typing import NamedTuple, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from src import configs
from src.l3c import quantizer

_NUM_PARAMS_RGB = 4  # mu, sigma, pi, lambda
_NUM_PARAMS_OTHER = 3  # mu, sigma, pi
_LOG_SCALES_MIN = -7.


class CDFOut(NamedTuple):
    logit_probs_c_sm: torch.Tensor
    means_c: torch.Tensor
    log_scales_c: torch.Tensor
    K: int
    targets: torch.Tensor


def non_shared_get_Kp(K, C, num_params):
    """ Get Kp=number of channels to predict. 
        See note where we define _NUM_PARAMS_RGB above """
    return num_params * C * K


def non_shared_get_K(Kp: int, C: int, num_params: int) -> int:
    """ Inverse of non_shared_get_Kp, get back K=number of mixtures """
    return Kp // (num_params * C)


# --------------------------------------------------------------------------------
class DiscretizedMixLogisticLoss(nn.Module):
    def __init__(self, rgb_scale: bool, x_min=0, x_max=255, L=256):
        """
        :param rgb_scale: Whether this is the loss for the RGB scale. In that case,
            use_coeffs=True
            _num_params=_NUM_PARAMS_RGB == 4, since we predict coefficients lambda. See note above.
        :param x_min: minimum value in targets x
        :param x_max: maximum value in targets x
        :param L: number of symbols
        """
        super(DiscretizedMixLogisticLoss, self).__init__()
        self.rgb_scale = rgb_scale
        self.x_min = x_min
        self.x_max = x_max
        self.L = L
        # whether to use coefficients lambda to weight
        # means depending on previously outputed means.
        self.use_coeffs = rgb_scale
        # P means number of different variables contained
        # in l, l means output of network
        self._num_params = (
            _NUM_PARAMS_RGB if rgb_scale else
            _NUM_PARAMS_OTHER)

        # NOTE: in contrast to the original code,
        # we use a sigmoid (instead of a tanh)
        # The optimizer seems to not care,
        # but it would probably be more principaled to use a tanh
        # Compare with L55 here:
        # https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py#L55
        self._nonshared_coeffs_act = torch.sigmoid

        # Adapted bounds for our case.
        self.bin_width = (x_max - x_min) / (L-1)
        self.x_lower_bound = x_min + 0.001
        self.x_upper_bound = x_max - 0.001

        self._extra_repr = 'DMLL: x={}, L={}, coeffs={}, P={}, bin_width={}'.format(
            (self.x_min, self.x_max), self.L, self.use_coeffs, self._num_params, self.bin_width)

    def extra_repr(self):
        return self._extra_repr

    @staticmethod
    def to_per_pixel(entropy, C):
        N, H, W = entropy.shape
        return entropy.sum() / (N*C*H*W)  # NHW -> scalar

    def to_sym(self, x):
        return quantizer.to_sym(x, self.x_min, self.x_max, self.L)

    def to_bn(self, S):
        return quantizer.to_bn(S, self.x_min, self.x_max, self.L)

    def cdf_step_non_shared(self, l, targets, c_cur, C, x_c=None) -> CDFOut:
        assert c_cur < C

        # NKHW         NKHW     NKHW
        logit_probs_c, means_c, log_scales_c, K = self._extract_non_shared_c(
            c_cur, C, l, x_c)

        logit_probs_c_softmax = F.softmax(logit_probs_c, dim=1)  # NKHW, pi_k
        return CDFOut(
            logit_probs_c_softmax, means_c,
            log_scales_c, K, targets.to(l.device))

    def sample(self, l, C):
        return self._non_shared_sample(l, C)

    def log_cdf(self, lo, hi, means, log_scales):
        assert torch.all(lo <= hi), f"{lo[lo > hi]} > {hi[lo > hi]}"
        assert lo.min() >= self.x_min and hi.max() <= self.x_max, \
            '{},{} not in {},{}'.format(
                lo.min(), hi.max(), self.x_min, self.x_max)

        centered_lo = lo - means  # NCKHW
        centered_hi = hi - means

        # Calc cdf_delta
        # all of the following is NCKHW
        # <= exp(7), is exp(-sigma), inverse std. deviation, i.e., sigma'
        inv_stdv = torch.exp(-log_scales)
        # sigma' * (x - mu + 0.5)
        # S(sigma' * (x - mu - 1/255)) = 1 / (1 + exp(sigma' * (x - mu - 1/255))
        normalized_lo = inv_stdv * (
            centered_lo - self.bin_width/2)  # sigma' * (x - mu - 1/255)
        lo_cond = (lo >= self.x_lower_bound).float()
        # log probability for edge case of 0
        cdf_lo = lo_cond * torch.sigmoid(normalized_lo)
        normalized_hi = inv_stdv * (centered_hi + self.bin_width/2)
        hi_cond = (hi <= self.x_upper_bound).float()
        cdf_hi = hi_cond * torch.sigmoid(normalized_hi) + (1 - hi_cond)  # * 1.
        # S(sigma' * (x - mu + 1/255))
        # NCKHW, cdf^k(c)
        cdf_delta = cdf_hi - cdf_lo
        log_cdf_delta = torch.log(torch.clamp(cdf_delta, min=1e-12))

        assert not torch.any(
            log_cdf_delta > 1e-6
        ), f"{log_cdf_delta[log_cdf_delta > 1e-6]}"
        return log_cdf_delta

    def forward(  # type: ignore
            self, x: torch.Tensor, l: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param x: labels, i.e., NCHW, float
        :param l: predicted distribution, i.e., NKpHW, see above
        :return: log-likelihood, as NHW if shared, NCHW if non_shared pis
        """
        assert x.min() >= self.x_min and x.max() <= self.x_max, \
            f'{x.min()},{x.max()} not in {self.x_min},{self.x_max}'

        # Extract ---
        #  NC1HW     NCKHW      NCKHW  NCKHW
        x, logit_pis, means, log_scales, _ = self._extract_non_shared(x, l)

        log_probs = self.log_cdf(x, x, means, log_scales)

        # combine with pi, NCKHW, (-inf, 0]
        log_weights = F.log_softmax(logit_pis, dim=2)
        log_probs_weighted = log_weights + log_probs

        # final log(P), NCHW
        nll = -torch.logsumexp(log_probs_weighted, dim=2)
        return nll

    def _extract_non_shared(self, x, l):
        """
        :param x: targets, NCHW
        :param l: output of net, NKpHW, see above
        :return:
            x NC1HW,
            logit_probs NCKHW (probabilites of scales, i.e., \pi_k)
            means NCKHW,
            log_scales NCKHW (variances),
            K (number of mixtures)
        """
        N, C, H, W = x.shape
        Kp = l.shape[1]

        K = non_shared_get_K(Kp, C, self._num_params)

        # we have, for each channel: K pi / K mu / K sigma / [K coeffs]
        # note that this only holds for C=3 as for other channels,
        # there would be more than 3*K coeffs
        # but non_shared only holds for the C=3 case
        l = l.reshape(N, self._num_params, C, K, H, W)

        logit_probs = l[:, 0, ...]  # NCKHW
        means = l[:, 1, ...]  # NCKHW
        log_scales = torch.clamp(
            l[:, 2, ...], min=_LOG_SCALES_MIN)  # NCKHW, is >= -7
        x = x.reshape(N, C, 1, H, W)

        if self.use_coeffs:
            # Coefficients only supported for multiples of 3,
            # see note where we define
            # _NUM_PARAMS_RGB NCKHW, basically coeffs_g_r, coeffs_b_r, coeffs_b_g
            assert C == 3, C
            # Each NCKHW
            coeffs = self._nonshared_coeffs_act(l[:, 3, ...])
            # each NKHW
            coeffs_g_r = coeffs[:, 0, ...]
            coeffs_b_r = coeffs[:, 1, ...]
            coeffs_b_g = coeffs[:, 2, ...]
            # NCKHW
            means = torch.stack(
                (means[:, 0, ...],
                 means[:, 1, ...] + coeffs_g_r * x[:, 0, ...],
                 means[:, 2, ...] + coeffs_b_r * x[:, 0, ...]
                                  + coeffs_b_g * x[:, 1, ...]),
                dim=1)

        means = torch.clamp(means, min=self.x_min, max=self.x_max)
        assert means.shape == (N, C, K, H, W), (means.shape, (N, C, K, H, W))
        return x, logit_probs, means, log_scales, K

    def _extract_non_shared_c(
            self, c: int, C: int, l: torch.Tensor,
            x: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Same as _extract_non_shared but only for c-th channel, used to get CDF
        """
        assert c < C, f'{c} >= {C}'

        N, Kp, H, W = l.shape
        K = non_shared_get_K(Kp, C, self._num_params)

        l = l.reshape(N, self._num_params, C, K, H, W)
        logit_probs_c = l[:, 0, c, ...]  # NKHW
        means_c = l[:, 1, c, ...]  # NKHW
        log_scales_c = torch.clamp(
            l[:, 2, c, ...], min=_LOG_SCALES_MIN)  # NKHW, is >= -7

        if self.use_coeffs and c != 0:
            # N C K H W, coeffs_g_r, coeffs_b_r, coeffs_b_g
            unscaled_coeffs = l[:, 3, ...]
            if c == 1:
                assert x is not None
                coeffs_g_r = self._nonshared_coeffs_act(
                    unscaled_coeffs[:, 0, ...])  # NKHW
                means_c += coeffs_g_r * x[:, 0, ...]
            elif c == 2:
                assert x is not None
                coeffs_b_r = self._nonshared_coeffs_act(
                    unscaled_coeffs[:, 1, ...])  # NKHW
                coeffs_b_g = self._nonshared_coeffs_act(
                    unscaled_coeffs[:, 2, ...])  # NKHW
                means_c += coeffs_b_r * x[:, 0, ...] + coeffs_b_g * x[:, 1, ...]

        #      NKHW           NKHW     NKHW
        return logit_probs_c, means_c, log_scales_c, K

    def _non_shared_sample(self, l, C):
        """ sample from model """
        N, Kp, H, W = l.shape
        K = non_shared_get_K(Kp, C, self._num_params)
        l = l.reshape(N, self._num_params, C, K, H, W)

        logit_probs = l[:, 0, ...]  # NCKHW

        # sample mixture indicator from softmax
        u = torch.zeros_like(logit_probs).uniform_(1e-5, 1. - 1e-5)  # NCKHW
        # argmax over K, results in NCHW,
        # specifies for each c: which of the K mixtures to take
        sel = torch.argmax(
            logit_probs - torch.log(-torch.log(u)),  # gumbel sampling
            dim=2)
        assert sel.shape == (N, C, H, W), (sel.shape, (N, C, H, W))

        sel = sel.unsqueeze(2)  # NC1HW

        means = torch.gather(l[:, 1, ...], 2, sel).squeeze(2)
        log_scales = torch.clamp(torch.gather(
            l[:, 2, ...], 2, sel).squeeze(2), min=_LOG_SCALES_MIN)

        # sample from the resulting logistic,
        # which now has essentially 1 mixture component only.
        # We use inverse transform sampling.
        # i.e. X~logistic; generate u ~ Unfirom; x = CDF^-1(u),
        #  where CDF^-1 for the logistic is CDF^-1(y) = \mu + \sigma * log(y / (1-y))
        u = torch.zeros_like(means).uniform_(1e-5, 1. - 1e-5)  # NCHW
        x = means + torch.exp(log_scales) * \
            (torch.log(u) - torch.log(1. - u))  # NCHW

        if self.use_coeffs:
            assert C == 3

            def clamp(x_):
                return torch.clamp(x_, 0, 255.)

            # Be careful about coefficients!
            # We need to use the correct selection mask, namely the one for the G and
            #  B channels, as we update the G and B means!
            # Doing torch.gather(l[:, 3, ...], 2, sel) would be completly
            #  wrong.
            coeffs = torch.sigmoid(l[:, 3, ...])
            sel_g, sel_b = sel[:, 1, ...], sel[:, 2, ...]
            coeffs_g_r = torch.gather(coeffs[:, 0, ...], 1, sel_g).squeeze(1)
            coeffs_b_r = torch.gather(coeffs[:, 1, ...], 1, sel_b).squeeze(1)
            coeffs_b_g = torch.gather(coeffs[:, 2, ...], 1, sel_b).squeeze(1)

            # Note: In theory, we should go step by step over the channels
            # and update means with previously sampled
            # xs. But because of the math above (x = means + ...),
            # we can just update the means here and it's all good.
            x0 = clamp(x[:, 0, ...])
            x1 = clamp(x[:, 1, ...] + coeffs_g_r * x0)
            x2 = clamp(x[:, 2, ...] + coeffs_b_r * x0 + coeffs_b_g * x1)
            x = torch.stack((x0, x1, x2), dim=1)
        return x
