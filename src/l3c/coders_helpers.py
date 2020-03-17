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

Very thin wrapper around DiscretizedMixLogisticLoss.cdf_step_non_shared that keeps track of targets, which are the
same for all channels of the bottleneck, as well as the current channel index.

"""

import torch

from src.l3c.logistic_mixture import CDFOut, DiscretizedMixLogisticLoss


class CodingCDFNonshared(object):
    def __init__(self, l, total_C, dmll: DiscretizedMixLogisticLoss):
        """
        :param l: predicted distribution, i.e., NKpHW, see DiscretizedMixLogisticLoss
        :param total_C:
        :param dmll:
        """
        self.l = l
        self.dmll = dmll

        # Lp = L+1
        self.targets = torch.linspace(dmll.x_min - dmll.bin_width / 2,
                                      dmll.x_max + dmll.bin_width / 2,
                                      dmll.L + 1, dtype=torch.float32, device=l.device)
        self.total_C = total_C
        self.c_cur = 0

    def get_next_C(self, decoded_x) -> CDFOut:
        """
        Get CDF to encode/decode next channel
        :param decoded_x: NCHW
        :return: C_cond_cur, NHWL'
        """
        C_Cur = self.dmll.cdf_step_non_shared(
            self.l, self.targets, self.c_cur, self.total_C, decoded_x)
        self.c_cur += 1
        return C_Cur
