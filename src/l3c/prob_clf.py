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
from typing import List, Union

import torch
from torch import nn

from src import util
from src.l3c.logistic_mixture import non_shared_get_Kp


class AtrousProbabilityClassifier(nn.Module):
    def __init__(self,
                 in_ch: int,
                 C: int,
                 num_params: int,
                 K: int = 10,
                 kernel_size: int = 3,
                 atrous_rates_str: str = '1,2,4') -> None:
        super(AtrousProbabilityClassifier, self).__init__()

        Kp = non_shared_get_Kp(K, C, num_params)

        self.atrous = StackedAtrousConvs(atrous_rates_str, in_ch, Kp,
                                         kernel_size=kernel_size)
        self._repr = f'C={C}; K={K}; Kp={Kp}; rates={atrous_rates_str}'

    def __repr__(self) -> str:
        return f'AtrousProbabilityClassifier({self._repr})'

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        """
        :param x: N C H W
        :return: N Kp H W
        """
        return self.atrous(x)


class StackedAtrousConvs(nn.Module):
    def __init__(self,
                 atrous_rates_str: Union[str, int],
                 Cin: int,
                 Cout: int,
                 bias: bool = True,
                 kernel_size: int = 3) -> None:
        super(StackedAtrousConvs, self).__init__()
        atrous_rates = self._parse_atrous_rates_str(atrous_rates_str)
        self.atrous = nn.ModuleList(
            [util.conv(Cin, Cin, kernel_size, rate=rate)
             for rate in atrous_rates])
        self.lin = util.conv(len(atrous_rates) * Cin, Cout, 1, bias=bias)
        self._extra_repr = 'rates={}'.format(atrous_rates)

    @staticmethod
    def _parse_atrous_rates_str(atrous_rates_str: Union[str, int]) -> List[int]:
        # expected to either be an int or a comma-separated string 1,2,4
        if isinstance(atrous_rates_str, int):
            return [atrous_rates_str]
        else:
            return list(map(int, atrous_rates_str.split(',')))

    def extra_repr(self) -> str:
        return self._extra_repr

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = torch.cat([atrous(x)
                       for atrous in self.atrous], dim=1)  # type: ignore
        x = self.lin(x)
        return x
