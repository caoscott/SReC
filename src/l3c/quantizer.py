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

Based on our TensorFlow implementation for the CVPR 2018 paper

"Conditional Probability Models for Deep Image Compression"

This is a PyTorch implementation of that quantization layer.

https://github.com/fab-jul/imgcomp-cvpr/blob/master/code/quantizer.py

"""


def to_sym(x, x_min, x_max, L):
    sym_range = x_max - x_min
    bin_size = sym_range / (L-1)
    return x.clamp(x_min, x_max).sub(x_min).div(bin_size).round()


def to_bn(S, x_min, x_max, L):
    sym_range = x_max - x_min
    bin_size = sym_range / (L-1)
    return S.float().mul(bin_size).add(x_min)
