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

# TODO some comments needed about [..., -1] == 0

import torch

# torchac can be built with and without CUDA support.
# Here, we try to import both torchac_backend_gpu and torchac_backend_cpu.
# If both fail, an exception is thrown here already.
#
# The right version is then picked in the functions below.
#
# NOTE:
# Without a clean build, multiple versions might be installed. You may use python seutp.py clean --all to prevent this.
# But it should not be an issue.


import_errors = []


try:
    import torchac_backend_gpu
    CUDA_SUPPORTED = True
except ImportError as e:
    CUDA_SUPPORTED = False
    import_errors.append(e)

try:
    import torchac_backend_cpu
    CPU_SUPPORTED = True
except ImportError as e:
    CPU_SUPPORTED = False
    import_errors.append(e)


print(f"Using torchac: {CUDA_SUPPORTED}")


imported_at_least_one = CUDA_SUPPORTED or CPU_SUPPORTED


# if import_errors:
#     import_errors_str = '\n'.join(map(str, import_errors))
#     print(f'*** Import errors:\n{import_errors_str}')


if not imported_at_least_one:
    raise ImportError('*** Failed to import any torchac_backend! Make sure to install torchac with torchac/setup.py. '
                      'See the README for details.')


any_backend = torchac_backend_cpu if CPU_SUPPORTED else torchac_backend_gpu


# print(f'*** torchac: GPU support: {CUDA_SUPPORTED} // CPU support: {CPU_SUPPORTED}')


def _get_gpu_backend():
    if not CUDA_SUPPORTED:
        raise ValueError('Got CUDA tensor, but torchac_backend_gpu is not available. '
                         'Compile torchac with CUDA support, or use CPU mode (see README).')
    return torchac_backend_gpu


def _get_cpu_backend():
    if not CPU_SUPPORTED:
        raise ValueError('Got CPU tensor, but torchac_backend_cpu is not available. '
                         'Compile torchac without CUDA support, or use GPU mode (see README).')
    return torchac_backend_cpu


def encode_cdf(cdf, sym):
    """
    :param cdf: CDF as 1HWLp, as int16, on CPU!
    :param sym: the symbols to encode, as int16, on CPU
    :return: byte-string, encoding `sym`
    """
    if cdf.is_cuda or sym.is_cuda:
        raise ValueError('CDF and symbols must be on CPU for `encode_cdf`')
    # encode_cdf is defined in both backends, so doesn't matter which one we use!
    return any_backend.encode_cdf(cdf, sym)


def decode_cdf(cdf, input_string):
    """
    :param cdf: CDF as 1HWLp, as int16, on CPU
    :param input_string: byte-string, encoding some symbols `sym`.
    :return: decoded `sym`.
    """
    if cdf.is_cuda:
        raise ValueError('CDF must be on CPU for `decode_cdf`')
    # encode_cdf is defined in both backends, so doesn't matter which one we use!
    return any_backend.decode_cdf(cdf, input_string)


def encode_logistic_mixture(
        targets, means, log_scales, logit_probs_softmax,  # CDF
        sym):
    """
    NOTE: This function uses either the CUDA or CPU backend, depending on the device of the input tensors.
    NOTE: targets, means, log_scales, logit_probs_softmax must all be on the same device (CPU or GPU)
    In the following, we use
        Lp: Lp = L+1, where L = number of symbols.
        K: number of mixtures
    :param targets: values of symbols, tensor of length Lp, float32
    :param means: means of mixtures, tensor of shape 1KHW, float32
    :param log_scales: log(scales) of mixtures, tensor of shape 1KHW, float32
    :param logit_probs_softmax: weights of the mixtures (PI), tensorf of shape 1KHW, float32
    :param sym: the symbols to encode. MUST be on CPU!!
    :return: byte-string, encoding `sym`.
    """
    if not (targets.is_cuda == means.is_cuda == log_scales.is_cuda == logit_probs_softmax.is_cuda):
        raise ValueError('targets, means, log_scales, logit_probs_softmax must all be on the same device! Got '
                         f'{targets.device}, {means.device}, {log_scales.device}, {logit_probs_softmax.device}.')
    if sym.is_cuda:
        raise ValueError('sym must be on CPU!')

    if targets.is_cuda:
        return _get_gpu_backend().encode_logistic_mixture(
            targets, means, log_scales, logit_probs_softmax, sym)
    else:
        cdf = _get_uint16_cdf(logit_probs_softmax, targets, means, log_scales)
        return encode_cdf(cdf, sym)


def decode_logistic_mixture(
        targets, means, log_scales, logit_probs_softmax,  # CDF
        input_string):
    """
    NOTE: This function uses either the CUDA or CPU backend, depending on the device of the input tensors.
    NOTE: targets, means, log_scales, logit_probs_softmax must all be on the same device (CPU or GPU)
    In the following, we use
        Lp: Lp = L+1, where L = number of symbols.
        K: number of mixtures
    :param targets: values of symbols, tensor of length Lp, float32
    :param means: means of mixtures, tensor of shape 1KHW, float32
    :param log_scales: log(scales) of mixtures, tensor of shape 1KHW, float32
    :param logit_probs_softmax: weights of the mixtures (PI), tensorf of shape 1KHW, float32
    :param input_string: byte-string, encoding some symbols `sym`.
    :return: decoded `sym`.
    """
    if not (targets.is_cuda == means.is_cuda == log_scales.is_cuda == logit_probs_softmax.is_cuda):
        raise ValueError('targets, means, log_scales, logit_probs_softmax must all be on the same device! Got '
                         f'{targets.device}, {means.device}, {log_scales.device}, {logit_probs_softmax.device}.')

    if targets.is_cuda:
        return _get_gpu_backend().decode_logistic_mixture(
            targets, means, log_scales, logit_probs_softmax, input_string)
    else:
        cdf = _get_uint16_cdf(logit_probs_softmax, targets, means, log_scales)
        return decode_cdf(cdf, input_string)


# ------------------------------------------------------------------------------

# The following code is invoced for when the CDF is not on GPU, and we cannot use torchac/torchac_kernel.cu
# This basically replicates that kernel in pure PyTorch.

def _get_uint16_cdf(logit_probs_softmax, targets, means, log_scales):
    cdf_float = _get_C_cur_weighted(
        logit_probs_softmax, targets, means, log_scales)
    cdf = _renorm_cast_cdf_(cdf_float, precision=16)
    cdf = cdf.cpu()
    return cdf


def _get_C_cur_weighted(logit_probs_softmax_c, targets, means_c, log_scales_c):
    C_cur = _get_C_cur(targets, means_c, log_scales_c)  # NKHWL
    C_cur = C_cur.mul(logit_probs_softmax_c.unsqueeze(-1)).sum(1)  # NHWL
    return C_cur


def _get_C_cur(targets, means_c, log_scales_c):  # NKHWL
    """
    :param targets: Lp floats
    :param means_c: NKHW
    :param log_scales_c: NKHW
    :return:
    """
    # NKHW1
    inv_stdv = torch.exp(-log_scales_c).unsqueeze(-1)
    # NKHWL'
    centered_targets = (targets - means_c.unsqueeze(-1))
    # NKHWL'
    cdf = centered_targets.mul(inv_stdv).sigmoid()  # sigma' * (x - mu)
    return cdf


def _renorm_cast_cdf_(cdf, precision):
    Lp = cdf.shape[-1]
    finals = 1  # NHW1
    # RENORMALIZATION_FACTOR in cuda
    f = torch.tensor(2, dtype=torch.float32, device=cdf.device).pow_(precision)
    cdf = cdf.mul((f - (Lp - 1)) / finals)  # TODO
    cdf = cdf.round()
    cdf = cdf.to(dtype=torch.int16, non_blocking=True)
    r = torch.arange(Lp, dtype=torch.int16, device=cdf.device)
    cdf.add_(r)
    return cdf
