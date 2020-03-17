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

NOTE: Needs PyTorch 1.0 or newer, as the C++ code relies on that API!

Depending on the environment variable COMPILE_CUDA, compiles the torchac_backend with or
without support for CUDA, into a module called torchac_backend_gpu or torchac_backend_cpu.

COMPILE_CUDA = auto is equal to yes if one of the supported combinations of nvcc and gcc is found (see
_supported_compilers_available).
COMPILE_CUDA = force means compile with CUDA, even if it is not one of the supported combinations
COMPILE_CUDA = no means no CUDA.

The only difference between the CUDA and non-CUDA versions is: With CUDA, _get_uint16_cdf from torchac is done with a
simple/non-optimized CUDA kernel (torchac_kernel.cu), which has one benefit: we can directly write into shared memory!
This saves an expensive copying step from GPU to CPU.

Flags read by this script:
    COMPILE_CUDA=[auto|force|no]
    
"""

import sys
import re
import subprocess
from setuptools import setup
from distutils.version import LooseVersion
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension
import os


MODULE_BASE_NAME = 'torchac_backend'


def prefixed(prefix, l):
    ps = [os.path.join(prefix, el) for el in l]
    for p in ps:
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
    return ps


def compile_ext(cuda_support):
    print('Compiling, cuda_support={}'.format(cuda_support))
    ext_module = get_extension(cuda_support)

    setup(name=ext_module.name,
          version='1.0.0',
          ext_modules=[ext_module],
          extra_compile_args=['-mmacosx-version-min=10.9'],
          cmdclass={'build_ext': BuildExtension})


def get_extension(cuda_support):
    # dir of this file
    setup_dir = os.path.dirname(os.path.realpath(__file__))
    # Where the cpp and cu files are
    prefix = os.path.join(setup_dir, MODULE_BASE_NAME)
    if not os.path.isdir(prefix):
        raise ValueError('Did not find backend foler: {}'.format(prefix))
    if cuda_support:
        nvcc_avaible, nvcc_version = supported_nvcc_available()
        if not nvcc_avaible:
            print(_bold_warn_str('***WARN') + ': Found untested nvcc {}'.format(nvcc_version))

        return CUDAExtension(
                MODULE_BASE_NAME + '_gpu',
                prefixed(prefix, ['torchac.cpp', 'torchac_kernel.cu']),
                define_macros=[('COMPILE_CUDA', '1')])
    else:
        return CppExtension(
                MODULE_BASE_NAME + '_cpu',
                prefixed(prefix, ['torchac.cpp']))


# TODO:
# Add further supported version as specified in readme



def _supported_compilers_available():
    """
    To see an up-to-date list of tested combinations of GCC and NVCC, see the README
    """
    return _supported_gcc_available()[0] and supported_nvcc_available()[0]


def _supported_gcc_available():
    v = _get_version(['gcc', '-v'], r'version (.*?)\s+')
    return LooseVersion('6.0') > LooseVersion(v) >= LooseVersion('5.0'), v


def supported_nvcc_available():
    v = _get_version(['nvcc', '-V'], 'release (.*?),')
    if v is None:
        return False, 'nvcc unavailable!'
    return LooseVersion(v) >= LooseVersion('9.0'), v


def _get_version(cmd, regex):
    try:
        otp = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode()
        if len(otp.strip()) == 0:
            raise ValueError('No output')
        m = re.search(regex, otp)
        if not m:
            raise ValueError('Regex does not match output:\n{}'.format(otp))
        return m.group(1)
    except FileNotFoundError:
        return None


def _bold_warn_str(s):
    return '\x1b[91m\x1b[1m' + s + '\x1b[0m'


def _assert_torch_version_sufficient():
    import torch
    if LooseVersion(torch.__version__) >= LooseVersion('1.0'):
        return
    print(_bold_warn_str('Error:'), 'Need PyTorch version >= 1.0, found {}'.format(torch.__version__))
    sys.exit(1)


def main():
    _assert_torch_version_sufficient()

    cuda_flag = os.environ.get('COMPILE_CUDA', 'no')

    if cuda_flag == 'auto':
        cuda_support = _supported_compilers_available()
        print('Found CUDA supported:', cuda_support)
    elif cuda_flag == 'force':
        cuda_support = True
    elif cuda_flag == 'no':
        cuda_support = False
    else:
        raise ValueError('COMPILE_CUDA must be in (auto, force, no), got {}'.format(cuda_flag))

    compile_ext(cuda_support)


if __name__ == '__main__':
    main()
