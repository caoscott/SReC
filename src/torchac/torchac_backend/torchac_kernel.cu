/*
 * COPYRIGHT 2019 ETH Zurich
 */
#include <ATen/ATen.h>

#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>


using cdf_t = uint16_t;
const int PRECISION = 16;
const int RENORMALIZATION_FACTOR = 2 << (PRECISION - 1);

namespace {
    __device__ __forceinline__ float sigmoidf (float a) {
        return 1.0 / (1.0 + expf (-a));
    }

    __device__ __forceinline__ cdf_t renorm(float cdf, const int Lp, const int l) {
        cdf *= (RENORMALIZATION_FACTOR - (Lp - 1));
        cdf_t cdf_int = static_cast<cdf_t>(lrintf(cdf) + l);
        return cdf_int;
    }

    __global__ void calculate_cdf_kernel(
            const int N, const int Lp, const int K,
            const float* __restrict__ targets,  // Lp length vector
            const float* __restrict__ means,
            const float* __restrict__ log_scales,
            const float* __restrict__ logit_probs_softmax,
            cdf_t* __restrict__ cdf_mem /* out */) {
        /**
         * Expects to be launched on a N*Lp grid? TODO
         *
         * means, log_scales, logit_probs_softmax:
         *      each is a 1KHW matrix reshaped to KN, where N = H*W
         * cdf_mem:
         *      an array of length N * Lp, representing a NxLp matrix, where
         *      cdf[n][l] = cdf_mem[n*Lp + l]
         *
         * Code:
         *      for n, l in range(N) x range(Lp)
         *          target = l
         *          cdf_n_l = 0;
         *          for k in range(K)
         *              log_scale = log_scales[k][n]
         *              mean = means[k][n]
         *              logit_prob = logit_probs_softmax[k][n]
         *              inv_stdv = exp(log_scale)
         *              centered_target = target - mean
         *              cdf_n_l += logit_prob * sigmoid(centered_target * inv_stdv)
         *           cdf[n][l] = cdf_mem
         */
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        for (int i_1d = index; i_1d < N * Lp; i_1d += stride) {
            const int n = i_1d / Lp;
            const int l = i_1d % Lp;

            const float target = targets[l];
            float cdf_n_l_float = 0;  // initialize

            for (int k = 0; k < K; ++k) {
                const float log_scale = log_scales[k * N + n];
                const float mean = means[k * N + n];
                const float logit_prob = logit_probs_softmax[k * N + n];
                const float inv_stdv = expf(-log_scale);
                const float centered_target = target - mean;
                cdf_n_l_float += logit_prob * sigmoidf(centered_target * inv_stdv);
            }

            const int cdf_n_l_idx = i_1d;
            cdf_mem[cdf_n_l_idx] = renorm(cdf_n_l_float, Lp, l);
        }
    }
}


cdf_t* malloc_cdf(const int N, const int Lp) {
    cdf_t* cdf_mem;
    cudaMallocManaged(&cdf_mem, N*Lp*sizeof(cdf_t));
    return cdf_mem;
}


void free_cdf(cdf_t* cdf_mem) {
    cudaFree(cdf_mem);
}


template <typename T>
std::string to_string(const T& object) {
    std::ostringstream ss;
    ss << object;
    return ss.str();
}

#define CHECK_1KHW(K, x) AT_CHECK(x.sizes().size() == 4 && x.sizes()[0] == 1 && x.sizes()[1] == K,  \
        "#x must be 4D, got %s", to_string(x.sizes()))

#define CHECK_CONTIGUOUS_AND_CUDA(x) AT_CHECK(x.is_contiguous() && x.is_cuda(), \
        "#x must be contiguous and on GPU, got %d and %d", x.is_contiguous(), x.is_cuda())

void calculate_cdf(
        const at::Tensor& targets,
        const at::Tensor& means,
        const at::Tensor& log_scales,
        const at::Tensor& logit_probs_softmax,
        cdf_t * cdf_mem,
        const int K, const int Lp, const int N_cdf) {

    CHECK_1KHW(K, means);
    CHECK_1KHW(K, log_scales);
    CHECK_1KHW(K, logit_probs_softmax);

    CHECK_CONTIGUOUS_AND_CUDA(targets);
    CHECK_CONTIGUOUS_AND_CUDA(means);
    CHECK_CONTIGUOUS_AND_CUDA(log_scales);
    CHECK_CONTIGUOUS_AND_CUDA(logit_probs_softmax);

    AT_CHECK(means.sizes() == log_scales.sizes() &&
             log_scales.sizes() == logit_probs_softmax.sizes())

    const auto param_sizes = means.sizes();
    const auto N = param_sizes[2] * param_sizes[3];  // H * W
    AT_CHECK(N == N_cdf, "%d != %d", N, N_cdf);

    const int blockSize = 1024;
    const int numBlocks = (N * Lp + blockSize - 1) / blockSize;

    calculate_cdf_kernel<<<numBlocks, blockSize>>>(
                    N, Lp, K,
                    targets.data<float>(),
                    means.data<float>(),
                    log_scales.data<float>(),
                    logit_probs_softmax.data<float>(),
                    cdf_mem);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
}



