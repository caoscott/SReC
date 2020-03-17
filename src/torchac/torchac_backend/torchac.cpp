/**
 * COPYRIGHT 2019 ETH Zurich
 *
 * LINUX:

 - GPU: nvcc 9 and gcc 5.5 works
 - without: ???

 *
 * MACOS:
 *
 * CC=clang++ -std=libc++
 * MACOSX_DEPLOYMENT_TARGET=10.14
 *
 * BASED on
 *
 * https://marknelson.us/posts/2014/10/19/data-compression-with-arithmetic-coding.html
 */

#include <torch/extension.h>

#include <iostream>
#include <vector>
#include <tuple>
#include <fstream>
#include <algorithm>
#include <string>
#include <chrono>
#include <numeric>
#include <iterator>

#include <bitset>


using cdf_t = uint16_t;

/** Encapsulates a pointer to a CDF tensor */
struct cdf_ptr {
    const cdf_t* data;  // expected to be a N_sym x Lp matrix, stored in row major.
    const int N_sym;  // Number of symbols stored by `data`.
    const int Lp;  // == L+1, where L is the number of possible values a symbol can take.
    cdf_ptr(const cdf_t* data,
            const int N_sym,
            const int Lp) : data(data), N_sym(N_sym), Lp(Lp) {};
};

/// This is set by setup.py if CUDA support is desired
#ifdef COMPILE_CUDA
/// All these are defined in torchac_kernel.cu
cdf_t* malloc_cdf(const int N, const int Lp);
void free_cdf(cdf_t* cdf_mem);
void calculate_cdf(
        const at::Tensor& targets,
        const at::Tensor& means,
        const at::Tensor& log_scales,
        const at::Tensor& logit_probs_softmax,
        cdf_t* cdf_mem,
        const int K, const int Lp, const int N_cdf);

#endif  // COMPILE_CUDA

/** Class to save output bit by bit to a byte string */
class OutCacheString {
private:
public:
    std::string out="";
    uint8_t cache=0;
    uint8_t count=0;
    void append(const int bit) {
        cache <<= 1;
        cache |= bit;
        count += 1;
        if (count == 8) {
            out.append(reinterpret_cast<const char *>(&cache), 1);
            count = 0;
        }
    }
    void flush() {
        if (count > 0) {
            for (int i = count; i < 8; ++i) {
                append(0);
            }
            assert(count==0);
        }
    }
    void append_bit_and_pending(const int bit, uint64_t &pending_bits) {
        append(bit);
        while (pending_bits > 0) {
            append(!bit);
            pending_bits -= 1;
        }
    }
};

/** Class to read byte string bit by bit */
class InCacheString {
private:
    const std::string& in_;

public:
    explicit InCacheString(const std::string& in) : in_(in) {};

    uint8_t cache=0;
    uint8_t cached_bits=0;  // num
    size_t in_ptr=0;

    void get(uint32_t& value) {
        if (cached_bits == 0) {
            if (in_ptr == in_.size()){
                value <<= 1;
                return;
            }
            /// Read 1 byte
            cache = (uint8_t) in_[in_ptr];
            in_ptr++;
            cached_bits = 8;
        }
        value <<= 1;
        value |= (cache >> (cached_bits - 1)) & 1;
        cached_bits--;
    }

    void initialize(uint32_t& value) {
        for (int i = 0; i < 32; ++i) {
            get(value);
        }
    }
};

/** Get an instance of the `cdf_ptr` struct. */
const struct cdf_ptr get_cdf_ptr(const at::Tensor& cdf)
{
    AT_CHECK(!cdf.is_cuda(), "cdf must be on CPU!")
    const auto s = cdf.sizes();
    AT_CHECK(s.size() == 4 && s[0] == 1, "Invalid size for cdf! Expected 1HWLp")

    const int N_sym = s[1] * s[2];
    const int Lp = s[3];
    const auto cdf_reshaped = at::reshape(cdf, {N_sym, -1});
    const auto cdf_acc = cdf_reshaped.accessor<int16_t, 2>();
    const cdf_t* cdf_ptr = (uint16_t*)cdf_acc.data();

    const struct cdf_ptr res(cdf_ptr, N_sym, Lp);
    return res;
}


// -----------------------------------------------------------------------------


/** Encode symbols `sym` with CDF represented by `cdf_ptr`. NOTE: this is not exposted to python. */
py::bytes encode(
        const cdf_ptr& cdf_ptr,
        const at::Tensor& sym){

#ifdef VERBOSE
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
#endif
    OutCacheString out_cache;

    uint32_t low = 0;
    uint32_t high = 0xFFFFFFFFU;
    uint64_t pending_bits = 0;

    const int precision = 16;

    const cdf_t* cdf = cdf_ptr.data;
    const int N_sym = cdf_ptr.N_sym;
    const int Lp = cdf_ptr.Lp;
    const int max_symbol = Lp - 2;

    auto sym_ = sym.accessor<int16_t, 1>();

    for (int i = 0; i < N_sym; ++i) {
        const int16_t sym_i = sym_[i];

        const uint64_t span = static_cast<uint64_t>(high) - static_cast<uint64_t>(low) + 1;

        const int offset = i * Lp;
        const uint32_t c_low = cdf[offset + sym_i];
        const uint32_t c_high = sym_i == max_symbol ? 0x10000U : cdf[offset + sym_i + 1];

        high = (low - 1) + ((span * static_cast<uint64_t>(c_high)) >> precision);
        low =  (low)     + ((span * static_cast<uint64_t>(c_low))  >> precision);

        while (true) {
            if (high < 0x80000000U) {
                out_cache.append_bit_and_pending(0, pending_bits);
                low <<= 1;
                high <<= 1;
                high |= 1;
            } else if (low >= 0x80000000U) {
                out_cache.append_bit_and_pending(1, pending_bits);
                low <<= 1;
                high <<= 1;
                high |= 1;
            } else if (low >= 0x40000000U && high < 0xC0000000U) {
                pending_bits++;
                low <<= 1;
                low &= 0x7FFFFFFF;
                high <<= 1;
                high |= 0x80000001;
            } else {
                break;
            }
        }
    }

    pending_bits += 1;

    if (pending_bits) {
        if (low < 0x40000000U) {
            out_cache.append_bit_and_pending(0, pending_bits);
        } else {
            out_cache.append_bit_and_pending(1, pending_bits);
        }
    }

    out_cache.flush();

#ifdef VERBOSE
    std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
    std::cout << "Time difference (sec) = " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 <<std::endl;
#endif

    return py::bytes(out_cache.out);
}

#ifdef COMPILE_CUDA
/** See torchac.py */
py::bytes encode_logistic_mixture(
        /// --- CDF
        const at::Tensor &targets,
        const at::Tensor &means,
        const at::Tensor &log_scales,
        const at::Tensor &logit_probs_softmax,
        /// ---
        const at::Tensor& sym)
{
    const int Lp = targets.sizes()[0];
    const auto s = means.sizes();
    AT_CHECK(s.size() == 4 && s[0] == 1, "Invalid size for means! Expected 1KHW")
    AT_CHECK(means.sizes() == log_scales.sizes(), "Invalid size for log_scales! Expected 1KHW")
    AT_CHECK(means.sizes() == logit_probs_softmax.sizes(), "Invalid size for logit_probs_softmax! Expected 1KHW")
    AT_CHECK(means.is_cuda() && log_scales.is_cuda() && logit_probs_softmax.is_cuda(), "Tensors must be on CUDA!")
    const int N_sym = s[2] * s[3];
    const int K = s[1];

    cdf_t* cdf_mem = malloc_cdf(N_sym, Lp);

    calculate_cdf(targets, means, log_scales, logit_probs_softmax, cdf_mem, K, Lp, N_sym);

    const cdf_ptr cdf_ptr(cdf_mem, N_sym, Lp);
    const auto res = encode(cdf_ptr, sym);
    free_cdf(cdf_mem);

    return res;
}
#endif


/** See torchac.py */
py::bytes encode_cdf(
        const at::Tensor& cdf, /* NHWLp, must be on CPU! */
        const at::Tensor& sym)
{
    const auto cdf_ptr = get_cdf_ptr(cdf);
    return encode(cdf_ptr, sym);
}



//------------------------------------------------------------------------------


cdf_t binsearch(const cdf_t* cdf, cdf_t target, cdf_t max_sym,
                   const int offset)  /* i * Lp */
{
    cdf_t left = 0;
    cdf_t right = max_sym + 1;  // len(cdf) == max_sym + 2

    while (left + 1 < right) {  // ?
        // left and right will be < 0x10000 in practice, so left+right fits in uint16_t...
        const auto m = static_cast<const cdf_t>((left + right) / 2);
        const auto v = cdf[offset + m];
        if (v < target) {
            left = m;
        } else if (v > target) {
            right = m;
        } else {
            return m;
        }
    }

    return left;
}


at::Tensor decode(
        const cdf_ptr& cdf_ptr,
        const std::string& in) {

#ifdef VERBOSE
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
#endif

    const cdf_t* cdf = cdf_ptr.data;
    const int N_sym = cdf_ptr.N_sym;  // To know the # of syms to decode. is encoded in file!
    const int Lp = cdf_ptr.Lp;  // To calculate offset
    const int max_symbol = Lp - 2;

    // 16 bit!
    auto out = torch::empty({N_sym}, at::kShort);
    auto out_ = out.accessor<int16_t, 1>();

    uint32_t low = 0;
    uint32_t high = 0xFFFFFFFFU;
    uint32_t value = 0;
    const uint32_t c_count = 0x10000U;
    const int precision = 16;  // TODO: unify with torchac_kernel.cu

    InCacheString in_cache(in);
    in_cache.initialize(value);

    for (int i = 0; i < N_sym; ++i) {
        const uint64_t span = static_cast<uint64_t>(high) - static_cast<uint64_t>(low) + 1;
        // always < 0x10000 ???
        const uint16_t count = ((static_cast<uint64_t>(value) - static_cast<uint64_t>(low) + 1) * c_count - 1) / span;

        const int offset = i * Lp;
        auto sym_i = binsearch(cdf, count, (cdf_t)max_symbol, offset);

        out_[i] = (int16_t)sym_i;

        if (i == N_sym-1) {
            break;
        }

        const uint32_t c_low = cdf[offset + sym_i];
        const uint32_t c_high = sym_i == max_symbol ? 0x10000U : cdf[offset + sym_i + 1];

        high = (low - 1) + ((span * static_cast<uint64_t>(c_high)) >> precision);
        low =  (low)     + ((span * static_cast<uint64_t>(c_low))  >> precision);

        while (true) {
            if (low >= 0x80000000U || high < 0x80000000U) {
                low <<= 1;
                high <<= 1;
                high |= 1;
                in_cache.get(value);
            } else if (low >= 0x40000000U && high < 0xC0000000U) {
                /**
                 * 0100 0000 ... <= value <  1100 0000 ...
                 * <=>
                 * 0100 0000 ... <= value <= 1011 1111 ...
                 * <=>
                 * value starts with 01 or 10.
                 * 01 - 01 == 00  |  10 - 01 == 01
                 * i.e., with shifts
                 * 01A -> 0A  or  10A -> 1A, i.e., discard 2SB as it's all the same while we are in
                 *    near convergence
                 */
                low <<= 1;
                low &= 0x7FFFFFFFU;  // make MSB 0
                high <<= 1;
                high |= 0x80000001U;  // add 1 at the end, retain MSB = 1
                value -= 0x40000000U;
                in_cache.get(value);
            } else {
                break;
            }
        }
    }

#ifdef VERBOSE
    std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
    std::cout << "Time difference (sec) = " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 <<std::endl;
#endif

    return out;
}

#ifdef COMPILE_CUDA
/** See torchac.py */
at::Tensor decode_logistic_mixture(
        const at::Tensor &targets,
        const at::Tensor &means,
        const at::Tensor &log_scales,
        const at::Tensor &logit_probs_softmax,
        const std::string& in)
{
    const int Lp = targets.sizes()[0];
    const auto s = means.sizes();
    AT_CHECK(s.size() == 4 && s[0] == 1, "Invalid size for means! Expected 1KHW")
    AT_CHECK(means.sizes() == log_scales.sizes(), "Invalid size for log_scales! Expected 1KHW")
    AT_CHECK(means.sizes() == logit_probs_softmax.sizes(), "Invalid size for logit_probs_softmax! Expected 1KHW")
    AT_CHECK(means.is_cuda() && log_scales.is_cuda() && logit_probs_softmax.is_cuda(), "Tensors must be on CUDA!")
    const int N_sym = s[2] * s[3];
    const int K = s[1];

    cdf_t* cdf_mem = malloc_cdf(N_sym, Lp);
#ifdef VERBOSE
    std::cout << "Calculating CDF " << N_sym << "x" << Lp << "... ";
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
#endif

    calculate_cdf(targets, means, log_scales, logit_probs_softmax, cdf_mem, K, Lp, N_sym);
#ifdef VERBOSE
    std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
    std::cout << "CDF [sec]: " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 <<std::endl;
#endif

    const cdf_ptr cdf_ptr(cdf_mem, N_sym, Lp);

    const auto res = decode(cdf_ptr, in);

    free_cdf(cdf_mem);
    return res;
}
#endif


/** See torchac.py */
at::Tensor decode_cdf(
        const at::Tensor& cdf, /* NHWLp */
        const std::string& in)
{
    const auto cdf_ptr = get_cdf_ptr(cdf);
    return decode(cdf_ptr, in);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("encode_cdf", &encode_cdf, "Encode from CDF");
    m.def("decode_cdf", &decode_cdf, "Decode from CDF");
#ifdef COMPILE_CUDA
    m.def("encode_logistic_mixture", &encode_logistic_mixture, "Encode logistic");
    m.def("decode_logistic_mixture", &decode_logistic_mixture, "Decode logistic");
    m.def("cuda_supported", []() { return true; });
#else
    m.def("cuda_supported", []() { return false; });
#endif
}
