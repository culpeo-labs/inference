#pragma once

#include <cassert>
#include <cmath>
#include <cstring>
#include <cstddef>
#include <limits>
#include <pmmintrin.h>
#include <ranges>
#include <stdint.h>
#include <source_location>
#include <immintrin.h>

#include <type_traits>
#include <util/timers.h>
#include <util/types.h>
#include <xmmintrin.h>

namespace culpeo::inference::operations {
    void embed(util::mat_t<float, 1> out, util::float_matrix auto table, std::ptrdiff_t token_id)
    {
        assert(token_id >= 0 && static_cast<std::size_t>(token_id) < table.extent(0));
        assert(out.extent(0) == table.extent(1));
        for(std::size_t i{ 0 }; i < out.extent(0); ++i) out[i] = table[token_id, i];
    }

    void rmsnorm(util::mat_t<float, 1> out, util::float_vector auto x, util::float_vector auto weight, float eps)
    {
        assert(out.extent(0) == x.extent(0));
        assert(out.extent(0) == weight.extent(0));
        double sigma{ 0 };
        for (auto i{ 0ull }; i < out.extent(0); i++)
        {
            sigma += x[i] * x[i];
        }
        double rms = std::sqrt(sigma / static_cast<double>(x.extent(0)) + eps);
        for (auto i{ 0ull}; i < out.extent(0); i++)
        {
            out[i] = x[i] / rms * weight[i];
        }
    }

    inline float horizontal(const __m256 val)
    {
        /* lower: val[0:127] (f1, f2, f3, f4), cast (free) */
        auto lower = _mm256_castps256_ps128(val);
        /* upper: val[128, 255] (f5, f6, f7, f8), extract (instruction) */
        auto upper = _mm256_extractf128_ps(val, 1);
        /* lower: (f1 + f5, f2 + f6, f3 + f7, f4 + f8)*/
        lower = _mm_add_ps(lower, upper);
        /* hadd(a, b) -> (a1 + a2, a3 + a4, b1 + b2, b3 + b4) */
        /* lower: (l1 + l2, l3 + l4, _, _)*/
        lower = _mm_hadd_ps(lower, lower);
        /* lower: (l1 + l2 + l3 + l3, _, _, _) */
        lower = _mm_hadd_ps(lower, lower);
        return _mm_cvtss_f32(lower);
    }

    template<typename T>
    inline __m256 load(const T * ptr)
    {
        if constexpr (sizeof(T) == 2)
        {
            /* T == int16_t (bfloat16) */
            __m128i i128 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(ptr));
            auto i256 = _mm256_cvtepu16_epi32(i128);
            /* shift, bf16 -> f32 */
            i256 = _mm256_slli_epi32(i256, 16);
            return _mm256_castsi256_ps(i256);
        }
        else
        {
            /* T == float */
            return _mm256_loadu_ps(ptr);
        }
    }

    void matvec(util::mat_t<float, 1> out, util::float_matrix auto W, util::float_vector auto x)
    {
        assert(out.extent(0) == W.extent(0));
        assert(x.extent(0) == W.extent(1));
        assert(W.stride(0) == W.extent(1));
        for (std::size_t r = 0; r < W.extent(0); r++)
        {
            auto acc0 = _mm256_setzero_ps(),
                 acc1 = _mm256_setzero_ps(),
                 acc2 = _mm256_setzero_ps(),
                 acc3 = _mm256_setzero_ps();

            for (auto c : std::ranges::views::iota(std::size_t{0}, W.extent(1) - W.extent(1) % 32) | std::ranges::views::stride(32))
            {
                std::ptrdiff_t offset = r * W.extent(1) + c;
                acc0 = _mm256_fmadd_ps(load(W.data_handle() + offset), load(x.data_handle() + c), acc0);
                acc1 = _mm256_fmadd_ps(load(W.data_handle() + offset + 8), load(x.data_handle() + c + 8), acc1);
                acc2 = _mm256_fmadd_ps(load(W.data_handle() + offset + 16), load(x.data_handle() + c + 16), acc2);
                acc3 = _mm256_fmadd_ps(load(W.data_handle() + offset + 24), load(x.data_handle() + c + 24), acc3);
            }
            for (auto c : std::ranges::views::iota(W.extent(1) - W.extent(1) % 32, W.extent(1) - W.extent(1) % 8) | std::ranges::views::stride(8))
            {
                std::ptrdiff_t offset = r * W.extent(1) + c;
                acc0 = _mm256_fmadd_ps(load(W.data_handle() + offset), load(x.data_handle() +c), acc0);
            }
            acc0 = _mm256_add_ps(acc0, acc1);
            acc2 = _mm256_add_ps(acc2, acc3);
            acc0 = _mm256_add_ps(acc0, acc2);
            float tail{ 0 };
            for (std::size_t c = W.extent(1) - W.extent(1) % 8; c < W.extent(1); c++)
            {
                tail += W[r, c] * x[c];
            }
            /* Horizontal sum */
            out[r] = tail + horizontal(acc0);
        }
    }

    void rope(util::mat_t<float, 2> t, std::ptrdiff_t pos, float theta_base);

    void softmax(util::mat_t<float, 1> x);

    void add(util::mat_t<float, 1> out, util::float_vector auto a, util::float_vector auto b)
    {
        assert(out.extent(0) == a.extent(0));
        assert(out.extent(0) == b.extent(0));
        for (auto i{ 0ull }; i < out.extent(0); i++)
        {
            out[i] = a[i] + b[i];
        }
    }

    void silu_mul(util::mat_t<float, 1> out, util::float_vector auto gate, util::float_vector auto up)
    {
        assert(out.extent(0) == gate.extent(0));
        assert(out.extent(0) == up.extent(0));
        for (auto i{ 0ull }; i < out.extent(0); i++)
        {
            auto gate_i = gate[i];
            auto silu = gate_i / (1 + std::exp(-gate_i));
            out[i] = silu * up[i];
        }
    }

    std::size_t argmax(util::float_vector auto x)
    {
        float max{ std::numeric_limits<float>::min() };
        std::size_t index{ 0 };
        for (auto i{ 0ull }; i < x.extent(0); i++)
        {
            if (max < x[i])
            {
                max = x[i];
                index = i;
            }

        }
        return index;
    }
}
