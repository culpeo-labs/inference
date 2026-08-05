#pragma once

#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdint.h>

#include <util/types.h>

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

    void matvec(util::mat_t<float, 1> out, util::float_matrix auto W, util::float_vector auto x)
    {
        assert(out.extent(0) == W.extent(0));
        assert(x.extent(0) == W.extent(1));
        assert(W.stride(0) == W.extent(1));
        for (std::size_t r = 0; r < W.extent(0); r++)
        {
            float acc{ 0 };
            for (std::size_t c = 0; c < W.extent(1); c++)
            {
                acc += W[r, c] * x[c];
            }
            out[r] = acc;
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
