#pragma once

#include <cstddef>
#include <stdint.h>
#include <mdspan>

namespace culpeo::inference {

    template<typename T>
    using matrix_t = std::mdspan<T, std::dextents<std::size_t, 2>>;

    template<typename T>
    using vector_t = std::span<T>;

    void embed(vector_t<float> out, matrix_t<const float> table, std::ptrdiff_t token_id);

    void rmsnorm(vector_t<float> out, vector_t<const float> x, vector_t<const float> weight, float eps);

    void matvec(vector_t<float> out, matrix_t<const float> W, vector_t<const float> x);

    void rope(matrix_t<float> t, std::ptrdiff_t pos, float theta_base);

    void softmax(vector_t<float> x);

    void add(vector_t<float> out, vector_t<const float> a, vector_t<const float> b);

    void silu_mul(vector_t<float> out, vector_t<const float> gate, vector_t<const float> up);

    std::ptrdiff_t argmax(vector_t<const float> x);
}
