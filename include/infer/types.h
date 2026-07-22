#pragma once

#include <concepts>
#include <mdspan>

namespace culpeo::inference::types {

    template<typename T>
        requires(std::floating_point<T>)
    using mat_t = std::mdspan<T, std::dextents<std::size_t, 2>>;

    template<typename T>
        requires(std::floating_point<T>)
    using cmat_t = mat_t<const T>;

    template<typename T>
        requires(std::floating_point<T>)
    using vec_t = std::span<T>;

    template<typename T>
        requires(std::floating_point<T>)
    using cvec_t = vec_t<const T>;
}