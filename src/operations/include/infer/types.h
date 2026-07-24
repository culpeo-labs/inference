#pragma once

#include <concepts>
#include <mdspan>

namespace culpeo::inference::types {

    template<typename M>
    concept float_matrix = (M::rank() == 2 && std::convertible_to<typename M::reference, float>);

    template<typename T, typename AccessorPolicy = std::default_accessor<T>>
    using mat_t = std::mdspan<T, std::dextents<std::size_t, 2>, std::layout_right, AccessorPolicy>;

    template<typename T, typename AccessorPolicy = std::default_accessor<const T>>
    using cmat_t = mat_t<const T, AccessorPolicy>;

    template<typename T>
        requires(std::floating_point<T>)
    using vec_t = std::span<T>;

    template<typename T>
        requires(std::floating_point<T>)
    using cvec_t = vec_t<const T>;
}