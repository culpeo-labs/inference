#pragma once

#include <mdspan>
#include <type_traits>

namespace culpeo::inference::util
{
    template<typename T, bool C>
        requires(!std::is_pointer_v<T>)
    using conditionally_const_t = std::conditional_t<C, std::add_const_t<T>, T>;

    template<typename T>
    struct is_mdspan: std::false_type {};

    template<typename T, typename Extents, typename Layout, typename Accessor>
    struct is_mdspan<std::mdspan<T, Extents, Layout, Accessor>>: std::true_type {};

    template<typename T>
    inline constexpr bool is_mdspan_v = is_mdspan<T>::value;

    template<typename T, typename Element>
    concept mdspan_of = is_mdspan_v<T> && std::same_as<std::remove_const_t<typename T::element_type>, Element>;
}