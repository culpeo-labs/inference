#pragma once

#include <cstddef>
#include <utility>

namespace culpeo::inference::util
{
    template<std::size_t N, typename F>
    constexpr void unroll(F&& f)
    {
        [&]<std::size_t... Is>(std::index_sequence<Is...>)
        {
            (f(std::integral_constant<std::size_t, Is>{}) ,...);
        }(std::make_index_sequence<N>{});
    }
}
