#pragma once

#include <cstddef>
#include <memory>

#include <util/types.h>

namespace culpeo::inference::util
{
    template <data_type D, std::size_t Rank>
    class mdarray
    {
    public:
        using mat_t = typename matrix<D>::template type<Rank>;
        using element_type = typename mat_t::element_type;

        template<typename... Ts>
        mdarray(Ts... extents):
            m_extents(extents...),
            m_size((1 * ... * extents)),
            m_data(std::make_unique<element_type[]>(m_size)),
            m_span(m_data.get(), m_extents)
        {
            static_assert(sizeof...(Ts) == Rank);
        }

        std::size_t size() const { return m_size; }

        mat_t mdspan() { return m_span; }

        auto operator[](std::ptrdiff_t i)
        {
            return m_span[i];
        }

    private:
        const mat_t::extents_type m_extents;
        const std::size_t m_size;
        std::unique_ptr<element_type[]> m_data;
        mat_t m_span;
    };
}
