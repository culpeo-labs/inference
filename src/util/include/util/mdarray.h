#pragma once

#include <cstddef>
#include <mdspan>
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

        template<std::size_t ViewRank, typename... Ts>
        typename matrix<D>::template type<ViewRank> view(Ts... extents)
        {
            static_assert(sizeof...(Ts) == ViewRank);
            const auto view_size = (1 * ... * extents);
            assert(view_size == m_size);
            return (typename matrix<D>::template type<ViewRank>){ m_data.get(), extents...};

        }

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


    template<typename ElementType, typename Extents, typename AccessorPolicy>
    auto get_row(std::mdspan<ElementType, Extents, std::layout_right, AccessorPolicy> mat, std::size_t row)
    {
        static_assert(decltype(mat)::rank() == 2);
        assert(row < mat.extent(0));
        return std::mdspan<ElementType, std::dextents<std::size_t, 1>, std::layout_right, AccessorPolicy>
        {
            mat.data_handle() + mat.extent(1) * row,
            mat.extent(1)
        };
    }

    template<typename ElementType, typename Extents, typename AccessorPolicy>
    auto sub_view(std::mdspan<ElementType, Extents, std::layout_right, AccessorPolicy> mat, std::size_t len)
    {
        static_assert(decltype(mat)::rank() == 1);
        assert(len < mat.extent(0));
        return std::mdspan<ElementType, std::dextents<std::size_t, 1>, std::layout_right, AccessorPolicy>
        {
            mat.data_handle(),
            len
        };
    }
}
