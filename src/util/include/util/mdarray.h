#pragma once

#include <cassert>
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
        using mat_t = typename matrix<D>::template mut<Rank>;
        using view_t = typename matrix<D>::template view<Rank>;
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

        view_t mdspan() const { return view_t{ m_data.get(), m_extents }; }

        template<std::size_t ViewRank, typename... Ts>
        typename matrix<D>::template mut<ViewRank> view(Ts... extents)
        {
            static_assert(sizeof...(Ts) == ViewRank);
            const auto view_size = (1 * ... * extents);
            assert(view_size == m_size);
            return (typename matrix<D>::template mut<ViewRank>){ m_data.get(), extents...};
        }

        template<std::size_t ViewRank, typename... Ts>
        typename matrix<D>::template view<ViewRank> view(Ts... extents) const
        {
            static_assert(sizeof...(Ts) == ViewRank);
            const auto view_size = (1 * ... * extents);
            assert(view_size == m_size);
            return (typename matrix<D>::template view<ViewRank>){ m_data.get(), extents...};
        }

        template<typename... Idx>
        auto operator[](Idx... i) const &
        {
            return m_span[i...];
        }

        template<typename... Idx>
        auto & operator[](Idx... i) &
        {
            return m_span[i...];
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
        using mat_t = decltype(mat);
        static_assert(mat_t::rank() > 1);
        assert(row < mat.extent(0));
        auto extents = [&]()
        {
            std::array<std::size_t, mat_t::rank() - 1> extents{};
            for (size_t i{ 1}; i < mat_t::rank(); i++)
            {
                extents[i - 1] = mat.extent(i);
            }
            return extents;
        }();
        return std::mdspan<ElementType, std::dextents<std::size_t, decltype(mat)::rank() - 1>, std::layout_right, AccessorPolicy>
        {
            mat.data_handle() + mat.stride(0) * row,
            extents
        };
    }

    template<typename ElementType, typename Extents, typename AccessorPolicy>
    auto sub_view(std::mdspan<ElementType, Extents, std::layout_right, AccessorPolicy> mat, std::size_t count)
    {
        using mat_t = decltype(mat);
        static_assert(decltype(mat)::rank() > 0);
        assert(count <= mat.extent(0));
        auto extents = [&]()
        {
            std::array<std::size_t, mat_t::rank()> extents{};
            extents[0] = count;
            for (size_t i{ 1}; i < mat_t::rank(); i++)
            {
                extents[i] = mat.extent(i);
            }
            return extents;
        }();
        return std::mdspan<ElementType, std::dextents<std::size_t, mat_t::rank()>, std::layout_right, AccessorPolicy>
        {
            mat.data_handle(),
            extents
        };
    }
}
