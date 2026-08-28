#pragma once

#include <cassert>
#include <cstddef>
#include <cstring>
#include <memory>

#include <util/mdarray.h>
#include <util/types.h>

namespace culpeo::inference::cache {

    template<util::data_type D>
    class cache_block
    {
    public:
        using vector_type = typename util::matrix<D>::template mut<1>;
        using cmatrix_type = typename util::matrix<D>::template view<2>;

        explicit cache_block(std::size_t max_sequence_length, std::size_t heads, std::size_t dims):
            m_heads{ heads },
            m_max_sequence_length{ max_sequence_length },
            m_dims{ dims },
            m_size{ max_sequence_length * heads * dims },
            m_data{ max_sequence_length * heads, dims }
        {}

        void write(std::size_t head, std::size_t pos, const vector_type& vec)
        {
            assert(vec.extent(0) == m_dims);
            assert(head < m_heads);
            assert(pos < m_max_sequence_length);
            auto view = m_data.template view<3>(m_heads, m_max_sequence_length, m_dims);
            auto head_values = util::get_row(view, head);
            auto pos_values = util::get_row(head_values, pos);
            std::memcpy(pos_values.data_handle() , vec.data_handle(), vec.size() * sizeof(element_type));
        }

        cmatrix_type read(std::size_t head, std::size_t pos) const
        {
            assert(head < m_heads);
            assert(pos < m_max_sequence_length);
            auto view = m_data.template view<3>(m_heads, m_max_sequence_length, m_dims);
            auto head_values = util::get_row(view, head);
            return util::sub_view(head_values, pos + 1);
        }


    private:
        using element_type = util::mdarray<D, 2>::element_type;
        std::size_t m_heads;
        std::size_t m_max_sequence_length;
        std::size_t m_dims;
        std::size_t m_size;
        util::mdarray<D, 2> m_data;
    };
}