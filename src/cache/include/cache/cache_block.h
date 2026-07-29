#pragma once

#include <cassert>
#include <cstddef>
#include <cstring>
#include <memory>

#include <util/types.h>

namespace culpeo::inference::cache {

    template<util::data_type D>
    class cache_block
    {
    public:
        using vector_type = typename util::matrix<D>::template type<1>;
        using cmatrix_type = typename util::matrix<D>::template const_type<2>;

        explicit cache_block(std::size_t max_sequence_length, std::size_t heads, std::size_t dims):
            m_heads{ heads },
            m_max_sequence_length{ max_sequence_length },
            m_dims{ dims },
            m_size{ max_sequence_length * heads * dims },
            m_buffer{ std::make_unique<element_type[]>(m_size) }
        {}

        void write(std::size_t head, std::size_t pos, const vector_type& vec)
        {
            assert(vec.extent(0) == m_dims);
            assert(head < m_heads);
            assert(pos < m_max_sequence_length);
            auto offset = head * m_max_sequence_length * m_dims + pos * m_dims;
            std::memcpy(m_buffer.get() + offset, vec.data_handle(), vec.size() * sizeof(element_type));
        }


        cmatrix_type read(std::size_t head, std::size_t pos) const
        {
            assert(head < m_heads);
            assert(pos < m_max_sequence_length);
            auto offset = head * m_max_sequence_length * m_dims;
            return cmatrix_type{ m_buffer.get() + offset, pos + 1, m_dims };
        }


    private:
        using element_type = vector_type::element_type;
        std::size_t m_heads;
        std::size_t m_max_sequence_length;
        std::size_t m_dims;
        std::size_t m_size;
        std::unique_ptr<element_type[]> m_buffer;
    };
}