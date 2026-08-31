#pragma once

#include <cstddef>
#include <vector>

#include <cache/cache_block.h>
#include <util/types.h>

namespace culpeo::inference::cache
{
    template<util::data_type data_type>
    class kv_cache
    {
    public:
        using vector_type = cache_block<data_type>::vector_type;
        using cmatrix_type = cache_block<data_type>::cmatrix_type;

        explicit kv_cache(std::size_t num_layers, std::size_t num_heads, std::size_t head_dim, std::size_t max_seq_len)
            : m_cursor{ 0 }, m_k_caches{}, m_v_caches{}
        {
            m_k_caches.reserve(num_layers);
            m_v_caches.reserve(num_layers);
            for (std::size_t i{ 0 }; i < num_layers; i++)
            {
                m_k_caches.emplace_back(max_seq_len, num_heads, head_dim);
                m_v_caches.emplace_back(max_seq_len, num_heads, head_dim);
            }
        }

        void reset_cursor()
        {
            m_cursor = 0;
        }

        void advance_cursor()
        {
            m_cursor++;
        }

        std::size_t cursor() const
        {
            return m_cursor;
        }

        void write_kv(std::size_t layer, std::size_t head, const vector_type & k, const vector_type & v)
        {
            assert(layer < m_k_caches.size());
            m_k_caches[layer].write(head, m_cursor, k);
            m_v_caches[layer].write(head, m_cursor, v);
        }

        cmatrix_type keys(std::size_t layer, std::size_t head) const
        {
            assert(layer < m_k_caches.size());
            return m_k_caches[layer].read(head, m_cursor);
        }

        cmatrix_type values(std::size_t layer, std::size_t head) const
        {
            assert(layer < m_v_caches.size());
            return m_v_caches[layer].read(head);
        }
    private:
        std::size_t m_cursor;
        std::vector<cache_block<data_type>> m_k_caches;
        std::vector<cache_block<data_type, true>> m_v_caches;

    };
}