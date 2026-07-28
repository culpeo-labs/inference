#pragma once

#include <cstddef>
#include <expected>
#include <memory>
#include <vector>

#include <util/types.h>

namespace culpeo::inference::model
{
    class tensor
    {
    public:
        tensor(std::shared_ptr<const void> data, std::size_t original_offset, std::size_t size, util::data_type dtype, std::vector<std::size_t> shape);

        template<util::data_type dtype>
        std::expected<typename util::matrix<dtype>::template const_type<2>, std::string> as_mat() const
        {
            using mat_t = typename util::matrix<dtype>::template const_type<2>;
            if (dtype != m_dtype)
            {
                return std::unexpected{ "Request type does not match tensor type." };
            }
            if (m_shape.size() != 2)
            {
                return std::unexpected{ "Tensor is not a matrix" };
            }
            using T = std::remove_cvref_t<typename mat_t::element_type>;
            if (m_original_offset % alignof(T) != 0)
            {
                return std::unexpected{ "Tensor data is not aligned for requested type" };
            }
            if (m_size != sizeof(T) * m_shape[0] * m_shape[1])
            {
                return std::unexpected{ "Tensor size does not match shape" };
            }
            auto tensor_data = std::start_lifetime_as_array<T>(m_data.get(), m_size / sizeof(T));
            return mat_t{ tensor_data, m_shape[0], m_shape[1] };
        }

        template<util::data_type dtype>
        std::expected<typename util::matrix<dtype>::template const_type<1>, std::string> as_vec() const
        {
            using vec_t = typename util::matrix<dtype>::template const_type<1>;
            if (dtype != m_dtype)
            {
                return std::unexpected{ "Request type does not match tensor type." };
            }
            if (m_shape.size() != 1)
            {
                return std::unexpected{ "Tensor is not a vector" };
            }
            using T = std::remove_cvref_t<typename vec_t::element_type>;
            if (m_original_offset % alignof(T) != 0)
            {
                return std::unexpected{ "Tensor data is not aligned for requested type" };
            }
            if (m_size != sizeof(T) * m_shape[0])
            {
                return std::unexpected{ "Tensor size does not match shape" };
            }
            auto tensor_data = std::start_lifetime_as_array<T>(m_data.get(), m_size / sizeof(T));
            return vec_t{ tensor_data, m_shape[0] };
        }

    private:
        std::shared_ptr<const void> m_data;
        std::size_t m_original_offset;
        std::size_t m_size;
        util::data_type m_dtype;
        std::vector<std::size_t> m_shape;
    };
}