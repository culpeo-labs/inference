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
        tensor(std::shared_ptr<const void> data, std::size_t size, util::data_type dtype, std::vector<std::size_t> shape);

        template<util::data_type dtype>
        std::expected<typename util::matrix<dtype>::const_type, std::string> as_mat() const
        {
            using mat_t = typename util::matrix<dtype>::const_type;
            if (dtype != m_dtype)
            {
                return std::unexpected{ "Request type does not match tensor type." };
            }
            if (m_shape.size() != 2)
            {
                return std::unexpected{ "Tensor is not a matrix" };
            }
            return mat_t{ static_cast<mat_t::data_handle_type>(m_data.get()), m_shape[0], m_shape[1] };
        }
    private:
        std::shared_ptr<const void> m_data;
        std::size_t m_size;
        util::data_type m_dtype;
        std::vector<std::size_t> m_shape;
    };
}