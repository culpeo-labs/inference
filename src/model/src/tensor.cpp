#include <model/tensor.h>

using namespace culpeo::inference;

model::tensor::tensor(std::shared_ptr<const void> data, std::size_t size, util::data_type dtype, std::vector<std::size_t> shape)
    : m_data{ std::move(data) }, m_size{ size }, m_dtype{ dtype }, m_shape{ shape }
{
}

