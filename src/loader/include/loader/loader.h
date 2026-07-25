#pragma once

#include <util/types.h>
#include "loader/types/header.h"
#include <expected>
#include <filesystem>
#include <memory>
#include <string_view>
#include <vector>
#include <stdfloat>

namespace culpeo::inference::loader {

    class tensor
    {
        std::shared_ptr<FILE> m_file;
        std::unique_ptr<void, std::function<void(void *)>> m_data;
        const types::tensor_descriptor m_desc;
    public:
        tensor(std::shared_ptr<FILE> file, std::unique_ptr<void, std::function<void(void *)>> data, const types::tensor_descriptor desc);
        ~tensor() = default;
        tensor(const tensor &) = delete;
        tensor(tensor &&) = default;
        tensor & operator=(const tensor &) = delete;
        // tensor & operator=(tensor &&) = default;

        inline util::data_type dtype() const { return m_desc.dtype; }

        template<util::data_type D>
        inline util::matrix<D>::const_type as_cmat() const
        {
            static_assert(D == m_desc.dtype, "Tensor data type does not match requested data type");
            using T = typename util::matrix<D>::const_type;
            return T{
                reinterpret_cast<T::data_handle_type>(m_data.get()), m_desc.shape[0], m_desc.shape[1] };
        }
    };


    class safe_tensors
    {
        types::header m_header;
        std::shared_ptr<FILE> m_file;
        size_t m_data_offset;
    public:
        static safe_tensors load(const std::filesystem::path & path);

        size_t tensor_count() const;
        std::vector<std::string_view> keys() const;

        std::expected<tensor, std::string> get_tensor(std::string_view name) const;
    private:
        inline safe_tensors(types::header header, std::shared_ptr<FILE> file, size_t data_offset)
            : m_header{ std::move(header) }, m_file{ std::move(file) }, m_data_offset{ data_offset }
        {}
    };



}