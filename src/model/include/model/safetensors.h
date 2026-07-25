#pragma once

#include <cstddef>
#include <expected>
#include <filesystem>
#include <map>
#include <memory>
#include <string_view>
#include <vector>

#include <model/tensor.h>
#include <util/types.h>

namespace culpeo::inference::model
{
    class safetensors
    {
    public:
        struct tensor_descriptor
        {
            util::data_type dtype;
            std::vector<std::size_t> shape;
            std::vector<std::size_t> offsets;
        };

        static std::expected<safetensors, std::string> load(const std::filesystem::path& path);

        size_t tensor_count() const;

        std::vector<std::string_view> keys() const;

        std::expected<tensor, std::string> get_tensor(std::string_view name) const;
    private:
        safetensors(
            std::shared_ptr<void> data,
            std::map<std::string, tensor_descriptor, std::less<>> tensors,
            std::map<std::string, std::string, std::less<>> metadata,
            std::size_t data_offset);

        std::shared_ptr<void> m_data;
        std::map<std::string, tensor_descriptor, std::less<>> m_tensors;
        std::map<std::string, std::string, std::less<>> m_metadata;
        std::size_t m_data_offset;
    };
}