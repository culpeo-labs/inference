
#include "model/tensor.h"
#include <expected>
#include <filesystem>
#include <memory>
#include <ranges>

#include <glaze/core/context.hpp>
#include <string_view>
#include <sys/mman.h>

#include <glaze/glaze.hpp>

#include <model/safetensors.h>
#include <util/scope_guard.h>
#include <util/types.h>

using namespace culpeo::inference;

namespace
{
    struct header
    {
        std::map<std::string, std::string, std::less<>> metadata;
        std::map<std::string, model::safetensors::tensor_descriptor, std::less<>> tensors;
    };
}

namespace glz
{
    template<>
    struct meta<util::data_type>
    {
        using enum util::data_type;

        static constexpr auto value = glz::enumerate(
            BF16,
            F64,
            F32,
            F16,
            I64,
            I32,
            I16,
            I8,
            U8,
            BOOL);

    };

    template<>
    struct meta<header>
    {
        using T = header;

        static constexpr auto value = object("__metadata__", &T::metadata);

        static constexpr auto unknown_read{&T::tensors};

        // static constexpr auto unknown_write{&T::tensors};
    };
}

std::expected<model::safetensors, std::string> model::safetensors::load(const std::filesystem::path& path)
{
    const auto file_size = std::filesystem::file_size(path);
    auto fd = fopen(path.c_str(), "rb");
    if (!fd)
    {
        return std::unexpected("Failed to open file");
    }
    util::scope_guard guard([fd]() { if (fd) fclose(fd); });
    auto file_data = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fileno(fd), 0);
    if (file_data == MAP_FAILED)
    {
        return std::unexpected("Failed to mmap file");
    }
    std::shared_ptr<void> data_ptr{ file_data, [file_size](void* p) { munmap(p, file_size); } };
    std::uint64_t header_size = *reinterpret_cast<std::uint64_t*>(file_data);
    std::string_view json{ reinterpret_cast<const char*>(file_data) + sizeof(std::uint64_t), header_size };
    header h{};
    glz::context ctx{};
    const auto error = glz::read<glz::opts{ .error_on_unknown_keys = false }>(h, json, ctx);
    if (error)
    {
        return std::unexpected("Error parsing header");
    }
    const auto data_offset = sizeof(std::uint64_t) + header_size;
    return model::safetensors{ data_ptr, std::move(h.tensors), std::move(h.metadata), data_offset };
}

model::safetensors::safetensors(
    std::shared_ptr<void> data,
    std::map<std::string, tensor_descriptor, std::less<>> tensors, std::map<std::string, std::string, std::less<>> metadata, std::size_t data_offset)
    : m_data{ std::move(data) }, m_tensors{ std::move(tensors) }, m_metadata{ std::move(metadata) }, m_data_offset{ data_offset }
{
}

size_t model::safetensors::tensor_count() const
{
    return m_tensors.size();
}

std::vector<std::string_view> model::safetensors::keys() const
{
    return m_tensors | std::views::transform([](auto && i) { return std::string_view{i.first}; }) | std::ranges::to<std::vector<std::string_view>>();
}

std::expected<model::tensor, std::string> model::safetensors::get_tensor(std::string_view name) const
{
    if (!m_tensors.contains(name))
    {
        std::unexpected{ "Tensor not found." };
    }
    const auto & descriptor = m_tensors.find(name)->second;
    auto offset = descriptor.offsets[0] + m_data_offset;
    auto size = descriptor.offsets[1] - descriptor.offsets[0];
    std::shared_ptr<const void> data{ m_data, reinterpret_cast<const void *>(static_cast<const uint8_t *>(m_data.get()) + offset) };
    return tensor{ data, size, descriptor.dtype, descriptor.shape };
}