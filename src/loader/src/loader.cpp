
#include "loader/loader.h"
#include <util/types.h>
#include "loader/types/header.h"
#include "loader/types/tensor.h"
#include <cstddef>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <glaze/core/common.hpp>
#include <glaze/core/context.hpp>
#include <glaze/core/opts.hpp>
#include <iostream>
#include <ranges>
#include <string_view>
#include <vector>
#include <stdfloat>

#include <glaze/glaze.hpp>


using namespace culpeo::loader;

template<>
struct glz::meta<types::header>
{
    using T = types::header;

    static constexpr auto value = object("__metadata__", &T::metadata);

    static constexpr auto unknown_read{&T::tensors};

    static constexpr auto unknown_write{&T::tensors};
};

template<>
struct glz::meta<types::data_type>
{
    using enum types::data_type;

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

uint64_t read_u64_le(std::ifstream & file)
{
    std::uint64_t value;
    file.read(reinterpret_cast<char *>(&value), sizeof(std::uint64_t));
    return value;
}

safe_tensors safe_tensors::load(const std::filesystem::path & path)
{
    std::ifstream file(path, std::ios::binary);
    const auto header_size = read_u64_le(file);
    std::vector<char> buffer(header_size);
    file.read(buffer.data(), header_size);
    std::string_view json{ buffer.data(), header_size };
    types::header h{};
    glz::context ctx{};
    const auto error = glz::read<glz::opts{ .error_on_unknown_keys = false }>(h, json, ctx);
    if (error)
    {
        std::cerr << "Error parsing header.\n" << error.count << std::endl;
        std::terminate();
    }
    auto file_ptr = fopen(path.c_str(), "rb");
    std::shared_ptr<FILE> ptr{ file_ptr, [](FILE * f) { fclose(f); } };
    return safe_tensors{ std::move(h), ptr, header_size + sizeof(std::uint64_t) };
}

size_t safe_tensors::tensor_count() const
{
    return m_header.tensors.size();
}

std::vector<std::string_view> safe_tensors::keys() const
{
    return m_header.tensors | std::views::transform([](auto && i) { return std::string_view{i.first}; }) | std::ranges::to<std::vector<std::string_view>>();
}


#include <sys/mman.h>

std::expected<tensor, std::string> safe_tensors::get_tensor(std::string_view name) const
{
    if (!m_header.tensors.contains(name))
    {
        return std::unexpected{ "Tensor not found" };
    }
    const auto & desc = m_header.tensors.find(name)->second;
    const auto offset = desc.offsets[0] + m_data_offset;
    const auto size = desc.offsets[1] - desc.offsets[0];
    auto data = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fileno(m_file.get()), offset);
    if (data == MAP_FAILED)
    {
        return std::unexpected{ "Failed to mmap tensor" };
    }
    std::unique_ptr<void, std::function<void(void *)>> data_ptr{ data, [size](void * p) { munmap(p, size); } };
    return tensor{ m_file, std::move(data_ptr), desc };
}

tensor::tensor(std::shared_ptr<FILE> file, std::unique_ptr<void, std::function<void(void *)>> data, types::tensor_descriptor desc)
    : m_file{ std::move(file) }, m_data{ std::move(data) }, m_desc{ std::move(desc) }
{
}

// culpeo::inference::types::cmat_t<float, std::bfloat16_t> tensor::as_cmat() const
// {
//     assert(m_desc.dtype == types::data_type::BF16);
//     assert(m_desc.shape.size() == 2);
//     return culpeo::inference::types::cmat_t<float, bf16_accessor>{ static_cast<bf16_accessor::element_type *>(m_data.get()), m_desc.shape[0], m_desc.shape[1] };
// }

