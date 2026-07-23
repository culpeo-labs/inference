
#include "loader/loader.h"
#include "loader/types/header.h"
#include "loader/types/tensor.h"
#include <exception>
#include <filesystem>
#include <fstream>
#include <glaze/core/common.hpp>
#include <glaze/core/context.hpp>
#include <glaze/core/opts.hpp>
#include <iostream>
#include <string_view>
#include <vector>

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
    return safe_tensors{ std::move(h) };
}
