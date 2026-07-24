#pragma once

#include <cstddef>
#include <vector>
namespace culpeo::loader::types {

    enum class data_type
    {
        BF16,
        F64,
        F32,
        F16,
        I64,
        I32,
        I16,
        I8,
        U8,
        BOOL,
    };

    struct tensor_descriptor
    {
        data_type dtype;
        std::vector<long> shape;
        std::vector<std::size_t> offsets;
    };

}