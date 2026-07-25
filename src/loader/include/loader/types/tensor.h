#pragma once

#include <cstddef>
#include <vector>

#include <util/types.h>

namespace culpeo::loader::types {


    struct tensor_descriptor
    {
        inference::util::data_type dtype;
        std::vector<long> shape;
        std::vector<std::size_t> offsets;
    };

}