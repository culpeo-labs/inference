#pragma once

#include <functional>
#include <map>
#include <string>

#include "loader/types/tensor.h"

namespace culpeo::inference::loader::types {

    struct header
    {
        std::map<std::string, std::string, std::less<>> metadata;
        std::map<std::string, tensor_descriptor, std::less<>> tensors;
    };

}