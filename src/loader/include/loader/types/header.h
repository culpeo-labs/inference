#pragma once

#include <map>
#include <string>

#include "loader/types/tensor.h"

namespace culpeo::loader::types {

    struct header
    {
        std::map<std::string, std::string> metadata;
        std::map<std::string, tensor> tensors;
    };

}