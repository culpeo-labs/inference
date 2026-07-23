#pragma once

#include "loader/types/header.h"
#include <filesystem>

namespace culpeo::loader {
    class safe_tensors
    {
        types::header m_header;
    public:
        static safe_tensors load(const std::filesystem::path & path);

    private:
        inline safe_tensors(types::header header): m_header{ std::move(header) }
        {}
    };

}