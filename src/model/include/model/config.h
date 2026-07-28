#pragma once

#include <expected>
#include <type_traits>
#include <filesystem>

#include <model/model_family.h>
#include <model/hf/llama_config.h>

namespace culpeo::inference::model
{
    struct config
    {
        model_family family;
        std::size_t head_dim;
        std::size_t hidden_size;
        std::size_t intermediate_size;
        std::size_t num_attention_heads;
        std::size_t num_hidden_layers;
        std::size_t num_key_value_heads;
        float rms_norm_eps;
        float rope_theta;
        bool tie_word_embeddings;
        std::size_t vocab_size;
        std::optional<std::size_t> original_max_position_embeddings;

        template<typename ConfigT>
        static std::expected<config, std::string> load(const std::filesystem::path &)
        {
            static_assert(std::false_type::value, "Unsupported config type");
        }
    };

    template<>
    std::expected<config, std::string> config::load<hf::llama_config>(const std::filesystem::path & path);
}