#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include <util/types.h>

namespace culpeo::inference::model::hf
{
    struct rope_scaling // this could potentially move to util for when we incorporate into our rope
    {
        float factor;
        float high_freq_factor;
        float low_freq_factor;
        std::size_t original_max_position_embeddings;
        std::string rope_type; // could this be a union.
    };

    struct llama_config
    {
        std::vector<std::string> architectures;
        bool attention_bias;
        std::optional<size_t> head_dim;
        std::size_t hidden_size;
        std::size_t intermediate_size;
        std::size_t num_attention_heads;
        std::size_t num_hidden_layers;
        std::optional<size_t> num_key_value_heads;
        float rms_norm_eps;
        std::optional<rope_scaling> rope_scaling;
        float rope_theta;
        bool tie_word_embeddings;
        std::string torch_dtype;
        std::size_t vocab_size;
    };
}