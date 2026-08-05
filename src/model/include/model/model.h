#pragma once

#include <expected>
#include <filesystem>
#include <string>
#include <vector>

#include <model/config.h>
#include <model/safetensors.h>
#include <util/types.h>

namespace culpeo::inference::model
{
    using bf16_cmat = util::matrix<util::data_type::BF16>::const_type<2>;
    using bf16_cvec = util::matrix<util::data_type::BF16>::const_type<1>;


    struct layer_weights
    {
        // TODO: evaluate prewiden to f32 (possibly wired behind compile def)
        // In the hot path we would do a lot of convesions, so worth eagerly widening
        // From a learning perspective, going naive first, measuring and optimizing later.
        bf16_cmat k_proj;
        bf16_cmat v_proj;
        bf16_cmat q_proj;
        bf16_cmat o_proj;
        bf16_cvec input_layernorm;
        bf16_cmat up_proj;
        bf16_cmat down_proj;
        bf16_cvec post_attention_layernorm;
        bf16_cmat gate_proj;
    };

    struct model
    {
        config config;
        safetensors safetensors;
        std::vector<layer_weights> layers;
        bf16_cmat embed_tokens;
        bf16_cvec norm;
        bf16_cmat lm_head;
        static std::expected<model, std::string> load(const std::filesystem::path & dir);
    };
}