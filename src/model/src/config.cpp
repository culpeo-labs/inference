#include <expected>
#include <filesystem>
#include <glaze/core/opts.hpp>
#include <glaze/glaze.hpp>

#include <glaze/json/read.hpp>
#include <model/config.h>
#include <model/hf/llama_config.h>

using namespace culpeo::inference;

template<>
std::expected<model::config, std::string> model::config::load<model::hf::llama_config>(const std::filesystem::path & path)
{
    if (!std::filesystem::exists(path))
    {
        return std::unexpected("File does not exist.");
    }
    hf::llama_config config{};

    const auto error = glz::read_file_json<glz::opts{ .error_on_unknown_keys = false }>(config, path.c_str(), std::string{});
    if (error)
    {
        return std::unexpected("Error parsing headers.");
    }

    return model::config{
        .hidden_size = config.hidden_size,
        .intermediate_size = config.intermediate_size,
        .num_attention_heads = config.num_attention_heads,
        .num_hidden_layers = config.num_hiddlen_layers,
        .rms_norm_eps = config.rms_norm_eps,
        .rope_theta = config.rope_theta,
        .tie_word_embeddings = config.tie_word_embeddings,
        .vocab_size = config.vocab_size,
    };
}