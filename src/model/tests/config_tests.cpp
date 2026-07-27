// Config parser tests. Each test writes a config.json into a temp dir and
// loads it through config::load<hf::llama_config>. The suite encodes the
// *contract* from our design discussion — fallbacks, validation, and
// parse-correctness — so tests that are currently red mark work still to do,
// not test bugs.
//
// Contract under test:
//   - required shape fields parse correctly (num_hidden_layers is the
//     canary: a wire-struct typo makes it silently zero)
//   - head_dim absent  -> hidden_size / num_attention_heads
//   - num_key_value_heads absent -> num_attention_heads (no GQA)
//   - num_attention_heads % num_kv_heads != 0 -> error (GQA precondition)
//   - rope_scaling absent -> tolerated (base/finetuned configs omit it)
//   - unknown keys (repackager cruft) -> tolerated
//   - missing required key -> error (not silent zero)
//   - malformed json / missing file -> error
#include <atomic>
#include <filesystem>
#include <fstream>
#include <glaze/json/write.hpp>
#include <initializer_list>
#include <map>
#include <sstream>
#include <string>
#include <string_view>

#include <glaze/glaze.hpp>
#include <catch2/catch_test_macros.hpp>

#include <model/config.h>
#include <model/hf/llama_config.h>

using namespace culpeo::inference;

namespace
{
    // RAII temp file: writes contents, removes on scope exit. Unique name per
    // instance so parallel Catch2 shards don't collide.
    struct temp_config
    {
        std::filesystem::path path;

        explicit temp_config(std::string_view contents)
        {
            static std::atomic<std::size_t> counter{ 0 };
            const auto name = "cfg_test_" + std::to_string(counter++) + ".json";
            path = std::filesystem::temp_directory_path() / name;
            std::ofstream out{ path, std::ios::binary | std::ios::trunc };
            out << contents;
        }
        ~temp_config() { std::error_code ec; std::filesystem::remove(path, ec); }

        temp_config(const temp_config &) = delete;
        temp_config & operator=(const temp_config &) = delete;
    };

    auto load(std::string_view contents)
    {
        temp_config cfg{ contents };
        return model::config::load<model::hf::llama_config>(cfg.path);
    }

    auto load(model::hf::llama_config contents, std::initializer_list<std::string_view> fields_to_remove = {}, std::map<std::string_view, std::string> fields_to_add = {})
    {
        std::map<std::string_view, std::string> fields{};

        fields.emplace("architectures", glz::write_json(contents.architectures).value());
        fields.emplace("attention_bias", contents.attention_bias ? "true" : "false");
        if (contents.head_dim.has_value())
        {
            fields.emplace("head_dim", std::to_string(*contents.head_dim));
        }
        fields.emplace("hidden_size", std::to_string(contents.hidden_size));
        fields.emplace("intermediate_size", std::to_string(contents.intermediate_size));
        fields.emplace("num_attention_heads", std::to_string(contents.num_attention_heads));
        fields.emplace("num_hidden_layers", std::to_string(contents.num_hidden_layers));
        if (contents.num_key_value_heads.has_value())
        {
            fields.emplace("num_key_value_heads", std::to_string(*contents.num_key_value_heads));
        }
        fields.emplace("rms_norm_eps", std::to_string(contents.rms_norm_eps));
        if (contents.rope_scaling.has_value())
        {
            fields.emplace("rope_scaling", glz::write_json(*contents.rope_scaling).value());
        }
        fields.emplace("rope_theta", std::to_string(contents.rope_theta));
        fields.emplace("tie_word_embeddings", contents.tie_word_embeddings ? "true" : "false");
        fields.emplace("torch_dtype", "\"" + contents.torch_dtype + "\"");
        fields.emplace("vocab_size", std::to_string(contents.vocab_size));
        for (const auto & field : fields_to_add)
        {
            fields.emplace(field.first, field.second);
        }
        for (const auto & field : fields_to_remove)
        {
            fields.erase(field);
        }

        std::ostringstream ss;
        ss << "{";
        bool first = true;
        for (const auto & [key, value] : fields)
        {
            if (!first) ss << ",";
            first = false;
            ss << "\"" << key << "\":" << value;
        }
        ss << "}";
        std::string cfg_str{ ss.str() };
        return load(cfg_str);
    }

    // A complete, valid config mirroring Llama 3.2 1B. Helper builders below
    // mutate copies of this so each test varies exactly one thing.
    const model::hf::llama_config full_config{
        .architectures = { "LlamaForCausalLM" },
        .attention_bias = false,
        .head_dim = 64,
        .hidden_size = 2048,
        .intermediate_size = 8192,
        .num_attention_heads = 32,
        .num_hidden_layers = 16,
        .num_key_value_heads = 8,
        .rms_norm_eps = 1e-05f,
        .rope_scaling = model::hf::rope_scaling{
            .factor = 32.0f,
            .high_freq_factor = 4.0f,
            .low_freq_factor = 1.0f,
            .original_max_position_embeddings = 8192,
            .rope_type = "llama3"
        },
        .rope_theta = 500000.0f,
        .tie_word_embeddings = true,
        .torch_dtype = "bfloat16",
        .vocab_size = 128256
    };
}

TEST_CASE("config: full valid config parses with correct shapes", "[config]")
{
    const auto cfg = load(full_config);
    REQUIRE(cfg.has_value());
    CHECK(cfg->hidden_size == 2048);
    CHECK(cfg->intermediate_size == 8192);
    CHECK(cfg->num_attention_heads == 32);
    // Canary: with the wire-struct field misspelled, glaze skips the real
    // "num_hidden_layers" key (unknown-keys tolerated) and this reads 0.
    CHECK(cfg->num_hidden_layers == 16);
    CHECK(cfg->vocab_size == 128256);
    CHECK(cfg->rms_norm_eps == 1e-05f);
    CHECK(cfg->rope_theta == 500000.0f);
    CHECK(cfg->tie_word_embeddings == true);
}

TEST_CASE("config: head_dim and num_kv_heads carried into engine config", "[config]")
{
    // These are the fields the forward pass cannot run without. If the engine
    // config lacks them, this test won't compile — which is itself the signal.
    const auto cfg = load(full_config);
    REQUIRE(cfg.has_value());
    CHECK(cfg->head_dim == 64);
    CHECK(cfg->num_key_value_heads == 8);
}

TEST_CASE("config: head_dim absent falls back to hidden_size/num_heads", "[config]")
{
    // Remove the explicit head_dim line; resolver must derive 2048/32 = 64.
    const auto cfg = load(full_config, { "head_dim" });
    REQUIRE(cfg.has_value());
    CHECK(cfg->head_dim == 64);
}

TEST_CASE("config: num_kv_heads absent falls back to num_attention_heads", "[config]")
{
    const auto cfg = load(full_config, { "num_key_value_heads" });
    REQUIRE(cfg.has_value());
    CHECK(cfg->num_key_value_heads == 32);
}

TEST_CASE("config: non-divisible head counts are rejected", "[config]")
{
    // GQA precondition: num_attention_heads % num_kv_heads == 0. 32 % 7 != 0,
    // so kvh = qh / group_size would be ill-defined; resolver must error.
    auto bad_cfg = full_config;
    bad_cfg.num_key_value_heads = 7;
    const auto cfg = load(bad_cfg);
    CHECK_FALSE(cfg.has_value());
}

TEST_CASE("config: rope_scaling absent is tolerated", "[config]")
{
    const auto cfg = load(full_config, { "rope_scaling" });
    REQUIRE(cfg.has_value());
    CHECK(cfg->num_hidden_layers == 16);
}

TEST_CASE("config: unknown keys (repackager cruft) are tolerated", "[config]")
{
    const auto cfg = load(full_config, {}, { { "unsloth_fixed", "true" }, { "pretraining_tp", "1" }, { "transformers_version", "\"4.52.4\"" }, { "initializer_range", "0.02" }, { "some_future_field", "{\"nested\": [1, 2, 3]}" } });
    REQUIRE(cfg.has_value());
    CHECK(cfg->hidden_size == 2048);
}

TEST_CASE("config: missing required field errors, not silent zero", "[config]")
{
    // hidden_size omitted. Must NOT succeed with hidden_size == 0. This is the
    // case error_on_unknown_keys=false makes dangerous: absent required keys
    // are skipped unless the resolver validates them.
    const auto cfg = load(full_config, { "hidden_size" });
    CHECK_FALSE(cfg.has_value());
}

TEST_CASE("config: malformed json errors", "[config]")
{
    const auto cfg = load(R"({ "hidden_size": 2048, )");  // truncated
    CHECK_FALSE(cfg.has_value());
}

TEST_CASE("config: nonexistent file errors", "[config]")
{
    const auto path = std::filesystem::temp_directory_path() / "does_not_exist_9999.json";
    const auto cfg = model::config::load<model::hf::llama_config>(path);
    CHECK_FALSE(cfg.has_value());
}