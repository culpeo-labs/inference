#include <cmath>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <mdspan>
#include <string>
#include <string_view>
#include <vector>

#include <cache/kv_cache.h>
#include <model/config.h>
#include <model/model.h>
#include <operations/ops.h>
#include <util/mdarray.h>
#include <util/timers.h>
#include <util/types.h>

using namespace std::string_view_literals;

using namespace culpeo::inference;

struct buffers
{
    using vector_t = util::mdarray<util::data_type::F32, 1>;

    buffers(const model::config & config, std::size_t max_seq):
        X{ config.hidden_size },
        H{ config.hidden_size },
        Q{ config.num_attention_heads * config.head_dim },
        K{ config.num_key_value_heads * config.head_dim },
        V{ config.num_key_value_heads * config.head_dim },
        scores{ max_seq },
        attn{ config.num_attention_heads * config.head_dim },
        attn_proj{ config.hidden_size },
        gate{ config.intermediate_size },
        up{ config.intermediate_size },
        ffn_out{ config.hidden_size },
        logits{ config.vocab_size }
    {}

    vector_t X;
    vector_t H;
    vector_t Q;
    vector_t K;
    vector_t V;
    vector_t scores;
    vector_t attn;
    vector_t attn_proj;
    vector_t gate;
    vector_t up;
    vector_t ffn_out;
    vector_t logits;

};

struct context
{
    const model::model model;
    cache::kv_cache<util::data_type::F32> cache;
    buffers buffers;
};

std::ptrdiff_t forward(context & context, const auto token)
{
    auto & [ model, cache, buffers] = context;
    auto & config = model.config;
    auto & [ X, H,  Q, K, V, scores, attn, attn_proj, gate, up, ffn_out, logits ] = buffers;
    auto pos = cache.cursor();

    operations::embed(X.mdspan(), model.embed_tokens, token);
    for (std::size_t l{ 0 }; l < model.config.num_hidden_layers; l++)
    {
        const auto & layer = model.layers[l];
        operations::rmsnorm(H.mdspan(), X.mdspan(), layer.input_layernorm, config.rms_norm_eps);
        operations::matvec(Q.mdspan(), layer.q_proj, H.mdspan());
        auto q = Q.view<2>(config.num_attention_heads, config.head_dim);
        operations::matvec(K.mdspan(), layer.k_proj, H.mdspan());
        auto k = K.view<2>(config.num_key_value_heads, config.head_dim);
        operations::matvec(V.mdspan(), layer.v_proj, H.mdspan());
        auto v = V.view<2>(config.num_key_value_heads, config.head_dim);
        operations::rope(q, pos, config.rope_theta);
        operations::rope(k, pos, config.rope_theta);
        for (std::size_t h = 0; h < config.num_key_value_heads; h++)
        {
            auto k_row = util::get_row(k, h);
            auto v_row = util::get_row(v, h);
            cache.write_kv(l, h, k_row, v_row);
        }

        for (std::size_t h = 0; h < config.num_attention_heads; h++)
        {
            std::size_t kvh = h / (config.num_attention_heads / config.num_key_value_heads);
            auto keys = cache.keys(l, kvh);
            auto values = cache.values(l, kvh);
            auto q_h = util::get_row(q, h);
            auto scores_view = util::sub_view(scores.mdspan(), keys.extent(0));
            operations::matvec(scores_view, keys, q_h);
            const float inv = 1.0f / std::sqrt(float(config.head_dim));
            // TODO: add operations::scale
            for (std::size_t n{0}; n < scores_view.extent(0); n++)
            {
                scores_view[n] *= inv;
            }
            operations::softmax(scores_view);
            auto attn_2d = attn.view<2>(config.num_attention_heads, config.head_dim);

            auto attn_h = util::get_row(attn_2d, h);
            for (std::size_t d{0}; d < config.head_dim; d++)
            {
                float acc{ 0 };
                for (std::size_t t{0}; t < values.extent(0); t++)
                {
                    acc += scores_view[t] * values[t, d];
                }
                attn_h[d] = acc;
            }
        }
        operations::matvec(attn_proj.mdspan(), layer.o_proj, attn.mdspan());
        operations::add(X.mdspan(), X.mdspan(), attn_proj.mdspan());
        operations::rmsnorm(H.mdspan(), X.mdspan(), layer.post_attention_layernorm, config.rms_norm_eps);
        operations::matvec(gate.mdspan(), layer.gate_proj, H.mdspan());
        operations::matvec(up.mdspan(), layer.up_proj, H.mdspan());
        operations::silu_mul(gate.mdspan(), gate.mdspan(), up.mdspan());
        operations::matvec(ffn_out.mdspan(), layer.down_proj, gate.mdspan());
        operations::add(X.mdspan(), X.mdspan(), ffn_out.mdspan());
    }
    cache.advance_cursor();
    operations::rmsnorm(H.mdspan(), X.mdspan(), model.norm, config.rms_norm_eps);
    operations::matvec(logits.mdspan(), model.lm_head, H.mdspan());
    auto next = operations::argmax(logits.mdspan());
    return next;
}

void log(std::string_view message)
{
    static const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();

    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
    std::cerr << "[" << elapsed << "] " << message << std::endl;
}

int main(int argc, char ** argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <checkpoint_path>" << std::endl;
        return 1;
    }
    const std::string_view checkpoint_path{ argv[1] };
    std::vector<std::ptrdiff_t> prompt;
    for (std::string tok; std::cin >> tok; ) {
        prompt.push_back(std::stoull(tok.c_str()));
    }

    auto model = model::model::load(checkpoint_path);
    if (!model)
    {
        std::cerr << "Failed to load model from checkpoint: " << checkpoint_path << std::endl;
        return 1;
    }
    log(std::format("Model loaded successfully from checkpoint: {}", checkpoint_path));
    const auto & config = model->config;
    constexpr auto max_seq{ 100ull };
    cache::kv_cache<util::data_type::F32> cache{
        config.num_hidden_layers,
        config.num_key_value_heads,
        config.head_dim ,
        max_seq,
    };

    context ctx
    {
        .model = std::move(model).value(),
        .cache = std::move(cache),
        .buffers = buffers{ model->config, max_seq },
    };

    std::ptrdiff_t next{};
    for (std::size_t i{ 0 }; i < prompt.size(); i++)
    {
        static util::scope_timer input_timer("input-loop");
        auto _ = input_timer.probe();
        log(std::format("Processing input token {}", prompt[i]));
        next = forward(ctx, prompt[i]);
    }

    for (int step = 0; step < 50; step++) {
        static util::scope_timer generation_timer("generation-loop");
        auto _ = generation_timer.probe();
        std::cout << next << std::endl;
        if (next == 128001 || next == 128009) break;   // <|end_of_text|> / <|eot_id|>
        log(std::format("Generating next token (prev: {})", next));
        next = forward(ctx, next);
        log(std::format("Generated next token {}", next));
    }
    return 0;
}