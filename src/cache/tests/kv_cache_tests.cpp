
#include <cstdint>
#include <vector>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cache/kv_cache.h>
#include <util/types.h>

using namespace culpeo::inference;
using culpeo::inference::util::data_type;

namespace
{
    // Encode a float into dtype D's storage; reads always come back as float.
    template<data_type D>
    struct codec;

    template<>
    struct codec<data_type::F32>
    {
        using storage = float;
        static storage encode(float v) { return v; }
        static float expected(float v) { return v; }          // lossless round-trip
    };

    template<>
    struct codec<data_type::BF16>
    {
        using storage = std::uint16_t;
        static storage encode(float v)
        {
            return static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(v) >> 16);
        }
        // What the value reads back as after the bf16 round-trip. bf16 is lossy
        // above its exact-integer range (e.g. 5000 -> 4992), so the cache's
        // faithful output is the round-tripped value, not the original float.
        static float expected(float v)
        {
            std::uint32_t bits = std::bit_cast<std::uint32_t>(v) >> 16;
            return std::bit_cast<float>(bits << 16);
        }
    };

    template<data_type D>
    auto make_vec(std::vector<typename codec<D>::storage>& backing)
    {
        using vt = typename cache::kv_cache<D>::vector_type;
        return vt{ backing.data(), backing.size() };
    }

    // Distinct, bf16-exact tag per (layer, head, pos, dim, is_value).
    // The 500 offset separates value tensors from key tensors so a K/V mixup
    // is unmistakable.
    constexpr float tag(std::size_t layer, std::size_t head, std::size_t pos,
                        std::size_t dim, bool is_value)
    {
        return static_cast<float>(layer * 1000 + head * 100 + pos * 10 + dim)
             + (is_value ? 5000.0f : 0.0f);
    }

    // Write one (layer, head) k and v at the cache's current cursor.
    template<data_type D>
    void write_tagged(cache::kv_cache<D>& c, std::size_t layer, std::size_t head,
                      std::size_t pos, std::size_t dims)
    {
        std::vector<typename codec<D>::storage> kb(dims), vb(dims);
        for (std::size_t d = 0; d < dims; ++d)
        {
            kb[d] = codec<D>::encode(tag(layer, head, pos, d, false));
            vb[d] = codec<D>::encode(tag(layer, head, pos, d, true));
        }
        c.write_kv(layer, head, make_vec<D>(kb), make_vec<D>(vb));
    }
}

TEMPLATE_TEST_CASE_SIG("kv_cache: single position, keys and values read back",
                       "[cache]", ((data_type D), D),
                       data_type::F32, data_type::BF16)
{
    constexpr std::size_t layers = 2, heads = 2, dims = 4, max_seq = 8;
    cache::kv_cache<D> c{ layers, heads, dims, max_seq };

    // Write all layers, all heads at cursor 0, then read back BEFORE advancing.
    for (std::size_t l = 0; l < layers; ++l)
        for (std::size_t h = 0; h < heads; ++h)
            write_tagged(c, l, h, 0, dims);

    // Contract: read while cursor still points at the just-written position.
    for (std::size_t l = 0; l < layers; ++l)
        for (std::size_t h = 0; h < heads; ++h)
        {
            auto k = c.keys(l, h);
            auto v = c.values(l, h);
            REQUIRE(k.extent(0) == 1);       // cursor 0 -> 1 row (pos+1)
            REQUIRE(v.extent(0) == 1);
            REQUIRE(k.extent(1) == dims);
            for (std::size_t d = 0; d < dims; ++d)
            {
                CHECK(k[0, d] == codec<D>::expected(tag(l, h, 0, d, false)));
                CHECK(v[0, d] == codec<D>::expected(tag(l, h, 0, d, true)));   // V distinct from K
            }
        }
}

TEMPLATE_TEST_CASE_SIG("kv_cache: layers are independent",
                       "[cache]", ((data_type D), D),
                       data_type::F32, data_type::BF16)
{
    constexpr std::size_t layers = 3, heads = 2, dims = 4, max_seq = 8;
    cache::kv_cache<D> c{ layers, heads, dims, max_seq };

    for (std::size_t l = 0; l < layers; ++l)
        for (std::size_t h = 0; h < heads; ++h)
            write_tagged(c, l, h, 0, dims);

    // Layer l's data is tagged with l; reading layer l must never surface
    // another layer's values.
    for (std::size_t l = 0; l < layers; ++l)
    {
        auto k = c.keys(l, 0);
        CHECK(k[0, 0] == codec<D>::expected(tag(l, 0, 0, 0, false)));   // encodes l in the value
    }
}

TEMPLATE_TEST_CASE_SIG("kv_cache: cursor drives read length in lockstep",
                       "[cache]", ((data_type D), D),
                       data_type::F32, data_type::BF16)
{
    constexpr std::size_t layers = 2, heads = 1, dims = 4, max_seq = 8;
    cache::kv_cache<D> c{ layers, heads, dims, max_seq };

    // Three tokens: write all layers at the cursor, then advance — once.
    for (std::size_t pos = 0; pos < 3; ++pos)
    {
        for (std::size_t l = 0; l < layers; ++l)
            write_tagged(c, l, 0, pos, dims);
        c.advance_cursor();
    }
    CHECK(c.cursor() == 3);

    // After 3 tokens, cursor is 3. Reading now would be for the *next* token
    // (pos 3), so keys return cursor+1 == 4 rows — but only 0..2 were written.
    // The realistic read happens at the token being processed; simulate that
    // by rewinding conceptually: here we assert the lockstep property instead.
    // Every layer reports the SAME row count, because they share one cursor.
    // (Write pos 3 across layers so the read is well-defined.)
    for (std::size_t l = 0; l < layers; ++l)
        write_tagged(c, l, 0, 3, dims);

    const auto rows_l0 = c.keys(0, 0).extent(0);
    const auto rows_l1 = c.keys(1, 0).extent(0);
    CHECK(rows_l0 == rows_l1);            // lockstep: same length everywhere
    CHECK(rows_l0 == c.cursor() + 1);     // cursor 3 -> 4 rows (0..=3)

    // And the historical rows are the values written at those positions.
    auto k = c.keys(0, 0);
    for (std::size_t pos = 0; pos <= 3; ++pos)
        for (std::size_t d = 0; d < dims; ++d)
            CHECK(k[pos, d] == codec<D>::expected(tag(0, 0, pos, d, false)));
}

TEMPLATE_TEST_CASE_SIG("kv_cache: reset_cursor restarts the sequence",
                       "[cache]", ((data_type D), D),
                       data_type::F32, data_type::BF16)
{
    constexpr std::size_t layers = 1, heads = 1, dims = 4, max_seq = 8;
    cache::kv_cache<D> c{ layers, heads, dims, max_seq };

    write_tagged(c, 0, 0, 0, dims);
    c.advance_cursor();
    write_tagged(c, 0, 0, 1, dims);
    c.advance_cursor();
    CHECK(c.cursor() == 2);

    c.reset_cursor();
    CHECK(c.cursor() == 0);

    // After reset, writing overwrites from position 0; reads are cursor-bounded
    // so stale rows beyond the cursor are never observed.
    write_tagged(c, 0, 0, 0, dims);   // new sequence, distinct-by-pos still 0
    auto k = c.keys(0, 0);
    CHECK(k.extent(0) == 1);          // just the one fresh row
    for (std::size_t d = 0; d < dims; ++d)
        CHECK(k[0, d] == codec<D>::expected(tag(0, 0, 0, d, false)));
}
