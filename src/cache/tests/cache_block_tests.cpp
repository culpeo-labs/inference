// cache_block tests — the isolated bookkeeping spec from docs/kv_cache.md.
// Pure storage/indexing, no attention math. Proving this green means that
// when the forward pass later disagrees with HF, the cache is not the cause.
//
// The block is generic over util::data_type. Storage units differ by dtype
// (bf16/f16 store uint16_t bit patterns; f32 stores float), but reads always
// come back as float through the accessor. Helpers below bridge that: encode
// a known float value into the block's storage type, and compare read-back
// floats against the expected encoded-then-decoded value.
#include <cstdint>
#include <vector>

#include <catch2/catch_all.hpp>

#include <cache/cache_block.h>
#include <util/types.h>

using namespace culpeo::inference;
using culpeo::inference::util::data_type;

namespace
{
    // Encode a float into the storage representation for dtype D, and report
    // the value it will read back as (bf16/f16 are lossy, so expected == the
    // round-tripped value, not the original). Values in the test are chosen
    // exact in bf16 so encode/decode is identity and comparisons are ==.
    template<data_type D>
    struct codec;

    template<>
    struct codec<data_type::F32>
    {
        using storage = float;
        static storage encode(float v) { return v; }
        static float decode(float readback) { return readback; }
    };

    template<>
    struct codec<data_type::BF16>
    {
        using storage = std::uint16_t;
        static storage encode(float v)
        {
            // truncate float32 -> bf16 (top 16 bits). Exact for our values.
            return static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(v) >> 16);
        }
        static float decode(float readback) { return readback; }
    };

    // Build a rank-1 write vector of the block's storage type holding `dims`
    // encoded values. Backing store must outlive the mdspan view passed to
    // write(), so the caller owns the vector.
    template<data_type D>
    auto make_vec(std::vector<typename codec<D>::storage>& backing)
    {
        using vt = typename util::matrix<D>::template type<1>;
        return vt{ backing.data(), backing.size() };
    }

    // Encode a recognizable value: head*100 + pos*10 + dim. Distinct per
    // (head,pos,dim) so a wrong head or position is unmistakable, and exact
    // in bf16 for small indices.
    constexpr float known(std::size_t head, std::size_t pos, std::size_t dim)
    {
        return static_cast<float>(head * 100 + pos * 10 + dim);
    }
}

TEMPLATE_TEST_CASE_SIG("cache_block: write then read one head, one position",
                       "[cache]", ((data_type D), D),
                       data_type::F32, data_type::BF16)
{
    constexpr std::size_t heads = 2, max_seq = 8, dims = 4;
    cache::cache_block<D> block{ max_seq, heads, dims };

    std::vector<typename codec<D>::storage> backing(dims);
    for (std::size_t d = 0; d < dims; ++d)
        backing[d] = codec<D>::encode(known(1, 0, d));   // head 1, pos 0
    block.write(1, 0, make_vec<D>(backing));

    auto view = block.read(1, 0);            // head 1, up to pos 0
    REQUIRE(view.extent(0) == 1);            // pos+1 == 1 row
    REQUIRE(view.extent(1) == dims);
    for (std::size_t d = 0; d < dims; ++d)
        CHECK(view[0, d] == codec<D>::decode(known(1, 0, d)));
}

TEMPLATE_TEST_CASE_SIG("cache_block: three positions read back in order",
                       "[cache]", ((data_type D), D),
                       data_type::F32, data_type::BF16)
{
    constexpr std::size_t heads = 2, max_seq = 8, dims = 4;
    cache::cache_block<D> block{ max_seq, heads, dims };

    // Write head 0, positions 0..2 with distinct known values.
    for (std::size_t p = 0; p < 3; ++p)
    {
        std::vector<typename codec<D>::storage> backing(dims);
        for (std::size_t d = 0; d < dims; ++d)
            backing[d] = codec<D>::encode(known(0, p, d));
        block.write(0, p, make_vec<D>(backing));
    }

    auto view = block.read(0, 2);            // up to pos 2 -> 3 rows
    REQUIRE(view.extent(0) == 3);            // pos+1 == 3 (the fencepost)
    for (std::size_t p = 0; p < 3; ++p)
        for (std::size_t d = 0; d < dims; ++d)
            CHECK(view[p, d] == codec<D>::decode(known(0, p, d)));
}

TEMPLATE_TEST_CASE_SIG("cache_block: heads are isolated",
                       "[cache]", ((data_type D), D),
                       data_type::F32, data_type::BF16)
{
    constexpr std::size_t heads = 3, max_seq = 8, dims = 4;
    cache::cache_block<D> block{ max_seq, heads, dims };

    // Write the SAME position across three different heads with head-tagged
    // values; reading one head must not see another's data.
    for (std::size_t h = 0; h < heads; ++h)
    {
        std::vector<typename codec<D>::storage> backing(dims);
        for (std::size_t d = 0; d < dims; ++d)
            backing[d] = codec<D>::encode(known(h, 0, d));
        block.write(h, 0, make_vec<D>(backing));
    }

    for (std::size_t h = 0; h < heads; ++h)
    {
        auto view = block.read(h, 0);
        for (std::size_t d = 0; d < dims; ++d)
            CHECK(view[0, d] == codec<D>::decode(known(h, 0, d)));
    }
}

TEMPLATE_TEST_CASE_SIG("cache_block: reading pos 0 yields exactly one row",
                       "[cache]", ((data_type D), D),
                       data_type::F32, data_type::BF16)
{
    // The fencepost that a `pos`-instead-of-`pos+1` bug breaks: the very
    // first token must be able to attend to itself (1 row, not 0).
    constexpr std::size_t heads = 1, max_seq = 4, dims = 2;
    cache::cache_block<D> block{ max_seq, heads, dims };

    std::vector<typename codec<D>::storage> backing(dims);
    for (std::size_t d = 0; d < dims; ++d)
        backing[d] = codec<D>::encode(known(0, 0, d));
    block.write(0, 0, make_vec<D>(backing));

    auto view = block.read(0, 0);
    CHECK(view.extent(0) == 1);
}
