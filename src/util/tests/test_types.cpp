#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <type_traits>

#include <catch2/catch_test_macros.hpp>
#include <util/types.h>

using namespace culpeo::inference::util;

namespace {

    using f16_acc = matrix<data_type::F16>::accessor_policy<>;
    using bf16_acc = matrix<data_type::BF16>::accessor_policy<>;
    using bool_acc = matrix<data_type::BOOL>::accessor_policy<>;

    [[nodiscard]] bool bit_eq(float a, float b) noexcept
    {
        return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b);
    }

    [[nodiscard]] float f16(std::uint16_t bits) noexcept
    {
        return f16_acc{}.access(&bits, 0);
    }

    [[nodiscard]] float bf16(std::uint16_t bits) noexcept
    {
        return bf16_acc{}.access(&bits, 0);
    }

}

TEST_CASE("F16: normal values", "[f16]")
{
    REQUIRE(bit_eq(f16(0x3C00), 1.0f));
    REQUIRE(bit_eq(f16(0xBC00), -1.0f));
    REQUIRE(bit_eq(f16(0x4000), 2.0f));
    REQUIRE(bit_eq(f16(0xC000), -2.0f));
    REQUIRE(bit_eq(f16(0x3800), 0.5f));
    REQUIRE(bit_eq(f16(0x7BFF), 65504.0f)); // largest normal
    REQUIRE(bit_eq(f16(0x0400), 0x1p-14f)); // smallest normal, 2^-14
}

TEST_CASE("F16: signed zero", "[f16]")
{
    REQUIRE(bit_eq(f16(0x0000), 0.0f));
    REQUIRE(bit_eq(f16(0x8000), -0.0f)); // sign bit must survive
    REQUIRE_FALSE(bit_eq(f16(0x8000), 0.0f));
}

TEST_CASE("F16: subnormals", "[f16]")
{
    REQUIRE(bit_eq(f16(0x0001), 0x1p-24f));   // smallest positive subnormal
    REQUIRE(bit_eq(f16(0x03FF), 0x3FFp-24f)); // largest subnormal, 1023 * 2^-24
    REQUIRE(bit_eq(f16(0x8001), -0x1p-24f));  // negative subnormal
}

TEST_CASE("F16: infinities", "[f16]")
{
    REQUIRE(std::isinf(f16(0x7C00)));
    REQUIRE(f16(0x7C00) > 0.0f);
    REQUIRE(std::isinf(f16(0xFC00)));
    REQUIRE(f16(0xFC00) < 0.0f);
}

TEST_CASE("F16: NaN stays NaN", "[f16]")
{
    // Regression guard: the exponent==0x1F branch must keep the mantissa,
    // otherwise every NaN collapses to inf.
    REQUIRE(std::isnan(f16(0x7E00))); // quiet NaN
    REQUIRE(std::isnan(f16(0xFE00)));
    REQUIRE(std::isnan(f16(0x7C01))); // NaN with minimal mantissa
    REQUIRE(std::isnan(f16(0xFFFF)));
}

// ---------------------------------------------------------------------------
// BF16 accessor
// ---------------------------------------------------------------------------

TEST_CASE("BF16: normal values", "[bf16]")
{
    REQUIRE(bit_eq(bf16(0x3F80), 1.0f));
    REQUIRE(bit_eq(bf16(0xBF80), -1.0f));
    REQUIRE(bit_eq(bf16(0x4000), 2.0f));
    REQUIRE(bit_eq(bf16(0x3F00), 0.5f));
}

TEST_CASE("BF16: signed zero", "[bf16]")
{
    REQUIRE(bit_eq(bf16(0x0000), 0.0f));
    REQUIRE(bit_eq(bf16(0x8000), -0.0f));
    REQUIRE_FALSE(bit_eq(bf16(0x8000), 0.0f));
}

TEST_CASE("BF16: infinities and NaN", "[bf16]")
{
    REQUIRE(std::isinf(bf16(0x7F80)));
    REQUIRE(bf16(0x7F80) > 0.0f);
    REQUIRE(std::isinf(bf16(0xFF80)));
    REQUIRE(std::isnan(bf16(0x7FC0)));
    REQUIRE(std::isnan(bf16(0x7F81))); // NaN with minimal mantissa
}

TEST_CASE("BF16: widening is exact for arbitrary floats", "[bf16]")
{
    // bf16 is literally the top 16 bits of f32, so widening the truncated
    // pattern must reproduce that truncated float bit-for-bit.
    for (float orig : {3.14159f, -2.71828f, 1e10f, 1e-10f, 123456.0f, -0.03125f})
    {
        auto u = std::bit_cast<std::uint32_t>(orig);
        auto hi = static_cast<std::uint16_t>(u >> 16);
        REQUIRE(bit_eq(bf16(hi), std::bit_cast<float>(u & 0xFFFF0000u)));
    }
}

TEST_CASE("BF16: subnormal", "[bf16]")
{
    REQUIRE(bit_eq(bf16(0x0001), std::bit_cast<float>(std::uint32_t{0x00010000})));
}

// ---------------------------------------------------------------------------
// BOOL accessor
// ---------------------------------------------------------------------------

TEST_CASE("BOOL: zero is false, nonzero is true", "[bool]")
{
    bool_acc a;
    std::uint8_t zero = 0, one = 1, high = 255;
    REQUIRE(a.access(&zero, 0) == false);
    REQUIRE(a.access(&one, 0) == true);
    REQUIRE(a.access(&high, 0) == true);
}

// ---------------------------------------------------------------------------
// Accessor offset semantics
// ---------------------------------------------------------------------------

TEST_CASE("offset() satisfies the accessor invariant", "[accessor][offset]")
{
    // access(offset(p, i), j) == access(p, i + j)
    std::array<std::uint16_t, 4> d{0x3C00, 0x4000, 0x4200, 0x4400}; // 1, 2, 3, 4
    f16_acc a;

    REQUIRE(bit_eq(a.access(a.offset(d.data(), 1), 1), a.access(d.data(), 2)));
    REQUIRE(bit_eq(a.access(a.offset(d.data(), 2), 0), a.access(d.data(), 2)));
    REQUIRE(bit_eq(a.access(a.offset(d.data(), 0), 3), a.access(d.data(), 3)));
}

TEST_CASE("offset_policy is the accessor itself", "[accessor][traits]")
{
    STATIC_REQUIRE(std::is_same_v<f16_acc::offset_policy, f16_acc>);
    STATIC_REQUIRE(std::is_same_v<bf16_acc::offset_policy, bf16_acc>);
    STATIC_REQUIRE(std::is_same_v<bool_acc::offset_policy, bool_acc>);

    STATIC_REQUIRE(std::is_same_v<f16_acc::element_type, std::uint16_t>);
    STATIC_REQUIRE(std::is_same_v<f16_acc::reference, float>);
    STATIC_REQUIRE(std::is_same_v<bf16_acc::reference, float>);
    STATIC_REQUIRE(std::is_same_v<bool_acc::reference, bool>);
}

// ---------------------------------------------------------------------------
// mdspan integration (matrix<D>::type)
// ---------------------------------------------------------------------------

TEST_CASE("F16 matrix: 2D indexing through mdspan", "[f16][mdspan]")
{
    using MatF16 = matrix<data_type::F16>::mut<2>;

    // 2x3, layout_right:  [ 1 2 3 ]
    //                     [ 4 5 6 ]
    std::array<std::uint16_t, 6> buf{
        0x3C00, 0x4000, 0x4200, // 1, 2, 3
        0x4400, 0x4500, 0x4600  // 4, 5, 6
    };

    MatF16 m(buf.data(), 2, 3);

    REQUIRE(m.rank() == 2);
    REQUIRE(m.extent(0) == 2);
    REQUIRE(m.extent(1) == 3);

    REQUIRE(bit_eq(m[0, 0], 1.0f));
    REQUIRE(bit_eq(m[0, 2], 3.0f));
    REQUIRE(bit_eq(m[1, 0], 4.0f));
    REQUIRE(bit_eq(m[1, 2], 6.0f));
}

TEST_CASE("BF16 matrix: 2D indexing through mdspan", "[bf16][mdspan]")
{
    using MatBF16 = matrix<data_type::BF16>::mut<2>;

    std::array<std::uint16_t, 4> buf{
        0x3F80, 0x4000, // 1, 2
        0x4040, 0x4080  // 3, 4
    };

    MatBF16 m(buf.data(), 2, 2);

    REQUIRE(bit_eq(m[0, 0], 1.0f));
    REQUIRE(bit_eq(m[0, 1], 2.0f));
    REQUIRE(bit_eq(m[1, 0], 3.0f));
    REQUIRE(bit_eq(m[1, 1], 4.0f));
}

// ---------------------------------------------------------------------------
// Type aliases and the float_matrix concept
// ---------------------------------------------------------------------------

TEST_CASE("type aliases expose the expected reference type", "[traits]")
{
    STATIC_REQUIRE(std::is_same_v<matrix<data_type::F16>::mut<2>::reference, float>);
    STATIC_REQUIRE(std::is_same_v<matrix<data_type::BF16>::mut<2>::reference, float>);
    STATIC_REQUIRE(std::is_same_v<matrix<data_type::F32>::mut<2>::reference, float &>);

    // F32/F64 const_type is well-formed (default_accessor<const T>).
    STATIC_REQUIRE(std::is_same_v<matrix<data_type::F32>::view<2>::element_type, const float>);
    STATIC_REQUIRE(std::is_same_v<matrix<data_type::F64>::view<2>::element_type, const double>);
}

TEST_CASE("float_matrix concept is satisfied", "[concept]")
{
    STATIC_REQUIRE(float_matrix<matrix<data_type::F32>::mut<2>>);
    STATIC_REQUIRE(float_matrix<matrix<data_type::F64>::mut<2>>);
    STATIC_REQUIRE(float_matrix<matrix<data_type::F16>::mut<2>>);
    STATIC_REQUIRE(float_matrix<matrix<data_type::BF16>::mut<2>>);
}

// ---------------------------------------------------------------------------
// const_type — read-only views
// ---------------------------------------------------------------------------
//
// These assume the converting/bool accessors have been made const-correct so
// that matrix<D>::const_type is well-formed:
//
//     element_type     = const <storage>     (const uint16_t / const uint8_t)
//     data_handle_type = const <storage>*
//     reference        = float | bool        (unchanged; access is by value)
//
// The element_type assertions below aren't just documentation: mdspan Mandates
// element_type == accessor::element_type, so the only way const_type compiles
// at all is if these hold. They pin the contract the fix has to meet.

TEST_CASE("const_type: element_type is const-qualified", "[const][traits]")
{
    STATIC_REQUIRE(std::is_same_v<matrix<data_type::F16>::view<2>::element_type, const std::uint16_t>);
    STATIC_REQUIRE(std::is_same_v<matrix<data_type::BF16>::view<2>::element_type, const std::uint16_t>);
    STATIC_REQUIRE(std::is_same_v<matrix<data_type::BOOL>::view<2>::element_type, const std::uint8_t>);
    STATIC_REQUIRE(std::is_same_v<matrix<data_type::F32>::view<2>::element_type, const float>);
    STATIC_REQUIRE(std::is_same_v<matrix<data_type::F64>::view<2>::element_type, const double>);

    STATIC_REQUIRE(std::is_same_v<matrix<data_type::F16>::view<2>::reference, float>);
    STATIC_REQUIRE(std::is_same_v<matrix<data_type::BF16>::view<2>::reference, float>);
    STATIC_REQUIRE(std::is_same_v<matrix<data_type::BOOL>::view<2>::reference, bool>);
}

TEST_CASE("F16 const_type: reads through a const view", "[const][f16][mdspan]")
{
    using CMat = matrix<data_type::F16>::view<2>;

    const std::array<std::uint16_t, 6> buf{
        0x3C00, 0x4000, 0x4200, // 1, 2, 3
        0x4400, 0x4500, 0x4600  // 4, 5, 6
    };

    CMat m(buf.data(), 2, 3);

    REQUIRE(m.extent(0) == 2);
    REQUIRE(m.extent(1) == 3);
    REQUIRE(bit_eq(m[0, 0], 1.0f));
    REQUIRE(bit_eq(m[0, 2], 3.0f));
    REQUIRE(bit_eq(m[1, 0], 4.0f));
    REQUIRE(bit_eq(m[1, 2], 6.0f));
}

TEST_CASE("BF16 const_type: reads through a const view", "[const][bf16][mdspan]")
{
    using CMat = matrix<data_type::BF16>::view<2>;

    const std::array<std::uint16_t, 4> buf{
        0x3F80, 0x4000, // 1, 2
        0x4040, 0x4080  // 3, 4
    };

    CMat m(buf.data(), 2, 2);

    REQUIRE(bit_eq(m[0, 0], 1.0f));
    REQUIRE(bit_eq(m[0, 1], 2.0f));
    REQUIRE(bit_eq(m[1, 0], 3.0f));
    REQUIRE(bit_eq(m[1, 1], 4.0f));
}

TEST_CASE("BOOL const_type: reads through a const view", "[const][bool][mdspan]")
{
    using CMat = matrix<data_type::BOOL>::view<2>;

    const std::array<std::uint8_t, 4> buf{0, 1, 2, 0};
    CMat m(buf.data(), 2, 2);

    // Extra parens guard the multidim subscript's comma from the REQUIRE macro.
    REQUIRE_FALSE((m[0, 0]));
    REQUIRE((m[0, 1]));
    REQUIRE((m[1, 0]));
    REQUIRE_FALSE((m[1, 1]));
}

TEST_CASE("F32 const_type: reads through a const view", "[const][f32][mdspan]")
{
    using CMat = matrix<data_type::F32>::view<2>;

    const std::array<float, 4> buf{1.5f, 2.5f, 3.5f, 4.5f};
    CMat m(buf.data(), 2, 2);

    REQUIRE(bit_eq(m[0, 0], 1.5f));
    REQUIRE(bit_eq(m[1, 1], 4.5f));
}

TEST_CASE("const_type accessor: offset invariant", "[const][accessor][offset]")
{
    using CAcc = matrix<data_type::F16>::view<2>::accessor_type;
    STATIC_REQUIRE(std::is_same_v<CAcc::offset_policy, CAcc>);

    const std::array<std::uint16_t, 4> d{0x3C00, 0x4000, 0x4200, 0x4400}; // 1, 2, 3, 4
    CAcc a;

    // access(offset(p, i), j) == access(p, i + j)
    REQUIRE(bit_eq(a.access(a.offset(d.data(), 1), 1), a.access(d.data(), 2)));
    REQUIRE(bit_eq(a.access(a.offset(d.data(), 2), 0), a.access(d.data(), 2)));
}

TEST_CASE("const_type satisfies float_matrix", "[const][concept]")
{
    STATIC_REQUIRE(float_matrix<matrix<data_type::F16>::view<2>>);
    STATIC_REQUIRE(float_matrix<matrix<data_type::BF16>::view<2>>);
    STATIC_REQUIRE(float_matrix<matrix<data_type::F32>::view<2>>);
    STATIC_REQUIRE(float_matrix<matrix<data_type::F64>::view<2>>);
}