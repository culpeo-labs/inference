#pragma once

#include <bit>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <mdspan>
#include <span>
#include <type_traits>

namespace culpeo::inference::util {

    enum class data_type
    {
        BF16,
        F64,
        F32,
        F16,
        I64,
        I32,
        I16,
        I8,
        U8,
        BOOL,
    };

    template<typename M>
    concept float_matrix = (M::rank() == 2 && std::convertible_to<typename M::reference, float>);

    template<typename T, typename AccessorPolicy = std::default_accessor<T>>
    using mat_t = std::mdspan<T, std::dextents<std::size_t, 2>, std::layout_right, AccessorPolicy>;


    template<typename T, typename AccessorPolicy = std::default_accessor<const T>>
    using cmat_t = mat_t<const T, AccessorPolicy>;

    template<typename T>
        requires(std::floating_point<T>)
    using vec_t = std::span<T>;

    template<typename T>
        requires(std::floating_point<T>)
    using cvec_t = vec_t<const T>;


    template<data_type D>
    struct matrix
    {
        static_assert(std::false_type::value, "matrix<D> is not specialized for this data_type");
    };

    template<>
    struct matrix<data_type::F32>
    {
        using type = mat_t<float>;
        using const_type = cmat_t<float>;
    };

    template<>
    struct matrix<data_type::F64>
    {
        using type = mat_t<double>;
        using const_type = cmat_t<double>;
    };

    template<>
    struct matrix<data_type::F16>
    {
        struct accessor_policy
        {
            using element_type = std::uint16_t;
            using data_handle_type = std::uint16_t *;
            using reference = float;
            using offset_policy = accessor_policy;

            constexpr reference access(data_handle_type p, std::size_t i) const noexcept
            {
                std::uint16_t val = p[i];
                uint32_t sign = (val >> 15) & 0x1u;
                uint32_t exponent = (val >> 10) & 0x1Fu;
                uint32_t mantissa = val & 0x3FFu;
                uint32_t bits{};
                if (exponent == 0)
                {
                    if (mantissa == 0)
                    {
                        bits = sign << 31;
                    }
                    else
                    {
                        float v = std::ldexp(static_cast<float>(mantissa), -24);
                        return sign ? -v : v;
                    }
                }
                else if (exponent == 0x1F)
                {
                    bits = (sign << 31) | 0x7F800000u | (mantissa << 13);
                }
                else
                {
                    uint32_t exp = exponent - 15 + 127;
                    bits = (sign << 31) | (exp << 23) | (mantissa << 13);
                }
                return std::bit_cast<float>(bits);
            }

            constexpr data_handle_type offset(data_handle_type p, std::size_t i) const noexcept
            {
                return p + i;
            }
        };
        using type = mat_t<std::uint16_t, accessor_policy>;
        using const_type = cmat_t<std::uint16_t, accessor_policy>;
    };

    template<>
    struct matrix<data_type::BF16>
    {
        struct accessor_policy
        {
            using element_type = std::uint16_t;
            using data_handle_type = std::uint16_t *;
            using reference = float;
            using offset_policy = accessor_policy;

            constexpr reference access(data_handle_type p, std::size_t i) const noexcept
            {
                std::uint16_t val = p[i];
                return std::bit_cast<float>(static_cast<std::uint32_t>(val) << 16);
            }

            constexpr data_handle_type offset(data_handle_type p, std::size_t i) const noexcept
            {
                return p + i;
            }
        };

        using type = mat_t<std::uint16_t, accessor_policy>;
        using const_type = cmat_t<std::uint16_t, accessor_policy>;
    };

    template<>
    struct matrix<data_type::BOOL>
    {
        struct accessor_policy
        {
            using element_type = std::uint8_t;
            using data_handle_type = std::uint8_t *;
            using reference = bool;
            using offset_policy = accessor_policy;

            constexpr reference access(data_handle_type p, std::size_t i) const noexcept
            {
                return p[i] != 0;
            }

            constexpr data_handle_type offset(data_handle_type p, std::size_t i) const noexcept
            {
                return p + i;
            }
        };

        using type = mat_t<std::uint8_t, accessor_policy>;
        using const_type = cmat_t<std::uint8_t, accessor_policy>;
    };

    template<>
    struct matrix<data_type::I8>
    {
        using type = mat_t<std::int8_t>;
        using const_type = cmat_t<std::int8_t>;
    };

    template<>
    struct matrix<data_type::U8>
    {
        using type = mat_t<std::uint8_t>;
        using const_type = cmat_t<std::uint8_t>;
    };

    template<>
    struct matrix<data_type::I16>
    {
        using type = mat_t<std::int16_t>;
        using const_type = cmat_t<std::int16_t>;
    };

    template<>
    struct matrix<data_type::I32>
    {
        using type = mat_t<std::int32_t>;
        using const_type = cmat_t<std::int32_t>;
    };

    template<>
    struct matrix<data_type::I64>
    {
        using type = mat_t<std::int64_t>;
        using const_type = cmat_t<std::int64_t>;
    };
}