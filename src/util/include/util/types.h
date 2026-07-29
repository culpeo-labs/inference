#pragma once

#include <bit>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <mdspan>
#include <span>
#include <type_traits>

namespace culpeo::inference::util {

    template<typename T, bool C>
        requires(!std::is_pointer_v<T>)
    using conditionally_const_t = std::conditional_t<C, std::add_const_t<T>, T>;

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

    template<typename M>
    concept float_vector = (M::rank() == 1 && std::convertible_to<typename M::reference, float>);

    template<typename T, size_t Rank, typename AccessorPolicy = std::default_accessor<T>>
    using mat_t = std::mdspan<T, std::dextents<std::size_t, Rank>, std::layout_right, AccessorPolicy>;

    template<typename T, size_t Rank, typename AccessorPolicy = std::default_accessor<const T>>
    using cmat_t = mat_t<const T, Rank, AccessorPolicy>;

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
        template<std::size_t Rank>
        using type = mat_t<float, Rank>;
        template<std::size_t Rank>
        using const_type = cmat_t<float, Rank>;
    };

    template<>
    struct matrix<data_type::F64>
    {
        template<std::size_t Rank>
        using type = mat_t<double, Rank>;
        template<std::size_t Rank>
        using const_type = cmat_t<double, Rank>;
    };

    template<>
    struct matrix<data_type::F16>
    {
        template<bool C = false>
        struct accessor_policy
        {
            using element_type = conditionally_const_t<std::uint16_t, C>;
            using data_handle_type = std::add_pointer_t<element_type>;
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
        template<std::size_t Rank>
        using type = mat_t<std::uint16_t, Rank, accessor_policy<>>;
        template<std::size_t Rank>
        using const_type = cmat_t<std::uint16_t, Rank, accessor_policy<true>>;
    };

    template<>
    struct matrix<data_type::BF16>
    {
        template<bool C = false>
        struct accessor_policy
        {
            using element_type = conditionally_const_t<std::uint16_t, C>;
            using data_handle_type = std::add_pointer_t<element_type>;
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

        template<std::size_t Rank>
        using type = mat_t<std::uint16_t, Rank, accessor_policy<>>;
        template<std::size_t Rank>
        using const_type = cmat_t<std::uint16_t, Rank,  accessor_policy<true>>;
    };

    template<>
    struct matrix<data_type::BOOL>
    {
        template<bool C = false>
        struct accessor_policy
        {
            using element_type = conditionally_const_t<std::uint8_t, C>;
            using data_handle_type = std::add_pointer_t<element_type>;
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

        template<std::size_t Rank>
        using type = mat_t<std::uint8_t, Rank, accessor_policy<>>;
        template<std::size_t Rank>
        using const_type = cmat_t<std::uint8_t, Rank, accessor_policy<true>>;
    };

    template<>
    struct matrix<data_type::I8>
    {
        template<std::size_t Rank>
        using type = mat_t<std::int8_t, Rank>;
        template<std::size_t Rank>
        using const_type = cmat_t<std::int8_t, Rank>;
    };

    template<>
    struct matrix<data_type::U8>
    {
        template<std::size_t Rank>
        using type = mat_t<std::uint8_t, Rank>;
        template<std::size_t Rank>
        using const_type = cmat_t<std::uint8_t, Rank>;
    };

    template<>
    struct matrix<data_type::I16>
    {
        template<std::size_t Rank>
        using type = mat_t<std::int16_t, Rank>;
        template<std::size_t Rank>
        using const_type = cmat_t<std::int16_t, Rank>;
    };

    template<>
    struct matrix<data_type::I32>
    {
        template<std::size_t Rank>
        using type = mat_t<std::int32_t, Rank>;
        template<std::size_t Rank>
        using const_type = cmat_t<std::int32_t, Rank>;
    };

    template<>
    struct matrix<data_type::I64>
    {
        template<std::size_t Rank>
        using type = mat_t<std::int64_t, Rank>;
        template<std::size_t Rank>
        using const_type = cmat_t<std::int64_t, Rank>;
    };
}