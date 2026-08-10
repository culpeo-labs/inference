#pragma once

#include <bit>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <mdspan>
#include <type_traits>

#include <util/traits.h>

namespace culpeo::inference::util
{
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

    namespace details
    {
        template<data_type D>
        struct data_type_storage_type;

        template<> struct data_type_storage_type<data_type::BF16>: std::type_identity<std::uint16_t> {};
        template<> struct data_type_storage_type<data_type::F64>: std::type_identity<double> {};
        template<> struct data_type_storage_type<data_type::F32>: std::type_identity<float> {};
        template<> struct data_type_storage_type<data_type::F16>: std::type_identity<std::uint16_t> {};
        template<> struct data_type_storage_type<data_type::I64>: std::type_identity<std::int64_t> {};
        template<> struct data_type_storage_type<data_type::I32>: std::type_identity<std::int32_t> {};
        template<> struct data_type_storage_type<data_type::I16>: std::type_identity<std::int16_t> {};
        template<> struct data_type_storage_type<data_type::I8>: std::type_identity<std::int8_t> {};
        template<> struct data_type_storage_type<data_type::U8>: std::type_identity<std::uint8_t> {};
        template<> struct data_type_storage_type<data_type::BOOL>: std::type_identity<std::uint8_t> {};
    }

    template<typename M>
    concept float_matrix = is_mdspan_v<M> && (M::rank() == 2 && std::convertible_to<typename M::reference, float>);

    template<typename M>
    concept float_vector = is_mdspan_v<M> && (M::rank() == 1 && std::convertible_to<typename M::reference, float>);

    template<typename T, size_t Rank, typename AccessorPolicy = std::default_accessor<T>>
    using mat_t = std::mdspan<T, std::dextents<std::size_t, Rank>, std::layout_right, AccessorPolicy>;

    template<data_type D>
    struct matrix
    {
        static_assert(std::false_type::value, "matrix<D> is not specialized for this data_type");
    };

    template<data_type D>
        requires (
            D == data_type::F64 ||
            D == data_type::F32 ||
            D == data_type::I8 ||
            D == data_type::U8 ||
            D == data_type::I16 ||
            D == data_type::I32 ||
            D == data_type::I64)
    struct matrix<D>
    {
        template<std::size_t Rank, bool Const>
        using type = mat_t<conditionally_const_t<typename details::data_type_storage_type<D>::type, Const>, Rank>;

        template<std::size_t Rank>
        using view= type<Rank, true>;

        template<std::size_t Rank>
        using mut= type<Rank, false>;
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
        template<std::size_t Rank, bool Const>
        using type = mat_t<conditionally_const_t<std::uint16_t, Const>, Rank, accessor_policy<Const>>;

        template<std::size_t Rank>
        using view= type<Rank, true>;

        template<std::size_t Rank>
        using mut= type<Rank, false>;
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

        template<std::size_t Rank, bool Const>
        using type = mat_t<conditionally_const_t<std::uint16_t, Const>, Rank, accessor_policy<Const>>;

        template<std::size_t Rank>
        using view= type<Rank, true>;

        template<std::size_t Rank>
        using mut= type<Rank, false>;
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

        template<std::size_t Rank, bool Const>
        using type = mat_t<conditionally_const_t<std::uint8_t, Const>, Rank, accessor_policy<Const>>;

        template<std::size_t Rank>
        using view= type<Rank, true>;

        template<std::size_t Rank>
        using mut= type<Rank, false>;
    };

    template<typename T>
    concept bool_matrix = mdspan_of<T, std::uint8_t> &&
    (
        std::same_as<typename T::accessor_type, matrix<data_type::BOOL>::accessor_policy<true>> ||
        std::same_as<typename T::accessor_type, matrix<data_type::BOOL>::accessor_policy<false>>
    );


    template<typename T>
    concept f16_matrix = mdspan_of<T, std::uint16_t> &&
    (
        std::same_as<typename T::accessor_type, matrix<data_type::F16>::accessor_policy<true>> ||
        std::same_as<typename T::accessor_type, matrix<data_type::F16>::accessor_policy<false>>
    );

    template<typename T>
    concept bf16_matrix = mdspan_of<T, std::uint16_t> &&
    (
        std::same_as<typename T::accessor_type, matrix<data_type::BF16>::accessor_policy<true>> ||
        std::same_as<typename T::accessor_type, matrix<data_type::BF16>::accessor_policy<false>>
    );

    template<typename T>
    concept f32_matrix = mdspan_of<T, float>;

    template<typename T>
    concept f64_matrix = mdspan_of<T, double>;

    template<typename T>
    concept i8_matrix = mdspan_of<T, std::int8_t>;

    template<typename T>
    concept u8_matrix = mdspan_of<T, std::uint8_t> && !bool_matrix<T>;

    template<typename T>
    concept i16_matrix = mdspan_of<T, std::int16_t>;

    template<typename T>
    concept i32_matrix = mdspan_of<T, std::int32_t>;

    template<typename T>
    concept i64_matrix = mdspan_of<T, std::int64_t>;
}