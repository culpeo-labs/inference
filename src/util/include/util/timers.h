#pragma once

#include <algorithm>
#include <chrono>
#include <iostream>
#include <print>
#include <ranges>
#include <string_view>
#include <source_location>
#include <vector>

#include <util/scope_guard.h>

#ifdef ENABLE_TIMERS
    #define ENABLE_TIMERS 1
#else
    #define ENABLE_TIMERS 0
#endif

namespace culpeo::inference::util
{
    namespace details
    {
        template<bool Enabled>
        struct scope_timer
        {
            scope_timer(std::string_view)
            {
            }

            int probe()
            {
                return 0;;
            }
        };

        template<>
        struct scope_timer<true>
        {
            scope_timer(std::string_view tag): m_tag(tag)
            {
            }

            ~scope_timer()
            {
                if (!m_durations.empty())
                {
                    std::ranges::sort(m_durations);
                    auto min = m_durations.front();
                    auto max = m_durations.back();
                    auto sum = std::ranges::fold_left(m_durations | std::ranges::views::transform([](auto d) { return d.count(); }), 0, [](auto a, auto b) { return a + b; });
                    auto median = m_durations[m_durations.size() / 2];
                    auto avg = sum / m_durations.size();
                    std::print(std::cerr, "{} count: {}, sum: {} [ns], min: {} [ns], max: {} [ns], median: {} [ns], avg: {} [ns]\n", m_tag, m_durations.size(), sum, min, max, median, avg);
                }
                else
                {
                    std::print(std::cerr, "{} count: 0\n", m_tag);
                }
            }

            [[nodiscard]] scope_guard probe()
            {
                const auto start = std::chrono::steady_clock::now();
                return scope_guard{
                    [this, start = std::move(start)]
                    {
                        auto end = std::chrono::steady_clock::now();
                        m_durations.push_back((end - start));
                    }
                };
            }

        private:
            std::string_view m_tag;
            std::vector<std::chrono::nanoseconds> m_durations{};
        };
    }

    using scope_timer = details::scope_timer<ENABLE_TIMERS>;

    struct function_timer: private scope_timer
    {
        function_timer(std::source_location location): scope_timer(location.function_name())
        {
        }

        using scope_timer::probe;
    };
}