#pragma once

#include <functional>

namespace culpeo::inference::util
{
    class scope_guard
    {
    public:
        template<typename F>
        scope_guard(F&& f) : m_func(std::forward<F>(f)) {}

        inline ~scope_guard()
        {
            m_func();
        }

        scope_guard(const scope_guard&) = delete;
        scope_guard& operator=(const scope_guard&) = delete;
        scope_guard(scope_guard&&) = delete;
        scope_guard& operator=(scope_guard&& other) = delete;
    private:
        std::function<void()> m_func;
    };
}