#pragma once

#include "util/mdarray.h"
#include "util/traits.h"
#include "util/types.h"
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <iterator>
#include <latch>
#include <mutex>
#include <ranges>
#include <thread>
#include <utility>
#include <vector>

namespace culpeo::inference::util
{
    template<typename TaskType>
    class thread_service
    {
    public:
        thread_service(std::size_t worker_count)
        {
            m_threads.reserve(worker_count);
            for (std::size_t i{0}; i < worker_count; i++)
            {
                m_threads.emplace_back(&thread_service::do_work, this);
            }

        }

        ~thread_service()
        {
            std::lock_guard lk{ m_cv };
            m_stopped = true;
            m_cv.notify_all();
        }

        void post(TaskType task)
        {
            std::lock_guard lk{ m_mutex };
            m_tasks.push_back(std::move(task));
            m_cv.notify_one();
        }

        std::size_t worker_count() const
        {
            return m_threads.size();
        }

    private:
        void do_work()
        {
            while(!m_stopped)
            {
                std::unique_lock lk{ m_mutex };
                m_cv.wait(lk, [&]{ return !m_tasks.empty() || m_stopped; });
                if (m_tasks.empty())
                {
                    continue;
                }
                auto task = std::move(m_tasks.front());
                m_tasks.pop_front();
                lk.release();
                task();
            }
        }

        std::mutex m_mutex{};
        std::condition_variable m_cv{};
        bool m_stopped{};
        std::vector<std::jthread> m_threads{};
        std::deque<TaskType> m_tasks{};
    };

    template<typename F>
    void parallel_row_for(thread_service<std::function<void()>> & service, float_matrix auto mat, std::size_t task_alignment, F&& task)
    {
        const auto worker_count = service.worker_count();
        const auto row_count = mat.extent(0);
        const auto base = (row_count + worker_count - 1) / worker_count;
        const auto chunk_size = ((base + task_alignment - 1) / task_alignment) * task_alignment;
        const auto chunks = std::ranges::views::iota(0, mat.extent(0)) | std::ranges::views::chunk(chunk_size);
        const auto chunk_count = std::ranges::distance(chunks);
        std::latch done{ chunk_count };
        for (auto chunk : chunks)
        {
            service.post([chunk, &mat, &task, &done]
            {
                for (auto row : chunk)
                {
                    auto v = get_row(mat, row);
                    task(v);
                }
                done.count_down();
            });
            done.wait();
        }
    }

}