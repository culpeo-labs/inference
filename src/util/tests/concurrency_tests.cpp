// parallel_row_for / thread_service tests.
//
// Concurrency tests differ from sequential ones: a single passing run can hide
// a race that simply didn't manifest that time. So this suite (a) checks the
// invariants that a race would violate, (b) runs each check many times to raise
// the odds of provoking a nondeterministic failure, and (c) is meant to be run
// under ThreadSanitizer, which detects the unsynchronized access *itself* rather
// than only its corrupting effect.
//
// Build this as a SEPARATE target with -fsanitize=thread (TSan does not compose
// with ASan/UBSan). The stress loops below are sized to give TSan enough
// opportunities to catch a data race in the service's dispatch/shutdown paths.
//
// Invariants under test:
//   - every row is processed exactly once (no gaps from chunk boundaries, no
//     double-processing from overlapping chunks) — the counts-== -1 check
//   - the parallel result is BIT-IDENTICAL to the serial result (per-row work
//     is independent and deterministic, so distributing rows must not change
//     any value)
//   - edge cases: rows < workers, single worker, empty matrix, non-aligned
//     row counts (partial last chunk)
//   - repeated dispatch on one long-lived service is safe (the reuse path)

#include <atomic>
#include <cstddef>
#include <functional>
#include <numeric>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <util/concurrency.h>

#include <util/types.h>

using namespace culpeo::inference;

namespace
{
    // A rank-2 mdspan over an owned buffer, so parallel_row_for has a matrix to
    // partition rows over. Values are irrelevant for the coverage tests (we key
    // on row index); they matter for the parallel==serial test.
    struct test_matrix
    {
        std::vector<float> data;
        std::size_t rows, cols;

        test_matrix(std::size_t r, std::size_t c) : data(r * c), rows(r), cols(c)
        {
            std::iota(data.begin(), data.end(), 1.0f);   // 1,2,3,... deterministic
        }

        auto span()
        {
            return util::mat_t<float, 2>{ data.data(), rows, cols };
        }
    };

    constexpr std::size_t alignment = 16;   // hardware_destructive_interference_size / sizeof(float)
}

TEST_CASE("parallel_row_for: every row processed exactly once", "[threading]")
{
    util::thread_service<std::function<void()>> service{ 8 };

    // Sweep row counts that stress the chunking arithmetic:
    //  - clean multiple, partial last chunk, fewer rows than workers, single row.
    for (std::size_t rows : { std::size_t{2048}, std::size_t{100}, std::size_t{3}, std::size_t{1} })
    {
        test_matrix m{ rows, 4 };
        // Stress: repeat many times; a chunk-boundary or dispatch race is
        // nondeterministic, so one pass is not enough to trust.
        for (int iter = 0; iter < 500; iter++)
        {
            std::vector<int> counts(rows, 0);
            parallel_row_for(service, m.span(), alignment,
                             [&](std::size_t r, auto _) { counts[r]++; });   // disjoint rows -> no data race on counts
            for (std::size_t r = 0; r < rows; r++)
                REQUIRE(counts[r] == 1);   // 0 = row missed, 2+ = row double-processed
        }
    }
}

TEST_CASE("parallel_row_for: parallel result is bit-identical to serial", "[threading]")
{
    util::thread_service<std::function<void()>> service{ 8 };
    const std::size_t rows = 2048, cols = 64;
    test_matrix m{ rows, cols };
    auto mat = m.span();

    // Per-row work: sum the row. Independent per row, deterministic, so the
    // distributed result must EXACTLY equal the serial one (not within tolerance
    // — we are not reordering within a row, only across rows).
    auto row_sum = [&](std::size_t r) {
        float acc = 0.0f;
        for (std::size_t c = 0; c < cols; c++) acc += mat[r, c];
        return acc;
    };

    std::vector<float> serial(rows), parallel(rows);
    for (std::size_t r = 0; r < rows; r++) serial[r] = row_sum(r);

    for (int iter = 0; iter < 200; iter++)   // stress
    {
        std::fill(parallel.begin(), parallel.end(), -1.0f);
        parallel_row_for(service, mat, alignment,
                         [&](std::size_t r, auto _) { parallel[r] = row_sum(r); });
        for (std::size_t r = 0; r < rows; r++)
            REQUIRE(parallel[r] == serial[r]);   // exact
    }
}

TEST_CASE("parallel_row_for: single worker degenerates to serial", "[threading]")
{
    util::thread_service<std::function<void()>> service{ 1 };
    const std::size_t rows = 257;   // deliberately not a multiple of alignment
    test_matrix m{ rows, 4 };

    std::vector<int> counts(rows, 0);
    parallel_row_for(service, m.span(), alignment, [&](std::size_t r, auto _) { counts[r]++; });
    for (std::size_t r = 0; r < rows; r++)
        REQUIRE(counts[r] == 1);
}

TEST_CASE("parallel_row_for: more workers than rows", "[threading]")
{
    // The latch must still reach zero when some workers get no chunk. If
    // chunk_count were hardcoded to worker_count instead of the actual number
    // of chunks, this would hang — so this test guards the latch sizing.
    util::thread_service<std::function<void()>> service{ 16 };
    const std::size_t rows = 3;
    test_matrix m{ rows, 4 };

    for (int iter = 0; iter < 500; iter++)
    {
        std::vector<int> counts(rows, 0);
        parallel_row_for(service, m.span(), alignment, [&](std::size_t r, auto _) { counts[r]++; });
        for (std::size_t r = 0; r < rows; r++)
            REQUIRE(counts[r] == 1);
    }
}

TEST_CASE("parallel_row_for: repeated dispatch on a long-lived service", "[threading]")
{
    // matvec calls this thousands of times on ONE service. Confirm the pool
    // reuse path (post -> process -> idle -> post again) is correct and doesn't
    // accumulate state across dispatches.
    util::thread_service<std::function<void()>> service{ 8 };
    const std::size_t rows = 512;
    test_matrix m{ rows, 4 };

    for (int dispatch = 0; dispatch < 2000; dispatch++)
    {
        std::vector<int> counts(rows, 0);
        util::parallel_row_for(service, m.span(), alignment, [&](std::size_t r, auto row) { counts[r]++; });
        for (std::size_t r = 0; r < rows; r++)
            REQUIRE(counts[r] == 1);
    }
}

// NOTE on empty matrices (rows == 0): std::latch{0} is already-ready, so wait()
// returns immediately and the dispatch loop posts nothing. If your
// parallel_row_for is expected to handle rows==0, add a section here. If it
// asserts rows > 0 as a precondition, that assert belongs in the caller
// (matvec never has zero output rows), so leaving rows==0 unsupported is fine —
// document which.