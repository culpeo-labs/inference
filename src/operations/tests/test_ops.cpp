#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cmath>
#include <mdspan>
#include <vector>

#include <util/concurrency.h>
#include <util/mdarray.h>
#include <util/types.h>
#include <operations/ops.h>

#include "golden.h"

using namespace culpeo::inference;

// Tolerance per docs/ops.md: abs 1e-5 + rel 1e-4 against float64-derived
// goldens. CHECK (not REQUIRE) so one bad element doesn't hide the rest.
template<size_t N>
static void check_close(const std::array<float, N> got, const std::array<float, N> want,
                        float abs_tol = 1e-5f, float rel_tol = 1e-4f) {
  for (int i = 0; i < N; ++i) {
    INFO("index " << i << ": got " << got[i] << ", want " << want[i]);
    CHECK(std::abs(got[i] - want[i]) <=
          abs_tol + rel_tol * std::abs(want[i]));
  }
}

template<std::size_t Rank>
static void check_close(util::mdarray<util::data_type::F32, Rank> & got, const std::mdspan<float, std::dextents<std::size_t, Rank>> want,
                        float abs_tol = 1e-5f, float rel_tol = 1e-4f) {
  assert(got.size() == want.size());
  auto vec = got.template view<1>(got.size());
  std::mdspan<float, std::dextents<std::size_t, 1>> want_vec{ want.data_handle(), want.size() };
  for (int i = 0; i < got.size(); ++i) {
    INFO("index " << i << ": got " << vec[i] << ", want " << want_vec[i]);
    CHECK(std::abs(vec[i] - want_vec[i]) <=
          abs_tol + rel_tol * std::abs(want_vec[i]));
  }
}

util::execution_context<util::execution_policy::sequential> execution_context{};

template<typename T>
util::mat_t<T, 1> as_mat(T* data, std::size_t extent) {
  return util::mat_t<T, 1>{data, extent};
}

// --------------------------------------------------------------- embed
TEST_CASE("embed: copies the token's row") {
  // 4-token vocab, dim 3; row r is {10r, 10r+1, 10r+2} so any wrong row
  // or off-by-one stride is unmistakable in the failure message.
  constexpr std::array<float, 12> data{0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 31, 32};
  const std::mdspan<const float, std::dextents<std::size_t, 2>> table_data{data.data(), 4, 3};
  std::array<float, 3> out {-1, -1, -1};
  operations::embed(as_mat(out.data(), 3), table_data, 2);
  std::array<float, 3> want {20, 21, 22};
  check_close(out, want);
}

TEST_CASE("embed: row 0 and last row (fencepost)") {
  constexpr std::array<float, 6> data{ 5, 6, 7, 8, 9, 10 };
  const std::mdspan<const float, std::dextents<std::size_t, 2>> table{data.data(), 2, 3};
  float out[3];
  operations::embed(as_mat(out, 3), table, 0);
  CHECK(out[0] == 5.0f);
  operations::embed(as_mat(out, 3), table, 1);
  CHECK(out[2] == 10.0f);
}

// ------------------------------------------------------------- rmsnorm
TEST_CASE("rmsnorm: golden vector") {
  util::mdarray<util::data_type::F32, 1> out{ golden::RMSNORM_DIM };

  operations::rmsnorm(
    out.mdspan(),
    golden::RMSNORM_X,
    golden::RMSNORM_W,
    1e-5f);
  check_close(out, golden::RMSNORM_OUT);
}

TEST_CASE("rmsnorm: in-place aliasing (out == x) per contract") {
  util::mdarray<util::data_type::F32, 1> x{ golden::RMSNORM_DIM };

  for (int i = 0; i < golden::RMSNORM_DIM; ++i) x[i] = golden::RMSNORM_X[i];
  operations::rmsnorm(
    x.mdspan(),
    x.mdspan(),
    golden::RMSNORM_W,
    1e-5f);
  check_close(x, golden::RMSNORM_OUT);
}

TEST_CASE("rmsnorm: unit weights on a constant vector are ~identity") {
  // x = c everywhere → rms = sqrt(c² + eps) ≈ |c|; out ≈ sign-preserved 1·c/|c|·...
  // Simplest invariant: with w=1 and c=2, every output ≈ 2/2 = 1... times c.
  const int d = 16;
  util::mdarray<util::data_type::F32, 1> x{ d }, w{ d }, out{ d };
  for (int i = 0; i < d; ++i) { x[i] = 2.0f; w[i] = 1.0f; }
  operations::rmsnorm(
    out.mdspan(),
    x.mdspan(),
    w.mdspan(),
    1e-5f);
  for (int i = 0; i < d; ++i) CHECK(out[i] == Catch::Approx(1.0f).epsilon(1e-4));
}

TEST_CASE("rmsnorm: eps is inside the sqrt (all-zero input stays finite)") {
  const int d = 4;
  float x[d] = {0, 0, 0, 0}, w[d] = {1, 1, 1, 1}, out[d];
  operations::rmsnorm(
    as_mat(out, d),
    as_mat(x, d),
    as_mat(w, d),
    1e-5f);
  for (int i = 0; i < d; ++i) {
    CHECK(std::isfinite(out[i]));
    CHECK(out[i] == 0.0f);
  }
}

// -------------------------------------------------------------- matvec
TEST_CASE("matvec: golden vector") {
  util::mdarray<util::data_type::F32, 1> out{ golden::MATVEC_ROWS };
  operations::matvec(
    execution_context,
    out.mdspan(),
    golden::MATVEC_W,
    golden::MATVEC_X);
  check_close(out, golden::MATVEC_OUT);
}

TEST_CASE("matvec: overwrites the output array") {
  util::mdarray<util::data_type::F32, 1> out{golden::MATVEC_ROWS};
  for (int i = 0; i < golden::MATVEC_ROWS; ++i) out.mdspan()[i] = 100.0f;
  operations::matvec(
    execution_context,
    out.mdspan(),
    golden::MATVEC_W,
    golden::MATVEC_X);
  check_close(out, golden::MATVEC_OUT);
}

TEST_CASE("matvec: identity matrix returns x") {
  constexpr int n = 5;
  std::vector<float> I(n * n, 0.0f);
  for (int i = 0; i < n; ++i) I[i * n + i] = 1.0f;
  const std::array<float, n> x = {3, -1, 4, -1, 5};
  std::array<float, n> out{};
  const std::mdspan<const float, std::dextents<std::size_t, 2>> I_data{I.data(), n, n};
  operations::matvec(
    execution_context,
    as_mat(out.data(), n),
    I_data,
    as_mat(x.data(), n));
  check_close(out, x);
}

TEST_CASE("matvec: non-square catches row/col swaps") {
  // [2×3] · [3] — a rows/cols mixup either miscomputes or walks OOB
  // under ASan. W.row(0)={1,2,3}, W.row(1)={4,5,6}, x={1,10,100}.
  const float W[6] = {1, 2, 3, 4, 5, 6};
  const float x[3] = {1, 10, 100};
  float out[2];
  const std::mdspan<const float, std::dextents<std::size_t, 2>> W_data{W, 2, 3};
  operations::matvec(
    execution_context,
    as_mat(out, 2),
    W_data,
    as_mat(x, 3));
  CHECK(out[0] == Catch::Approx(321.0f));
  CHECK(out[1] == Catch::Approx(654.0f));
}

// ---------------------------------------------------------------- rope
TEST_CASE("rope: golden vector (HF split-half, GQA head counts)") {
  util::mdarray<util::data_type::F32, 2> q{ golden::ROPE_NQ, golden::ROPE_HEAD_DIM };
  std::memcpy(q.mdspan().data_handle(), golden::ROPE_Q_IN.data_handle(), golden::ROPE_Q_IN.size() * sizeof(float));
  util::mdarray<util::data_type::F32, 2> k{ golden::ROPE_NKV, golden::ROPE_HEAD_DIM };
  std::memcpy(k.mdspan().data_handle(), golden::ROPE_K_IN.data_handle(), golden::ROPE_K_IN.size() * sizeof(float));
  operations::rope(q.mdspan(), golden::ROPE_POS, golden::ROPE_BASE);
  operations::rope(k.mdspan(), golden::ROPE_POS, golden::ROPE_BASE);
  check_close(q, golden::ROPE_Q_OUT);
  check_close(k, golden::ROPE_K_OUT);
}

TEST_CASE("rope: position 0 is the identity") {
  constexpr int hd = 8;
  std::array<float, hd> q, k, q0, k0;
  for (int i = 0; i < hd; ++i) q0[i] = q[i] = 0.5f * i - 2.0f;
  for (int i = 0; i < hd; ++i) k0[i] = k[i] = -0.25f * i + 1.0f;
  operations::rope(std::mdspan<float, std::dextents<std::size_t, 2>>{q.data(), 1, hd}, /*pos=*/0, 10000.0f);
  operations::rope(std::mdspan<float, std::dextents<std::size_t, 2>>{k.data(), 1, hd}, /*pos=*/0, 10000.0f);
  check_close(q, q0);
  check_close(k, k0);
}

TEST_CASE("rope: rotation preserves the norm of each half-pair") {
  // (v[i], v[i+d/2]) is rotated as a 2-vector; its length is invariant.
  // An interleaved (paper-style) implementation pairs (v[i], v[i+1]) and
  // fails this for asymmetric input.
  const int hd = 8, half = hd / 2;
  float q[hd], k[hd], q_in[hd];
  for (int i = 0; i < hd; ++i) q_in[i] = q[i] = float(i + 1);  // asymmetric
  for (int i = 0; i < hd; ++i) k[i] = 0.0f;
  operations::rope(std::mdspan<float, std::dextents<std::size_t, 2>>{q, 1, hd}, /*pos=*/7, 10000.0f);
  operations::rope(std::mdspan<float, std::dextents<std::size_t, 2>>{k, 1, hd}, /*pos=*/7, 10000.0f);
  for (int i = 0; i < half; ++i) {
    const float before = q_in[i] * q_in[i] + q_in[i + half] * q_in[i + half];
    const float after = q[i] * q[i] + q[i + half] * q[i + half];
    CHECK(after == Catch::Approx(before).epsilon(1e-4));
  }
}

// ------------------------------------------------------------- softmax
TEST_CASE("softmax: golden vector with overflow-inducing values") {
  std::array<float, golden::SOFTMAX_N> x{ golden::SOFTMAX_X };
  operations::softmax(as_mat(x.data(), golden::SOFTMAX_N));
  check_close(x, golden::SOFTMAX_OUT);
}

TEST_CASE("softmax: output sums to 1 and is finite") {
  std::array<float, 4> x{0.0f, 100.0f, -100.0f, 50.0f};
  operations::softmax(as_mat(x.data(), 4));
  float s = 0.0f;
  for (float v : x) {
    CHECK(std::isfinite(v));
    CHECK(v >= 0.0f);
    s += v;
  }
  CHECK(s == Catch::Approx(1.0f).epsilon(1e-5));
}

TEST_CASE("softmax: uniform input → uniform output") {
  std::array<float, 8> x{};
  for (float& v : x) v = 3.7f;
  operations::softmax(as_mat(x.data(), 8));
  for (float v : x) CHECK(v == Catch::Approx(0.125f).epsilon(1e-5));
}

// ----------------------------------------------------------------- add
TEST_CASE("add: elementwise and in-place aliasing") {
  const std::array<float, 4> a{1, 2, 3, 4}, b{10, 20, 30, 40};
  std::array<float, 4> out;
  operations::add(
    as_mat(out.data(), 4),
    as_mat(a.data(), 4),
    as_mat(b.data(), 4));
  const std::array<float, 4> want{11, 22, 33, 44};
  check_close(out, want);

  std::array<float, 4> acc{1, 2, 3, 4};
  operations::add(
    as_mat(acc.data(), 4),
    as_mat(acc.data(), 4),
    as_mat(b.data(), 4));  // residual-style: out == a
  check_close(acc, want);
}

// ------------------------------------------------------------ silu_mul
TEST_CASE("silu_mul: golden vector") {
  std::array<float, golden::SILU_N> out{};
  operations::silu_mul(
    as_mat(out.data(),  6),
    as_mat(golden::SILU_GATE.data(), 6),
    as_mat(golden::SILU_UP.data(), 6));
  check_close(out, golden::SILU_OUT);
}

TEST_CASE("silu_mul: gate/up roles are not symmetric (swap must fail)") {
  // silu(2)·5 = 8.808…, silu(5)·2 = 9.933… — a swapped implementation
  // produces the second value and fails.
  const float gate[1] = {2.0f}, up[1] = {5.0f};
  float out[1];
  operations::silu_mul(
    as_mat(out, 1),
    as_mat(gate, 1),
    as_mat(up, 1));
  const float want = 2.0f / (1.0f + std::exp(-2.0f)) * 5.0f;
  CHECK(out[0] == Catch::Approx(want).epsilon(1e-5));
}

// -------------------------------------------------------------- argmax
TEST_CASE("argmax: golden, ties resolve to lowest index") {
  CHECK(operations::argmax(as_mat(golden::ARGMAX_X.data(), golden::ARGMAX_X.size())) == golden::ARGMAX_EXPECT);
}

TEST_CASE("argmax: single element, max at ends") {
  const std::array<float, 1> one{ -5.0f };
  CHECK(operations::argmax(as_mat(one.data(), 1)) == 0);
  const std::array<float, 3> first{9, 1, 2};
  CHECK(operations::argmax(as_mat(first.data(), 3)) == 0);
  const std::array<float, 3> last{1, 2, 9};
  CHECK(operations::argmax(as_mat(last.data(), 3)) == 2);
}

// ---------------------------------- integration: single-head attention
// The stage gate: kernels compose into attention over a 3-entry KV cache
// with zero code beyond the ops themselves. Green here → wire the layer.
TEST_CASE("attention: one head over a small KV cache matches numpy") {
  const int T = golden::ATTN_T, d = golden::ATTN_HEAD_DIM;

  std::array<float, golden::ATTN_T> scores{};
  // K is [T × d]; scores = K·q is exactly a matvec.
  std::mdspan<const float, std::dextents<std::size_t, 2>> K{golden::ATTN_K.data(), T, d};
  operations::matvec(
    execution_context,as_mat(scores.data(), golden::ATTN_T), K, as_mat(golden::ATTN_Q.data(), golden::ATTN_HEAD_DIM));
  const float inv_sqrt_d = 1.0f / std::sqrt(float(d));
  for (int t = 0; t < T; ++t) scores[t] *= inv_sqrt_d;
  operations::softmax(as_mat(scores.data(), golden::ATTN_T));

  std::array<float, golden::ATTN_HEAD_DIM> out{};
  for (int t = 0; t < T; ++t)
    for (int j = 0; j < d; ++j)
      out[j] += scores[t] * golden::ATTN_V[t * d + j];

  check_close(out, golden::ATTN_OUT);
}
