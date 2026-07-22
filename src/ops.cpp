#include "infer/ops.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <ranges>
#include <cmath>
#include <span>

void culpeo::inference::embed(std::span<float> out, std::mdspan<const float, std::dextents<std::size_t, 2>> table, std::ptrdiff_t token_id)
{
  assert(token_id >= 0 && static_cast<std::size_t>(token_id) < table.extent(0));
  assert(out.size() == table.extent(1));
  for (std::size_t i = 0; i < out.size(); ++i) out[i] = table[token_id, i];
}

void culpeo::inference::rmsnorm(std::span<float> out, std::span<const float> x, std::span<const float> weight,  float eps)
{
  assert(out.size() == x.size());
  assert(out.size() == weight.size());
  const auto sigma = std::ranges::fold_left(x | std::views::transform([](auto v) { return v * v ; }), 0.0f, std::plus<float>{});
  const auto rms = std::sqrt(sigma / static_cast<float>(x.size()) + eps);
  std::ranges::copy(std::views::zip(x, weight) | std::views::transform([rms](const auto & pair) { const auto [xi, wi] = pair; return xi / rms * wi; }), out.begin());
}

void culpeo::inference::matvec(std::span<float> out, std::mdspan<const float, std::dextents<std::size_t, 2>> W, std::span<const float> x)
{
  assert(out.size() == W.extent(0));
  assert(x.size() == W.extent(1));
  assert(W.stride(0) == W.extent(1));
  for (std::size_t r = 0; r < W.extent(0); r++)
  {
    std::span<const float> row{W.data_handle() + r * W.extent(1), W.extent(1)};
    out[r] = std::ranges::fold_left(std::views::zip(row, x) | std::views::transform([](const auto & pair) { const auto [wi, xi] = pair; return wi * xi; }), 0.0f, std::plus<float>{});
  }
}

void culpeo::inference::rope(std::mdspan<float, std::dextents<std::size_t, 2>> t, std::ptrdiff_t pos, float theta_base) {
  assert(t.extent(1) % 2 == 0);
  for (std::size_t h = 0; h < t.extent(0); h++)
  {
    for (std::size_t i = 0; i < t.extent(1) / 2; i++)
    {
      const double inv_freq = std::pow(static_cast<double>(theta_base), -2.0 * static_cast<double>(i) / t.extent(1));
      auto angle = pos * inv_freq;
      auto cos = std::cos(angle);
      auto sin = std::sin(angle);
      auto a = t[h, i];
      auto b = t[h, i + t.extent(1) / 2];
      t[h, i] = static_cast<float>(a * cos - b * sin);
      t[h, i + t.extent(1) / 2] = static_cast<float>(b * cos + a * sin);
    }
  }
}

void culpeo::inference::softmax(std::span<float> x) {
  const float max = std::ranges::max(x);
  std::ranges::transform(x, x.begin(), [max](float v) { return std::exp(v - max); });
  const float sum = std::ranges::fold_left(x, 0.0f, std::plus<float>{});
  std::ranges::transform(x, x.begin(), [sum](float v) { return v / sum; });
}

void culpeo::inference::add(std::span<float> out, std::span<const float> a, std::span<const float> b) {
  assert(out.size() == a.size());
  assert(out.size() == b.size());
  std::ranges::copy(std::views::zip(a, b) | std::views::transform([](const auto & pair) { const auto [ai, bi] = pair; return ai + bi; }), out.begin());
}

void culpeo::inference::silu_mul(std::span<float> out, std::span<const float> gate, std::span<const float> up) {
  assert(out.size() == gate.size());
  assert(out.size() == up.size());
  auto silu = gate | std::views::transform([](auto gate_i) { return gate_i / (1 + std::exp(-gate_i)); });
  std::ranges::copy(std::views::zip(silu, up) | std::views::transform([](const auto & pair) { const auto [s_i, u_i] = pair; return s_i * u_i; }), out.begin());
}

std::ptrdiff_t culpeo::inference::argmax(std::span<const float> x) {
  return std::ranges::max_element(x) - x.begin();
}
