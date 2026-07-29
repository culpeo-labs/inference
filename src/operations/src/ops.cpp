#include <cassert>
#include <cstddef>
#include <cmath>

#include <operations/ops.h>
#include <util/types.h>

using namespace culpeo::inference;

void operations::rope(util::mat_t<float, 2> t, std::ptrdiff_t pos, float theta_base) {
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

void operations::softmax(util::mat_t<float, 1> x) {
  float max = std::numeric_limits<float>::lowest();
  for (std::size_t i = 0; i < x.extent(0); i++) {
      if (x[i] > max) {
          max = x[i];
      }
  }
  double sum = 0.0;
  for (std::size_t i = 0; i < x.extent(0); i++) {
      x[i] = std::exp(x[i] - max);
      sum += x[i];
  }
  for (std::size_t i = 0; i < x.extent(0); i++) {
      x[i] /= static_cast<float>(sum);
  }
}
