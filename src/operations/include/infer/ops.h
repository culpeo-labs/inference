#pragma once

#include <cstddef>
#include <stdint.h>

#include <util/types.h>

namespace culpeo::inference {


    void embed(util::vec_t<float> out, util::cmat_t<float> table, std::ptrdiff_t token_id);

    void rmsnorm(util::vec_t<float> out, util::cvec_t<float> x, util::cvec_t<float> weight, float eps);

    void matvec(util::vec_t<float> out, util::cmat_t<float> W, util::cvec_t<float> x);

    void rope(util::mat_t<float> t, std::ptrdiff_t pos, float theta_base);

    void softmax(util::vec_t<float> x);

    void add(util::vec_t<float> out, util::cvec_t<float> a, util::cvec_t<float> b);

    void silu_mul(util::vec_t<float> out, util::cvec_t<float> gate, util::cvec_t<float> up);

    std::ptrdiff_t argmax(util::cvec_t<float> x);
}
