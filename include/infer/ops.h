#pragma once

#include <cstddef>
#include <stdint.h>

#include <infer/types.h>

namespace culpeo::inference {


    void embed(types::vec_t<float> out, types::cmat_t<float> table, std::ptrdiff_t token_id);

    void rmsnorm(types::vec_t<float> out, types::cvec_t<float> x, types::cvec_t<float> weight, float eps);

    void matvec(types::vec_t<float> out, types::cmat_t<float> W, types::cvec_t<float> x);

    void rope(types::mat_t<float> t, std::ptrdiff_t pos, float theta_base);

    void softmax(types::vec_t<float> x);

    void add(types::vec_t<float> out, types::cvec_t<float> a, types::cvec_t<float> b);

    void silu_mul(types::vec_t<float> out, types::cvec_t<float> gate, types::cvec_t<float> up);

    std::ptrdiff_t argmax(types::cvec_t<float> x);
}
