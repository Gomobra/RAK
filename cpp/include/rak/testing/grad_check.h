#ifndef RAK_TESTING_GRAD_CHECK_H
#define RAK_TESTING_GRAD_CHECK_H

#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "rak/core/tensor.h"
#include "rak/core/autograd.h"

namespace rak::testing {

// Hasil gradient check untuk satu parameter.
struct GradCheckResult {
  std::string name;
  bool ok;
  float max_abs_diff;
  float max_rel_diff;
  size_t n_checked;
};

// Fungsi forward: mengambil vector input tensor, mengembalikan loss skalar (numel==1).
using ForwardFn = std::function<std::shared_ptr<rak::core::Tensor>(
    const std::vector<std::shared_ptr<rak::core::Tensor>>&)>;

// Verifikasi gradien analitis (dari backward) terhadap gradien numerik (finite
// difference central) untuk semua elemen semua input yang requires_grad.
std::vector<GradCheckResult> gradient_check(
    const ForwardFn& forward,
    const std::vector<std::shared_ptr<rak::core::Tensor>>& inputs,
    float eps = 1e-4f,
    float atol = 1e-3f,
    float rtol = 1e-2f);

// Gradien numerik untuk satu elemen input ke-elem.
float numerical_grad_elem(
    const ForwardFn& forward,
    const std::vector<std::shared_ptr<rak::core::Tensor>>& inputs,
    size_t input_idx,
    size_t elem_idx,
    float eps);

}  // namespace rak::testing

#endif  // RAK_TESTING_GRAD_CHECK_H
