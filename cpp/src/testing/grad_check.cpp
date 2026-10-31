#include "rak/testing/grad_check.h"

#include <algorithm>
#include <limits>

namespace rak::testing {

float numerical_grad_elem(
    const ForwardFn& forward,
    const std::vector<std::shared_ptr<rak::core::Tensor>>& inputs,
    size_t input_idx,
    size_t elem_idx,
    float eps) {
  auto& in = inputs[input_idx];
  float orig = in->ptr[elem_idx];
  in->ptr[elem_idx] = orig + eps;
  auto fp = forward(inputs);
  float vp = fp->at(0);
  in->ptr[elem_idx] = orig - eps;
  auto fm = forward(inputs);
  float vm = fm->at(0);
  in->ptr[elem_idx] = orig;
  return (vp - vm) / (2.0f * eps);
}

std::vector<GradCheckResult> gradient_check(
    const ForwardFn& forward,
    const std::vector<std::shared_ptr<rak::core::Tensor>>& inputs,
    float eps,
    float atol,
    float rtol) {
  std::vector<GradCheckResult> results;

  for (size_t ii = 0; ii < inputs.size(); ++ii) {
    auto& in = inputs[ii];
    if (!in->requires_grad) continue;

    GradCheckResult r;
    r.name = "input[" + std::to_string(ii) + "]";
    r.ok = true;
    r.max_abs_diff = 0.0f;
    r.max_rel_diff = 0.0f;
    r.n_checked = 0;

    // Zero gradien sebelum backward (backward mengakumulasi ke grad).
    for (auto& in : inputs) {
      if (in->requires_grad) {
        std::fill(in->grad.begin(), in->grad.end(), 0.0f);
      }
    }

    // Jalankan forward sekali untuk membangun grafik, lalu backward.
    rak::core::g_autograd.reset();
    auto loss = forward(inputs);
    if (loss->numel() != 1) {
      r.ok = false;
      results.push_back(r);
      continue;
    }
    rak::core::g_autograd.backward(loss);

    for (size_t e = 0; e < in->numel(); ++e) {
      float g_analytic = in->grad[e];
      float g_numeric = numerical_grad_elem(forward, inputs, ii, e, eps);
      float diff = std::abs(g_analytic - g_numeric);
      r.max_abs_diff = std::max(r.max_abs_diff, diff);
      float denom = std::max(std::abs(g_analytic), std::abs(g_numeric));
      float rel = (denom > 0.0f) ? (diff / denom) : diff;
      r.max_rel_diff = std::max(r.max_rel_diff, rel);
      ++r.n_checked;

      if (diff > atol && rel > rtol) {
        r.ok = false;
        std::cerr << "[GRAD_CHECK FAIL] " << r.name << " elem=" << e
                  << " analytic=" << g_analytic << " numeric=" << g_numeric
                  << " abs_diff=" << diff << " rel_diff=" << rel << "\n";
      }
    }
    results.push_back(r);
  }
  return results;
}

}  // namespace rak::testing
