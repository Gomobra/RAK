#include "rak/ops/norm.h"

#include <stdexcept>
#include <cmath>
#include <numeric>
#include <limits>

namespace rak::ops {

std::shared_ptr<core::Tensor> rms_norm(
    const std::shared_ptr<core::Tensor>& X,
    const std::shared_ptr<core::Tensor>& weight,
    float eps) {
  if (!X || !weight) throw std::invalid_argument("rms_norm: input null");
  if (X->ndim() < 1) throw std::invalid_argument("rms_norm: input butuh setidaknya 1 axis");
  size_t d_model = X->shape.back();
  if (weight->numel() != d_model) {
    throw std::invalid_argument("rms_norm: ukuran weight harus sama dengan axis terakhir");
  }

  size_t ndim = X->ndim();
  size_t outer = 1;
  for (size_t i = 0; i + 1 < ndim; ++i) outer *= X->shape[i];

  std::vector<float> out(X->numel(), 0.0f);
  // Simpan rms dan x untuk backward.
  std::vector<float> rms(outer, 0.0f);
  std::vector<float> inv_rms(outer, 0.0f);

  for (size_t o = 0; o < outer; ++o) {
    float sq = 0.0f;
    for (size_t d = 0; d < d_model; ++d) {
      float v = X->ptr[o * d_model + d];
      sq += v * v;
    }
    float mean_sq = sq / static_cast<float>(d_model);
    float r = std::sqrt(mean_sq + eps);
    rms[o] = r;
    inv_rms[o] = 1.0f / r;
    for (size_t d = 0; d < d_model; ++d) {
      out[o * d_model + d] = (X->ptr[o * d_model + d] * inv_rms[o]) * weight->ptr[d];
    }
  }

  return core::make_op_output(
      {X, weight},
      std::move(out),
      X->shape,
      [X, weight, d_model, outer, rms = std::move(rms), inv_rms = std::move(inv_rms), eps](
          const std::vector<float>& grad_out) {
        size_t xnumel = X->numel();
        std::vector<float> gX(xnumel, 0.0f);
        std::vector<float> gW(d_model, 0.0f);

        for (size_t o = 0; o < outer; ++o) {
          float ir = inv_rms[o];
          float r = rms[o];
          // grad wrt x: (1/r) * w * (gout - x * sum(gout * w * x) / (r^2 + eps) ... )
          // Rumus RMSNorm backward:
          // y = x * ir * w
          // dy/dx = ir * w * (I - x * x^T / (d_model * (r^2)))  (approx, karena ir bergantung pada x)
          // Untuk keefisienan: hitung direkt.
          float sum_gxw = 0.0f;  // sum(gout * w * x)
          for (size_t d = 0; d < d_model; ++d) {
            float x = X->ptr[o * d_model + d];
            float w = weight->ptr[d];
            float go = grad_out[o * d_model + d];
            gW[d] += go * x * ir;
            sum_gxw += go * w * x;
          }
          float coeff = sum_gxw / (static_cast<float>(d_model) * (r * r));
          for (size_t d = 0; d < d_model; ++d) {
            float x = X->ptr[o * d_model + d];
            float w = weight->ptr[d];
            float go = grad_out[o * d_model + d];
            gX[o * d_model + d] = ir * w * (go - x * coeff);
          }
        }
        return std::vector<std::vector<float>>{gX, gW};
      });
}

std::shared_ptr<core::Tensor> softmax(const std::shared_ptr<core::Tensor>& X) {
  if (!X) throw std::invalid_argument("softmax: input null");
  if (X->ndim() < 1) throw std::invalid_argument("softmax: input butuh setidaknya 1 axis");
  size_t ndim = X->ndim();
  size_t d = X->shape.back();
  size_t outer = 1;
  for (size_t i = 0; i + 1 < ndim; ++i) outer *= X->shape[i];

  std::vector<float> out(X->numel(), 0.0f);
  std::vector<float> row_max(outer, 0.0f);
  std::vector<float> row_sum(outer, 0.0f);

  for (size_t o = 0; o < outer; ++o) {
    float mx = -std::numeric_limits<float>::infinity();
    for (size_t j = 0; j < d; ++j) mx = std::max(mx, X->ptr[o * d + j]);
    row_max[o] = mx;
    float s = 0.0f;
    for (size_t j = 0; j < d; ++j) {
      float v = std::exp(X->ptr[o * d + j] - mx);
      out[o * d + j] = v;
      s += v;
    }
    row_sum[o] = s;
    for (size_t j = 0; j < d; ++j) out[o * d + j] /= s;
  }

  return core::make_op_output(
      {X},
      std::move(out),
      X->shape,
      [X, d, outer, out = std::move(out)](const std::vector<float>& grad_out) {
        // Softmax backward: dx = out * (gout - sum(gout * out))
        size_t xnumel = X->numel();
        std::vector<float> g(xnumel, 0.0f);
        for (size_t o = 0; o < outer; ++o) {
          float dot = 0.0f;
          for (size_t j = 0; j < d; ++j) {
            dot += grad_out[o * d + j] * out[o * d + j];
          }
          for (size_t j = 0; j < d; ++j) {
            g[o * d + j] = out[o * d + j] * (grad_out[o * d + j] - dot);
          }
        }
        return std::vector<std::vector<float>>{g};
      });
}

std::shared_ptr<core::Tensor> log_softmax(const std::shared_ptr<core::Tensor>& X) {
  if (!X) throw std::invalid_argument("log_softmax: input null");
  if (X->ndim() < 1) throw std::invalid_argument("log_softmax: input butuh setidaknya 1 axis");
  size_t ndim = X->ndim();
  size_t d = X->shape.back();
  size_t outer = 1;
  for (size_t i = 0; i + 1 < ndim; ++i) outer *= X->shape[i];

  std::vector<float> out(X->numel(), 0.0f);
  std::vector<float> row_max(outer, 0.0f);
  std::vector<float> row_sum(outer, 0.0f);

  for (size_t o = 0; o < outer; ++o) {
    float mx = -std::numeric_limits<float>::infinity();
    for (size_t j = 0; j < d; ++j) mx = std::max(mx, X->ptr[o * d + j]);
    row_max[o] = mx;
    float s = 0.0f;
    for (size_t j = 0; j < d; ++j) s += std::exp(X->ptr[o * d + j] - mx);
    row_sum[o] = s;
    float log_s = std::log(s);
    for (size_t j = 0; j < d; ++j) {
      out[o * d + j] = (X->ptr[o * d + j] - mx) - log_s;
    }
  }

  return core::make_op_output(
      {X},
      std::move(out),
      X->shape,
      [X, d, outer, out = std::move(out)](const std::vector<float>& grad_out) {
        // d log_softmax/dx = gout - softmax(x) * sum(gout)
        size_t xnumel = X->numel();
        std::vector<float> g(xnumel, 0.0f);
        for (size_t o = 0; o < outer; ++o) {
          float dot = 0.0f;
          for (size_t j = 0; j < d; ++j) dot += grad_out[o * d + j];
          for (size_t j = 0; j < d; ++j) {
            float p = std::exp(out[o * d + j]);
            g[o * d + j] = grad_out[o * d + j] - p * dot;
          }
        }
        return std::vector<std::vector<float>>{g};
      });
}

}  // namespace rak::ops
