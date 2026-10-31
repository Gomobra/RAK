#include "rak/ops/local_attention.h"

#include <stdexcept>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>

namespace rak::ops {

std::shared_ptr<core::Tensor> local_attention(
    const std::shared_ptr<core::Tensor>& Q,
    const std::shared_ptr<core::Tensor>& K,
    const std::shared_ptr<core::Tensor>& V,
    size_t window) {
  if (!Q || !K || !V) throw std::invalid_argument("local_attention: input null");
  if (Q->ndim() != 3 || K->ndim() != 3 || V->ndim() != 3) {
    throw std::invalid_argument("local_attention: input harus 3D [batch, seq, d]");
  }
  size_t batch = Q->shape[0];
  size_t seq = Q->shape[1];
  size_t d = Q->shape[2];
  if (K->shape[0] != batch || K->shape[1] != seq || K->shape[2] != d) {
    throw std::invalid_argument("local_attention: K harus sama shape dengan Q");
  }
  if (V->shape[0] != batch || V->shape[1] != seq || V->shape[2] != d) {
    throw std::invalid_argument("local_attention: V harus sama shape dengan Q");
  }
  if (window == 0) window = seq;  // window 0 = attention penuh

  std::vector<float> out(batch * seq * d, 0.0f);
  float scale = 1.0f / std::sqrt(static_cast<float>(d));

  for (size_t b = 0; b < batch; ++b) {
    for (size_t i = 0; i < seq; ++i) {
      size_t lo = (i >= window) ? i - window : 0;
      size_t hi = (i + window + 1 < seq) ? i + window + 1 : seq;
      // Hitung logits untuk jendela, cari max untuk stabilitas numerik.
      std::vector<float> logits(hi - lo);
      float mx = -std::numeric_limits<float>::infinity();
      for (size_t j = lo; j < hi; ++j) {
        float l = 0.0f;
        for (size_t k = 0; k < d; ++k) {
          l += Q->ptr[((b * seq + i) * d) + k] * K->ptr[((b * seq + j) * d) + k];
        }
        l *= scale;
        logits[j - lo] = l;
        mx = std::max(mx, l);
      }
      // Exp + sum.
      float sum = 0.0f;
      for (size_t j = lo; j < hi; ++j) {
        float v = std::exp(logits[j - lo] - mx);
        logits[j - lo] = v;
        sum += v;
      }
      // Weighted sum V.
      for (size_t k = 0; k < d; ++k) {
        float acc = 0.0f;
        for (size_t j = lo; j < hi; ++j) {
          float p = logits[j - lo] / sum;
          acc += p * V->ptr[((b * seq + j) * d) + k];
        }
        out[((b * seq + i) * d) + k] = acc;
      }
    }
  }

  return core::make_op_output(
      {Q, K, V},
      std::move(out),
      {batch, seq, d},
      [Q, K, V, window](const std::vector<float>& grad_out)
          -> std::vector<std::vector<float>> {
        auto [gq, gk, gv] = ops::local_attention_backward(Q, K, V, window, grad_out);
        return {std::move(gq), std::move(gk), std::move(gv)};
      });
}

std::tuple<std::vector<float>, std::vector<float>, std::vector<float>>
local_attention_backward(
    const std::shared_ptr<core::Tensor>& Q,
    const std::shared_ptr<core::Tensor>& K,
    const std::shared_ptr<core::Tensor>& V,
    size_t window,
    const std::vector<float>& grad_out) {
  size_t batch = Q->shape[0];
  size_t seq = Q->shape[1];
  size_t d = Q->shape[2];
  if (window == 0) window = seq;

  std::vector<float> grad_Q(batch * seq * d, 0.0f);
  std::vector<float> grad_K(batch * seq * d, 0.0f);
  std::vector<float> grad_V(batch * seq * d, 0.0f);
  float scale = 1.0f / std::sqrt(static_cast<float>(d));

  // Hitung ulang prob (sama seperti forward) untuk backward.
  for (size_t b = 0; b < batch; ++b) {
    for (size_t i = 0; i < seq; ++i) {
      size_t lo = (i >= window) ? i - window : 0;
      size_t hi = (i + window + 1 < seq) ? i + window + 1 : seq;
      size_t w = hi - lo;
      std::vector<float> logits(w);
      float mx = -std::numeric_limits<float>::infinity();
      for (size_t jj = 0; jj < w; ++jj) {
        size_t j = lo + jj;
        float l = 0.0f;
        for (size_t k = 0; k < d; ++k) {
          l += Q->ptr[((b * seq + i) * d) + k] * K->ptr[((b * seq + j) * d) + k];
        }
        l *= scale;
        logits[jj] = l;
        mx = std::max(mx, l);
      }
      std::vector<float> p(w);
      float sum = 0.0f;
      for (size_t jj = 0; jj < w; ++jj) {
        float v = std::exp(logits[jj] - mx);
        p[jj] = v;
        sum += v;
      }
      for (size_t jj = 0; jj < w; ++jj) p[jj] /= sum;

      // grad wrt logits: dp = sum_k grad_out[i,k] * V[j,k]; then dl = p * (dp - sum(dp*p))
      std::vector<float> dp(w, 0.0f);
      for (size_t jj = 0; jj < w; ++jj) {
        size_t j = lo + jj;
        for (size_t k = 0; k < d; ++k) {
          dp[jj] += grad_out[((b * seq + i) * d) + k] * V->ptr[((b * seq + j) * d) + k];
        }
      }
      float sum_dp = 0.0f;
      for (size_t jj = 0; jj < w; ++jj) sum_dp += dp[jj] * p[jj];
      std::vector<float> dl(w);
      for (size_t jj = 0; jj < w; ++jj) dl[jj] = p[jj] * (dp[jj] - sum_dp) * scale;

      // Akumulasi grad_Q[i] += sum_j dl[j] * K[j]
      for (size_t k = 0; k < d; ++k) {
        float acc = 0.0f;
        for (size_t jj = 0; jj < w; ++jj) {
          size_t j = lo + jj;
          acc += dl[jj] * K->ptr[((b * seq + j) * d) + k];
        }
        grad_Q[((b * seq + i) * d) + k] += acc;
      }
      // Akumulasi grad_K[j] += dl[j] * Q[i]
      for (size_t jj = 0; jj < w; ++jj) {
        size_t j = lo + jj;
        for (size_t k = 0; k < d; ++k) {
          grad_K[((b * seq + j) * d) + k] += dl[jj] * Q->ptr[((b * seq + i) * d) + k];
        }
      }
      // Akumulasi grad_V[j] += p[j] * grad_out[i]
      for (size_t jj = 0; jj < w; ++jj) {
        size_t j = lo + jj;
        for (size_t k = 0; k < d; ++k) {
          grad_V[((b * seq + j) * d) + k] += p[jj] * grad_out[((b * seq + i) * d) + k];
        }
      }
    }
  }

  return {grad_Q, grad_K, grad_V};
}

}  // namespace rak::ops
