#include "rak/ops/elementwise.h"

#include <stdexcept>
#include <algorithm>
#include <numeric>

namespace rak::ops {

// Hitung shape hasil broadcast dari dua shape (NumPy-style).
static std::vector<size_t> broadcast_shape(
    const std::vector<size_t>& a, const std::vector<size_t>& b) {
  size_t ndim = std::max(a.size(), b.size());
  std::vector<size_t> result(ndim, 1);
  for (size_t i = 0; i < ndim; ++i) {
    size_t da = (i < a.size()) ? a[a.size() - 1 - i] : 1;
    size_t db = (i < b.size()) ? b[b.size() - 1 - i] : 1;
    if (da != 1 && db != 1 && da != db) {
      throw std::invalid_argument("elementwise: shape tidak dapat di-broadcast");
    }
    result[ndim - 1 - i] = std::max(da, db);
  }
  return result;
}

// Untuk setiap input T dengan shape kecil, hitung stride-nya dalam ruang out_shape
// agar kita dapat memetakan indeks linear out_shape ke indeks linear input
// dengan broadcast. Axis yang size-nya 1 di input mendapat stride 0.
static std::vector<size_t> input_strides_in_out(
    const std::shared_ptr<core::Tensor>& T,
    const std::vector<size_t>& out_shape) {
  std::vector<size_t> strides(out_shape.size(), 0);
  size_t diff = out_shape.size() - T->shape.size();
  size_t acc = 1;
  for (size_t i = T->shape.size(); i-- > 0;) {
    size_t oi = diff + i;
    if (T->shape[i] == 1) {
      strides[oi] = 0;  // broadcast: selalu indeks 0
    } else {
      strides[oi] = acc;
      acc *= T->shape[i];
    }
  }
  return strides;
}

// Iterasi semua indeks linear out_shape, menjalankan lambda untuk setiap elemen.
static void iterate_out(
    const std::vector<size_t>& out_shape,
    const std::function<void(const std::vector<size_t>& coord, size_t linear_idx)>& fn) {
  size_t numel = 1;
  for (size_t d : out_shape) numel *= d;
  std::vector<size_t> coord(out_shape.size(), 0);
  std::vector<size_t> out_stride(out_shape.size());
  size_t acc = 1;
  for (size_t i = out_shape.size(); i-- > 0;) {
    out_stride[i] = acc;
    acc *= out_shape[i];
  }
  for (size_t i = 0; i < numel; ++i) {
    size_t linear_idx = 0;
    for (size_t d = 0; d < out_shape.size(); ++d) linear_idx += coord[d] * out_stride[d];
    fn(coord, linear_idx);
    // Odometer: tambahkan indeks paling lambat.
    for (size_t d = out_shape.size(); d-- > 0;) {
      coord[d]++;
      if (coord[d] < out_shape[d]) break;
      coord[d] = 0;
    }
  }
}

std::shared_ptr<core::Tensor> add(
    const std::shared_ptr<core::Tensor>& A,
    const std::shared_ptr<core::Tensor>& B) {
  if (!A || !B) throw std::invalid_argument("add: input null");
  auto out_shape = broadcast_shape(A->shape, B->shape);
  size_t numel = 1;
  for (size_t d : out_shape) numel *= d;

  std::vector<float> C(numel, 0.0f);
  auto sa = input_strides_in_out(A, out_shape);
  auto sb = input_strides_in_out(B, out_shape);

  iterate_out(out_shape, [&](const std::vector<size_t>& coord, size_t idx) {
    size_t ia = 0, ib = 0;
    for (size_t d = 0; d < out_shape.size(); ++d) {
      ia += coord[d] * sa[d];
      ib += coord[d] * sb[d];
    }
    C[idx] = A->ptr[ia] + B->ptr[ib];
  });

  return core::make_op_output(
      {A, B},
      std::move(C),
      out_shape,
      [A, B, out_shape](const std::vector<float>& grad_out) {
        auto ga = ops::unbroadcast(grad_out, out_shape, A->shape, A->stride);
        auto gb = ops::unbroadcast(grad_out, out_shape, B->shape, B->stride);
        return std::vector<std::vector<float>>{ga, gb};
      });
}

std::shared_ptr<core::Tensor> mul(
    const std::shared_ptr<core::Tensor>& A,
    const std::shared_ptr<core::Tensor>& B) {
  if (!A || !B) throw std::invalid_argument("mul: input null");
  auto out_shape = broadcast_shape(A->shape, B->shape);
  size_t numel = 1;
  for (size_t d : out_shape) numel *= d;

  std::vector<float> C(numel, 0.0f);
  auto sa = input_strides_in_out(A, out_shape);
  auto sb = input_strides_in_out(B, out_shape);

  iterate_out(out_shape, [&](const std::vector<size_t>& coord, size_t idx) {
    size_t ia = 0, ib = 0;
    for (size_t d = 0; d < out_shape.size(); ++d) {
      ia += coord[d] * sa[d];
      ib += coord[d] * sb[d];
    }
    C[idx] = A->ptr[ia] * B->ptr[ib];
  });

  return core::make_op_output(
      {A, B},
      std::move(C),
      out_shape,
      [A, B, out_shape](const std::vector<float>& grad_out) {
        // grad_A = grad_out * B (unbroadcasted ke shape A)
        // grad_B = grad_out * A (unbroadcasted ke shape B)
        std::vector<float> grad_A_weighted(grad_out.size());
        std::vector<float> grad_B_weighted(grad_out.size());
        auto sa = input_strides_in_out(A, out_shape);
        auto sb = input_strides_in_out(B, out_shape);
        iterate_out(out_shape, [&](const std::vector<size_t>& coord, size_t idx) {
          size_t ia = 0, ib = 0;
          for (size_t d = 0; d < out_shape.size(); ++d) {
            ia += coord[d] * sa[d];
            ib += coord[d] * sb[d];
          }
          grad_A_weighted[idx] = grad_out[idx] * B->ptr[ib];
          grad_B_weighted[idx] = grad_out[idx] * A->ptr[ia];
        });
        auto ga = ops::unbroadcast(grad_A_weighted, out_shape, A->shape, A->stride);
        auto gb = ops::unbroadcast(grad_B_weighted, out_shape, B->shape, B->stride);
        return std::vector<std::vector<float>>{ga, gb};
      });
}

std::shared_ptr<core::Tensor> silu(const std::shared_ptr<core::Tensor>& X) {
  if (!X) throw std::invalid_argument("silu: input null");
  size_t n = X->numel();
  std::vector<float> C(n);
  for (size_t i = 0; i < n; ++i) {
    float x = X->ptr[i];
    float sig = 1.0f / (1.0f + std::exp(-x));
    C[i] = x * sig;
  }
  return core::make_op_output(
      {X},
      std::move(C),
      X->shape,
      [X](const std::vector<float>& grad_out) {
        std::vector<float> g(X->numel());
        for (size_t i = 0; i < X->numel(); ++i) {
          float x = X->ptr[i];
          float sig = 1.0f / (1.0f + std::exp(-x));
          g[i] = grad_out[i] * (sig + x * sig * (1.0f - sig));
        }
        return std::vector<std::vector<float>>{g};
      });
}

std::shared_ptr<core::Tensor> gelu(const std::shared_ptr<core::Tensor>& X) {
  if (!X) throw std::invalid_argument("gelu: input null");
  size_t n = X->numel();
  std::vector<float> C(n);
  const float sqrt_2_over_pi = 0.7978845608028654f;
  const float coef = 0.044715f;
  for (size_t i = 0; i < n; ++i) {
    float x = X->ptr[i];
    float x2 = x * x;
    float x3 = x2 * x;
    float inner = sqrt_2_over_pi * (x + coef * x3);
    float tanh_inner = std::tanh(inner);
    C[i] = 0.5f * x * (1.0f + tanh_inner);
  }
  return core::make_op_output(
      {X},
      std::move(C),
      X->shape,
      [X](const std::vector<float>& grad_out) {
        const float sqrt_2_over_pi = 0.7978845608028654f;
        const float coef = 0.044715f;
        std::vector<float> g(X->numel());
        for (size_t i = 0; i < X->numel(); ++i) {
          float x = X->ptr[i];
          float x2 = x * x;
          float inner = sqrt_2_over_pi * (x + coef * x * x2);
          float tanh_inner = std::tanh(inner);
          float d_tanh = (1.0f - tanh_inner * tanh_inner) *
                         sqrt_2_over_pi * (1.0f + 3.0f * coef * x2);
          g[i] = grad_out[i] * (0.5f * (1.0f + tanh_inner) + 0.5f * x * d_tanh);
        }
        return std::vector<std::vector<float>>{g};
      });
}

// Kurangi grad (dengan shape grad_shape) ke target_shape, mengakomodasi broadcast.
// target_stride adalah stride tensor target di ruang koordinat aslinya.
std::vector<float> unbroadcast(
    const std::vector<float>& grad,
    const std::vector<size_t>& grad_shape,
    const std::vector<size_t>& target_shape,
    const std::vector<size_t>& target_stride) {
  size_t tnumel = 1;
  for (size_t d : target_shape) tnumel *= d;
  std::vector<float> out(tnumel, 0.0f);

  size_t gnumel = grad.size();
  std::vector<size_t> gcoord(grad_shape.size());
  size_t diff = grad_shape.size() - target_shape.size();

  for (size_t i = 0; i < gnumel; ++i) {
    // Dekomposisi indeks grad ke koordinat.
    size_t r = i;
    for (size_t d = grad_shape.size(); d-- > 0;) {
      gcoord[d] = r % grad_shape[d];
      r /= grad_shape[d];
    }
    size_t t_index = 0;
    bool mismatch = false;
    for (size_t d = 0; d < target_shape.size(); ++d) {
      size_t gd_idx = diff + d;
      size_t g_dim = grad_shape[gd_idx];
      size_t t_dim = target_shape[d];
      if (t_dim != 1 && g_dim != 1 && g_dim != t_dim) {
        mismatch = true;
        break;
      }
      if (t_dim != 1) {
        t_index += gcoord[gd_idx] * target_stride[d];
      }
    }
    if (!mismatch) {
      out[t_index] += grad[i];
    }
  }
  return out;
}

}  // namespace rak::ops
