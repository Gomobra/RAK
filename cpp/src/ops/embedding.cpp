#include "rak/ops/embedding.h"

#include <stdexcept>
#include <vector>

namespace rak::ops {

std::shared_ptr<core::Tensor> embedding(
    const std::shared_ptr<core::Tensor>& X,
    const std::shared_ptr<core::Tensor>& weight) {
  if (!X || !weight) throw std::invalid_argument("embedding: input null");
  if (weight->ndim() != 2) {
    throw std::invalid_argument("embedding: weight harus matriks 2D [vocab, dim]");
  }
  size_t vocab = weight->shape[0];
  size_t dim = weight->shape[1];

  size_t xnumel = X->numel();
  std::vector<size_t> out_shape = X->shape;
  out_shape.push_back(dim);
  std::vector<float> out(xnumel * dim, 0.0f);

  // Simpan indeks untuk backward.
  std::vector<int64_t> indices(xnumel);
  for (size_t i = 0; i < xnumel; ++i) {
    int64_t idx = static_cast<int64_t>(X->ptr[i]);
    if (idx < 0 || idx >= static_cast<int64_t>(vocab)) {
      throw std::out_of_range("embedding: indeks di luar jangkauan vocab");
    }
    indices[i] = idx;
    for (size_t d = 0; d < dim; ++d) {
      out[i * dim + d] = weight->ptr[idx * dim + d];
    }
  }

  return core::make_op_output(
      {X, weight},
      std::move(out),
      out_shape,
      [X, weight, vocab, dim, indices = std::move(indices)](
          const std::vector<float>& grad_out) {
        // grad_weight[idx] += grad_out[i, :]
        std::vector<float> gW(vocab * dim, 0.0f);
        size_t xnumel = X->numel();
        for (size_t i = 0; i < xnumel; ++i) {
          int64_t idx = indices[i];
          for (size_t d = 0; d < dim; ++d) {
            gW[idx * dim + d] += grad_out[i * dim + d];
          }
        }
        // X tidak menerima grad (indeks tidak differentiable).
        return std::vector<std::vector<float>>{{}, gW};
      });
}

std::vector<float> embedding_backward(
    const std::shared_ptr<core::Tensor>& X,
    const std::shared_ptr<core::Tensor>& weight,
    const std::vector<float>& grad_out) {
  (void)X;
  size_t vocab = weight->shape[0];
  size_t dim = weight->shape[1];
  std::vector<float> gW(vocab * dim, 0.0f);
  return gW;
}

}  // namespace rak::ops
