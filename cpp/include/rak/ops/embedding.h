#ifndef RAK_OPS_EMBEDDING_H
#define RAK_OPS_EMBEDDING_H

#include "rak/core/tensor.h"
#include "rak/core/autograd.h"

namespace rak::ops {

// Embedding gather: X berisi indeks (int) di buffer float (asumsikan indeks disimpan
// sebagai float), output shape = X.shape + [embedding_dim].
// weight shape = [vocab_size, embedding_dim].
std::shared_ptr<core::Tensor> embedding(
    const std::shared_ptr<core::Tensor>& X,
    const std::shared_ptr<core::Tensor>& weight);

// Scatter-add backward untuk embedding: grad_weight[idx] += grad_out.
std::vector<float> embedding_backward(
    const std::shared_ptr<core::Tensor>& X,
    const std::shared_ptr<core::Tensor>& weight,
    const std::vector<float>& grad_out);

}  // namespace rak::ops

#endif  // RAK_OPS_EMBEDDING_H
