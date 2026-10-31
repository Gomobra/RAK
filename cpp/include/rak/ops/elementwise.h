#ifndef RAK_OPS_ELEMENTWISE_H
#define RAK_OPS_ELEMENTWISE_H

#include "rak/core/tensor.h"
#include "rak/core/autograd.h"

#include <cmath>

namespace rak::ops {

// Add: C = A + B (dengan broadcast). A dan B harus broadcastable ke shape hasil.
std::shared_ptr<core::Tensor> add(
    const std::shared_ptr<core::Tensor>& A,
    const std::shared_ptr<core::Tensor>& B);

// Mul: C = A * B (dengan broadcast).
std::shared_ptr<core::Tensor> mul(
    const std::shared_ptr<core::Tensor>& A,
    const std::shared_ptr<core::Tensor>& B);

// SiLU (x * sigmoid(x)).
std::shared_ptr<core::Tensor> silu(const std::shared_ptr<core::Tensor>& X);

// GELU (aproksimasi tanh).
std::shared_ptr<core::Tensor> gelu(const std::shared_ptr<core::Tensor>& X);

// Backward bantu untuk broadcast: kurangi grad output ke shape input.
std::vector<float> unbroadcast(
    const std::vector<float>& grad,
    const std::vector<size_t>& grad_shape,
    const std::vector<size_t>& target_shape,
    const std::vector<size_t>& target_stride);

}  // namespace rak::ops

#endif  // RAK_OPS_ELEMENTWISE_H
