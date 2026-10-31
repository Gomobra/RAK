#ifndef RAK_OPS_NORM_H
#define RAK_OPS_NORM_H

#include "rak/core/tensor.h"
#include "rak/core/autograd.h"

namespace rak::ops {

// RMSNorm: y = x / RMS(x) * weight, dengan RMS = sqrt(mean(x^2) + eps).
// weight berukuran sama dengan axis terakhir (d_model).
std::shared_ptr<core::Tensor> rms_norm(
    const std::shared_ptr<core::Tensor>& X,
    const std::shared_ptr<core::Tensor>& weight,
    float eps = 1e-5f);

// Softmax stabil secara numerik along axis terakhir.
std::shared_ptr<core::Tensor> softmax(const std::shared_ptr<core::Tensor>& X);

// Log-softmax stabil secara numerik along axis terakhir.
std::shared_ptr<core::Tensor> log_softmax(const std::shared_ptr<core::Tensor>& X);

}  // namespace rak::ops

#endif  // RAK_OPS_NORM_H
