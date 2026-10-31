#ifndef RAK_OPS_REDUCE_H
#define RAK_OPS_REDUCE_H

#include "rak/core/tensor.h"
#include "rak/core/autograd.h"

namespace rak::ops {

// Sum along axis. Jika axis = -1, reduce semua elemen (skalar output shape {1}).
std::shared_ptr<core::Tensor> sum(
    const std::shared_ptr<core::Tensor>& X, int64_t axis = -1);

// Mean along axis.
std::shared_ptr<core::Tensor> mean(
    const std::shared_ptr<core::Tensor>& X, int64_t axis = -1);

// Max along axis.
std::shared_ptr<core::Tensor> max(
    const std::shared_ptr<core::Tensor>& X, int64_t axis = -1);

}  // namespace rak::ops

#endif  // RAK_OPS_REDUCE_H
