#ifndef RAK_OPS_LOCAL_ATTENTION_H
#define RAK_OPS_LOCAL_ATTENTION_H

#include "rak/core/tensor.h"
#include "rak/core/autograd.h"

namespace rak::ops {

// Attention jendela geser (sliding window local attention).
// Q: [batch, seq, d], K: [batch, seq, d], V: [batch, seq, d].
// Untuk setiap posisi i, hanya attend ke jendela [i - window, i + window].
// Output shape = [batch, seq, d].
std::shared_ptr<core::Tensor> local_attention(
    const std::shared_ptr<core::Tensor>& Q,
    const std::shared_ptr<core::Tensor>& K,
    const std::shared_ptr<core::Tensor>& V,
    size_t window);

// Backward local attention.
std::tuple<std::vector<float>, std::vector<float>, std::vector<float>>
local_attention_backward(
    const std::shared_ptr<core::Tensor>& Q,
    const std::shared_ptr<core::Tensor>& K,
    const std::shared_ptr<core::Tensor>& V,
    size_t window,
    const std::vector<float>& grad_out);

}  // namespace rak::ops

#endif  // RAK_OPS_LOCAL_ATTENTION_H
