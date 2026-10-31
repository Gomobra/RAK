#ifndef RAK_OPS_SCAN_H
#define RAK_OPS_SCAN_H

#include "rak/core/tensor.h"
#include "rak/core/autograd.h"

namespace rak::ops {

// Scan rekuren linear bergerbang.
// Input X shape [batch, seq, d]. State shape [batch, d].
// Mode:
//   "parallel": latih, semua langkah dihitung sekaligus via parallel scan (prefix sum).
//   "sequential": generasi, satu langkah per waktu.
//   "forward" / "backward": arah eksekusi.
// Persamaan: h_t = h_{t-1} * decay + x_t  (decay adalah skalar per fitur, shape [d])
std::shared_ptr<core::Tensor> recurrent_scan(
    const std::shared_ptr<core::Tensor>& X,
    const std::shared_ptr<core::Tensor>& decay,   // shape [d], nilai (0,1)
    const std::string& mode = "parallel",
    const std::string& direction = "forward");

// Backward scan: diberikan grad_out [batch, seq, d], hitung grad_X dan grad_decay.
std::pair<std::vector<float>, std::vector<float>> recurrent_scan_backward(
    const std::shared_ptr<core::Tensor>& X,
    const std::shared_ptr<core::Tensor>& decay,
    const std::vector<float>& grad_out,
    const std::string& direction);

}  // namespace rak::ops

#endif  // RAK_OPS_SCAN_H
