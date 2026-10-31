#include "rak/ops/scan.h"

#include <stdexcept>
#include <cmath>
#include <numeric>

namespace rak::ops {

std::shared_ptr<core::Tensor> recurrent_scan(
    const std::shared_ptr<core::Tensor>& X,
    const std::shared_ptr<core::Tensor>& decay,
    const std::string& mode,
    const std::string& direction) {
  if (!X || !decay) throw std::invalid_argument("recurrent_scan: input null");
  if (X->ndim() != 3) throw std::invalid_argument("recurrent_scan: X harus 3D [batch, seq, d]");
  if (decay->ndim() != 1) throw std::invalid_argument("recurrent_scan: decay harus 1D [d]");
  size_t batch = X->shape[0];
  size_t seq = X->shape[1];
  size_t d = X->shape[2];
  if (decay->numel() != d) {
    throw std::invalid_argument("recurrent_scan: panjang decay harus sama dengan d");
  }

  std::string m = mode;
  std::string dir = direction;
  if (m != "parallel" && m != "sequential") {
    throw std::invalid_argument("recurrent_scan: mode harus 'parallel' atau 'sequential'");
  }
  if (dir != "forward" && dir != "backward") {
    throw std::invalid_argument("recurrent_scan: direction harus 'forward' atau 'backward'");
  }

  std::vector<float> out(batch * seq * d, 0.0f);

  // Simpan state per langkah untuk backward.
  std::vector<std::vector<float>> states(seq + 1, std::vector<float>(batch * d, 0.0f));

  if (dir == "forward") {
    for (size_t b = 0; b < batch; ++b) {
      for (size_t i = 0; i < d; ++i) states[0][b * d + i] = 0.0f;
      for (size_t t = 0; t < seq; ++t) {
        for (size_t i = 0; i < d; ++i) {
          float x = X->ptr[((b * seq + t) * d) + i];
          float h_prev = states[t][b * d + i];
          float dec = decay->ptr[i];
          float h = h_prev * dec + x;
          states[t + 1][b * d + i] = h;
          out[((b * seq + t) * d) + i] = h;
        }
      }
    }
  } else {
    // backward: h_t = h_{t+1} * decay + x_t (dari akhir ke awal)
    for (size_t b = 0; b < batch; ++b) {
      for (size_t i = 0; i < d; ++i) states[seq][b * d + i] = 0.0f;
      for (size_t t = seq; t-- > 0;) {
        for (size_t i = 0; i < d; ++i) {
          float x = X->ptr[((b * seq + t) * d) + i];
          float h_next = states[t + 1][b * d + i];
          float dec = decay->ptr[i];
          float h = h_next * dec + x;
          states[t][b * d + i] = h;
          out[((b * seq + t) * d) + i] = h;
        }
      }
    }
  }

  return core::make_op_output(
      {X, decay},
      std::move(out),
      {batch, seq, d},
      [X, decay, dir](const std::vector<float>& grad_out)
          -> std::vector<std::vector<float>> {
        auto [gx, gd] = ops::recurrent_scan_backward(X, decay, grad_out, dir);
        return {std::move(gx), std::move(gd)};
      });
}

std::pair<std::vector<float>, std::vector<float>> recurrent_scan_backward(
    const std::shared_ptr<core::Tensor>& X,
    const std::shared_ptr<core::Tensor>& decay,
    const std::vector<float>& grad_out,
    const std::string& direction) {
  size_t batch = X->shape[0];
  size_t seq = X->shape[1];
  size_t d = X->shape[2];

  std::vector<float> grad_X(batch * seq * d, 0.0f);
  std::vector<float> grad_decay(d, 0.0f);

  // Rekonstruksi state (sama seperti forward, karena linear).
  std::vector<std::vector<float>> states(seq + 1, std::vector<float>(batch * d, 0.0f));
  if (direction == "forward") {
    for (size_t b = 0; b < batch; ++b) {
      for (size_t i = 0; i < d; ++i) states[0][b * d + i] = 0.0f;
      for (size_t t = 0; t < seq; ++t) {
        for (size_t i = 0; i < d; ++i) {
          float x = X->ptr[((b * seq + t) * d) + i];
          float h_prev = states[t][b * d + i];
          float dec = decay->ptr[i];
          states[t + 1][b * d + i] = h_prev * dec + x;
        }
      }
    }
  } else {
    for (size_t b = 0; b < batch; ++b) {
      for (size_t i = 0; i < d; ++i) states[seq][b * d + i] = 0.0f;
      for (size_t t = seq; t-- > 0;) {
        for (size_t i = 0; i < d; ++i) {
          float x = X->ptr[((b * seq + t) * d) + i];
          float h_next = states[t + 1][b * d + i];
          float dec = decay->ptr[i];
          states[t][b * d + i] = h_next * dec + x;
        }
      }
    }
  }

  // Backprop melalui waktu.
  // grad h_t diterima dari grad_out[t] + grad h_{t+1} * decay (forward) atau
  // grad h_{t-1} * decay (backward).
  if (direction == "forward") {
    // Implementasi sederhana dan benar: iterasi dari belakang ke depan.
    std::vector<float> grad_h_next(batch * d, 0.0f);
    for (size_t t = seq; t-- > 0;) {
      for (size_t b = 0; b < batch; ++b) {
        for (size_t i = 0; i < d; ++i) {
          size_t idx = ((b * seq + t) * d) + i;
          float go = grad_out[idx];
          float gh = go + grad_h_next[b * d + i];  // grad h_t = go + grad h_{t+1}
          // grad x_t = gh
          grad_X[idx] += gh;
          // grad decay_i += grad h_{t+1} * h_t (karena h_{t+1} = h_t * decay + x_{t+1})
          // Namun untuk h_t = h_{t-1} * decay + x_t: grad decay += h_{t-1} * gh
          float h_prev = (t == 0) ? 0.0f : states[t][b * d + i];
          grad_decay[i] += h_prev * gh;
          grad_h_next[b * d + i] = gh * decay->ptr[i];  // grad h_{t-1} = gh * decay
        }
      }
    }
  } else {
    std::vector<float> grad_h_prev(batch * d, 0.0f);
    for (size_t t = 0; t < seq; ++t) {
      for (size_t b = 0; b < batch; ++b) {
        for (size_t i = 0; i < d; ++i) {
          size_t idx = ((b * seq + t) * d) + i;
          float go = grad_out[idx];
          float gh = go + grad_h_prev[b * d + i];
          grad_X[idx] += gh;
          float h_next = (t == seq - 1) ? 0.0f : states[t + 1][b * d + i];
          grad_decay[i] += h_next * gh;
          grad_h_prev[b * d + i] = gh * decay->ptr[i];
        }
      }
    }
  }

  return {grad_X, grad_decay};
}

}  // namespace rak::ops
