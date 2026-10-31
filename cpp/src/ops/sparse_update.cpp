#include "rak/ops/sparse_update.h"

#include <stdexcept>
#include <vector>

namespace rak::ops {

std::shared_ptr<core::Tensor> sparse_update(
    const std::shared_ptr<core::Tensor>& slots,
    const std::shared_ptr<core::Tensor>& indices,
    const std::shared_ptr<core::Tensor>& updates,
    const std::string& mode) {
  if (!slots || !indices || !updates) throw std::invalid_argument("sparse_update: input null");
  if (slots->ndim() != 2) throw std::invalid_argument("sparse_update: slots harus 2D [n_slot, d]");
  if (indices->ndim() != 1) throw std::invalid_argument("sparse_update: indices harus 1D [n_update]");
  if (updates->ndim() != 2) throw std::invalid_argument("sparse_update: updates harus 2D [n_update, d]");
  size_t n_slot = slots->shape[0];
  size_t d = slots->shape[1];
  size_t n_update = indices->numel();
  if (updates->shape[0] != n_update || updates->shape[1] != d) {
    throw std::invalid_argument("sparse_update: shape updates harus [n_update, d]");
  }
  if (mode != "add" && mode != "replace") {
    throw std::invalid_argument("sparse_update: mode harus 'add' atau 'replace'");
  }

  std::vector<int64_t> idx(n_update);
  for (size_t i = 0; i < n_update; ++i) {
    int64_t id = static_cast<int64_t>(indices->ptr[i]);
    if (id < 0 || id >= static_cast<int64_t>(n_slot)) {
      throw std::out_of_range("sparse_update: indeks slot di luar jangkauan");
    }
    idx[i] = id;
  }

  std::vector<float> out = slots->data;  // copy
  if (mode == "add") {
    for (size_t i = 0; i < n_update; ++i) {
      for (size_t j = 0; j < d; ++j) {
        out[idx[i] * d + j] += updates->ptr[i * d + j];
      }
    }
  } else {
    for (size_t i = 0; i < n_update; ++i) {
      for (size_t j = 0; j < d; ++j) {
        out[idx[i] * d + j] = updates->ptr[i * d + j];
      }
    }
  }

  return core::make_op_output(
      {slots, indices, updates},
      std::move(out),
      {n_slot, d},
      [slots, indices, mode, n_slot, d, n_update, idx = std::move(idx)](
          const std::vector<float>& grad_out) {
        // grad_slots: grad_out untuk slot yang tidak di-update; slot yang di-update
        // menerima grad_out jika mode="add" (karena out = slots + updates).
        std::vector<float> g_slots(n_slot * d, 0.0f);
        std::vector<float> g_updates(n_update * d, 0.0f);
        if (mode == "add") {
          for (size_t i = 0; i < n_slot * d; ++i) g_slots[i] = grad_out[i];
          for (size_t i = 0; i < n_update; ++i) {
            for (size_t j = 0; j < d; ++j) {
              g_updates[i * d + j] = grad_out[idx[i] * d + j];
            }
          }
        } else {
          // replace: grad_slots untuk slot yang tidak di-update = grad_out;
          // slot yang di-update tidak menerima grad dari slots (diganti).
          std::vector<bool> updated(n_slot, false);
          for (size_t i = 0; i < n_update; ++i) updated[idx[i]] = true;
          for (size_t i = 0; i < n_slot; ++i) {
            if (!updated[i]) {
              for (size_t j = 0; j < d; ++j) g_slots[i * d + j] = grad_out[i * d + j];
            }
          }
          for (size_t i = 0; i < n_update; ++i) {
            for (size_t j = 0; j < d; ++j) {
              g_updates[i * d + j] = grad_out[idx[i] * d + j];
            }
          }
        }
        return std::vector<std::vector<float>>{g_slots, g_updates};
      });
}

std::pair<std::vector<float>, std::vector<float>> sparse_update_backward(
    const std::shared_ptr<core::Tensor>& slots,
    const std::shared_ptr<core::Tensor>& indices,
    const std::string& mode,
    const std::vector<float>& grad_out) {
  (void)slots;
  (void)indices;
  (void)mode;
  std::vector<float> g(grad_out.size(), 0.0f);
  return {g, g};
}

}  // namespace rak::ops
