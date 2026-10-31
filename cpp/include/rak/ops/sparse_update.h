#ifndef RAK_OPS_SPARSE_UPDATE_H
#define RAK_OPS_SPARSE_UPDATE_H

#include "rak/core/tensor.h"
#include "rak/core/autograd.h"

namespace rak::ops {

// Update sparse (scatter): hanya slot terpilih (indices) yang diperbarui.
// slots: [n_slot, d], indices: [n_update] (indeks slot), updates: [n_update, d].
// Output: salinan slots dengan slot terpilih ditambahkan (atau diganti) oleh updates.
// mode = "add" (accumulate) atau "replace".
std::shared_ptr<core::Tensor> sparse_update(
    const std::shared_ptr<core::Tensor>& slots,
    const std::shared_ptr<core::Tensor>& indices,  // 1D int (disimpan sebagai float)
    const std::shared_ptr<core::Tensor>& updates,
    const std::string& mode = "add");

// Backward sparse update: grad_slots dan grad_updates.
std::pair<std::vector<float>, std::vector<float>> sparse_update_backward(
    const std::shared_ptr<core::Tensor>& slots,
    const std::shared_ptr<core::Tensor>& indices,
    const std::string& mode,
    const std::vector<float>& grad_out);

}  // namespace rak::ops

#endif  // RAK_OPS_SPARSE_UPDATE_H
