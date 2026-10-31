#ifndef RAK_OPS_SPARSE_LOOKUP_H
#define RAK_OPS_SPARSE_LOOKUP_H

#include "rak/core/tensor.h"
#include "rak/core/autograd.h"

namespace rak::ops {

// Lookup top-k kunci produk (product key sparse lookup).
// query: [batch, n_query, d_key], keys: [n_slot, d_key].
// Untuk setiap query, pilih top-k slot berdasarkan dot product similarity.
// Output: [batch, n_query, k, d_key] (nilai slot terpilih) dan indeks slot.
struct SparseLookupResult {
  std::shared_ptr<core::Tensor> values;   // [batch, n_query, k, d_key]
  std::vector<std::vector<std::vector<int64_t>>> indices;  // [batch, n_query, k]
};

SparseLookupResult sparse_lookup_topk(
    const std::shared_ptr<core::Tensor>& query,
    const std::shared_ptr<core::Tensor>& keys,
    size_t k);

// Backward sparse lookup: grad_query dan grad_keys (hanya slot terpilih yang menerima grad).
std::pair<std::vector<float>, std::vector<float>> sparse_lookup_backward(
    const std::shared_ptr<core::Tensor>& query,
    const std::shared_ptr<core::Tensor>& keys,
    size_t k,
    const SparseLookupResult& result,
    const std::vector<float>& grad_values);

}  // namespace rak::ops

#endif  // RAK_OPS_SPARSE_LOOKUP_H
