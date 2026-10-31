#include "rak/ops/sparse_lookup.h"

#include <stdexcept>
#include <cmath>
#include <vector>
#include <algorithm>
#include <queue>

namespace rak::ops {

SparseLookupResult sparse_lookup_topk(
    const std::shared_ptr<core::Tensor>& query,
    const std::shared_ptr<core::Tensor>& keys,
    size_t k) {
  if (!query || !keys) throw std::invalid_argument("sparse_lookup_topk: input null");
  if (query->ndim() != 3) throw std::invalid_argument("sparse_lookup_topk: query harus 3D [batch, n_query, d]");
  if (keys->ndim() != 2) throw std::invalid_argument("sparse_lookup_topk: keys harus 2D [n_slot, d]");
  size_t batch = query->shape[0];
  size_t n_query = query->shape[1];
  size_t d = query->shape[2];
  size_t n_slot = keys->shape[0];
  if (keys->shape[1] != d) {
    throw std::invalid_argument("sparse_lookup_topk: dimensi keys tidak cocok dengan query");
  }
  if (k == 0 || k > n_slot) k = n_slot;

  SparseLookupResult result;
  result.values = std::make_shared<core::Tensor>(
      std::vector<size_t>{batch, n_query, k, d}, std::vector<float>(batch * n_query * k * d, 0.0f));
  result.indices.assign(batch, std::vector<std::vector<int64_t>>(n_query, std::vector<int64_t>(k, -1)));

  for (size_t b = 0; b < batch; ++b) {
    for (size_t q = 0; q < n_query; ++q) {
      // Hitung similarity (dot product) dengan semua keys.
      std::vector<std::pair<float, size_t>> scores(n_slot);
      for (size_t s = 0; s < n_slot; ++s) {
        float sim = 0.0f;
        for (size_t i = 0; i < d; ++i) {
          sim += query->ptr[((b * n_query + q) * d) + i] * keys->ptr[s * d + i];
        }
        scores[s] = {sim, s};
      }
      // Pilih top-k (gunakan partial sort).
      size_t ntop = std::min(k, n_slot);
      std::partial_sort(scores.begin(), scores.begin() + ntop, scores.end(),
                        [](const auto& a, const auto& b) { return a.first > b.first; });
      for (size_t t = 0; t < ntop; ++t) {
        size_t s = scores[t].second;
        result.indices[b][q][t] = static_cast<int64_t>(s);
        for (size_t i = 0; i < d; ++i) {
          result.values->ptr[((((b * n_query + q) * k) + t) * d) + i] =
              keys->ptr[s * d + i];
        }
      }
    }
  }
  return result;
}

std::pair<std::vector<float>, std::vector<float>> sparse_lookup_backward(
    const std::shared_ptr<core::Tensor>& query,
    const std::shared_ptr<core::Tensor>& keys,
    size_t k,
    const SparseLookupResult& result,
    const std::vector<float>& grad_values) {
  size_t batch = query->shape[0];
  size_t n_query = query->shape[1];
  size_t d = query->shape[2];
  size_t n_slot = keys->shape[0];
  if (k == 0 || k > n_slot) k = n_slot;

  std::vector<float> grad_query(batch * n_query * d, 0.0f);
  std::vector<float> grad_keys(n_slot * d, 0.0f);

  // grad_values: [batch, n_query, k, d]
  // grad_query[b,q,i] += sum_t grad_values[b,q,t,i]  (query terlibat dalam pemilihan,
  // namun karena top-k non-differentiable terhadap indeks, kita hanya mengalirkan grad
  // melalui nilai slot yang dipilih ke query dengan bobot = 1/k atau 1.0; di sini kita
  // mengalirkan grad ke query sebaris dengan nilai yang diambil — sederhana: grad_query
  // menerima grad yang sama dari setiap slot yang dipilih, dibagi k).
  for (size_t b = 0; b < batch; ++b) {
    for (size_t q = 0; q < n_query; ++q) {
      for (size_t t = 0; t < k; ++t) {
        int64_t s = result.indices[b][q][t];
        if (s < 0) continue;
        for (size_t i = 0; i < d; ++i) {
          size_t gv_idx = ((((b * n_query + q) * k) + t) * d) + i;
          float g = grad_values[gv_idx] / static_cast<float>(k);
          grad_query[((b * n_query + q) * d) + i] += g;
          grad_keys[s * d + i] += g;
        }
      }
    }
  }

  return {grad_query, grad_keys};
}

}  // namespace rak::ops
