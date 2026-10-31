#include "rak/ops/gemm.h"

#include <stdexcept>

namespace rak::ops {

std::shared_ptr<core::Tensor> gemm(
    const std::shared_ptr<core::Tensor>& A,
    const std::shared_ptr<core::Tensor>& B) {
  if (!A || !B) throw std::invalid_argument("gemm: input tensor null");
  if (A->ndim() != 2 || B->ndim() != 2) {
    throw std::invalid_argument("gemm: hanya mendukung matriks 2D");
  }
  size_t M = A->shape[0];
  size_t K = A->shape[1];
  size_t K2 = B->shape[0];
  size_t N = B->shape[1];
  if (K != K2) {
    throw std::invalid_argument("gemm: dimensi K tidak cocok (A: " + std::to_string(K) +
                                ", B: " + std::to_string(K2) + ")");
  }

  std::vector<float> C(M * N, 0.0f);

  // Loop order ikj: B diakses stride-1 (kontigu), C di-update per baris.
  // Cocok untuk matriks kecil hingga menengah di CPU tanpa SIMD.
  for (size_t i = 0; i < M; ++i) {
    for (size_t k = 0; k < K; ++k) {
      float a = A->at({i, k});
      if (a == 0.0f) continue;  // sparsity-friendly skip
      size_t base_c = i * N;
      size_t base_b = k * N;
      for (size_t j = 0; j < N; ++j) {
        C[base_c + j] += a * B->ptr[base_b + j];
      }
    }
  }

  auto out = core::make_op_output(
      {A, B},
      std::move(C),
      {M, N},
      [A, B, M, K, N](const std::vector<float>& grad_C)
          -> std::vector<std::vector<float>> {
        auto [ga, gb] = ops::gemm_backward(A, B, grad_C);
        return {std::move(ga), std::move(gb)};
      });
  return out;
}

std::pair<std::vector<float>, std::vector<float>> gemm_backward(
    const std::shared_ptr<core::Tensor>& A,
    const std::shared_ptr<core::Tensor>& B,
    const std::vector<float>& grad_C) {
  size_t M = A->shape[0];
  size_t K = A->shape[1];
  size_t N = B->shape[1];

  std::vector<float> grad_A(M * K, 0.0f);
  std::vector<float> grad_B(K * N, 0.0f);

  // grad_A = grad_C (MxN) @ B^T (NxK)  -> grad_A[i,k] = sum_j grad_C[i,j] * B[k,j]
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      float gc = grad_C[i * N + j];
      if (gc == 0.0f) continue;
      size_t base_b = j * K;  // B^T[j,k] = B[k,j], baris j di B^T adalah kolom j di B
      // B disimpan KxN; elemen B[k,j] berada pada [k * N + j].
      // Untuk grad_A, kita butuh B[k,j] untuk semua k: akses tidak kontigu,
      // tapi cukup untuk implementasi awal.
      for (size_t k = 0; k < K; ++k) {
        grad_A[i * K + k] += gc * B->ptr[k * N + j];
      }
    }
  }

  // grad_B = A^T (KxM) @ grad_C (MxN)  -> grad_B[k,j] = sum_i A[i,k] * grad_C[i,j]
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      float gc = grad_C[i * N + j];
      if (gc == 0.0f) continue;
      for (size_t k = 0; k < K; ++k) {
        grad_B[k * N + j] += A->ptr[i * K + k] * gc;
      }
    }
  }

  return {grad_A, grad_B};
}

}  // namespace rak::ops
