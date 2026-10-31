#ifndef RAK_OPS_GEMM_H
#define RAK_OPS_GEMM_H

#include "rak/core/tensor.h"
#include "rak/core/autograd.h"

namespace rak::ops {

// GEMM tiled (CPU): C = A * B, dengan A (MxK), B (KxN), C (MxN). Row-major.
// Implementasi cache-friendly (loop order ikj) dengan fallback skalar.
// Di masa depan akan ada varian NEON/AVX2 di sini berdasarkan RAK_HAVE_NEON / RAK_HAVE_AVX2.
std::shared_ptr<core::Tensor> gemm(
    const std::shared_ptr<core::Tensor>& A,
    const std::shared_ptr<core::Tensor>& B);

// Backward GEMM: diberikan grad_C (MxN), hitung grad_A = grad_C @ B^T (MxK)
// dan grad_B = A^T @ grad_C (KxN).
std::pair<std::vector<float>, std::vector<float>> gemm_backward(
    const std::shared_ptr<core::Tensor>& A,
    const std::shared_ptr<core::Tensor>& B,
    const std::vector<float>& grad_C);

}  // namespace rak::ops

#endif  // RAK_OPS_GEMM_H
