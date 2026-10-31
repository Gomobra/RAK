#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "rak/core/tensor.h"
#include "rak/core/autograd.h"
#include "rak/ops/gemm.h"
#include "rak/ops/elementwise.h"
#include "rak/ops/reduce.h"
#include "rak/ops/norm.h"
#include "rak/ops/embedding.h"
#include "rak/ops/scan.h"
#include "rak/ops/local_attention.h"
#include "rak/ops/sparse_lookup.h"
#include "rak/ops/sparse_update.h"
#include "rak/testing/grad_check.h"

#include "rak/train/optimizer.h"
#include "rak/train/checkpoint.h"

#include <cstring>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace rak::core;
using namespace rak::ops;
using namespace rak::testing;
using namespace rak::train;

using Catch::Approx;
using Catch::Matchers::WithinAbs;

// ============================================================
// Helper: forward wrapper untuk gradient_check
// gradient_check mengharapkan forward_fn(inputs) -> loss skalar (numel==1)
// ============================================================

// ---------- Helpers: forward untuk masing-masing op, mengembalikan skalar ----------

static std::shared_ptr<Tensor> to_scalar(std::shared_ptr<Tensor> t) {
  while (t->numel() > 1) {
    t = sum(t, -1);
  }
  return t;
}

static std::shared_ptr<Tensor> fwd_add(const std::vector<std::shared_ptr<Tensor>>& ins) {
  return to_scalar(add(ins[0], ins[1]));
}

static std::shared_ptr<Tensor> fwd_mul(const std::vector<std::shared_ptr<Tensor>>& ins) {
  return to_scalar(mul(ins[0], ins[1]));
}

static std::shared_ptr<Tensor> fwd_silu(const std::vector<std::shared_ptr<Tensor>>& ins) {
  return to_scalar(silu(ins[0]));
}

static std::shared_ptr<Tensor> fwd_gelu(const std::vector<std::shared_ptr<Tensor>>& ins) {
  return to_scalar(gelu(ins[0]));
}

static std::shared_ptr<Tensor> fwd_gemm(const std::vector<std::shared_ptr<Tensor>>& ins) {
  return to_scalar(gemm(ins[0], ins[1]));
}

static std::shared_ptr<Tensor> fwd_sum(const std::vector<std::shared_ptr<Tensor>>& ins) {
  return to_scalar(sum(ins[0], -1));
}

static std::shared_ptr<Tensor> fwd_mean(const std::vector<std::shared_ptr<Tensor>>& ins) {
  return to_scalar(mean(ins[0], -1));
}

static std::shared_ptr<Tensor> fwd_max(const std::vector<std::shared_ptr<Tensor>>& ins) {
  return to_scalar(max(ins[0], -1));
}

static std::shared_ptr<Tensor> fwd_rms_norm(const std::vector<std::shared_ptr<Tensor>>& ins) {
  return to_scalar(rms_norm(ins[0], ins[1]));
}

static std::shared_ptr<Tensor> fwd_softmax(const std::vector<std::shared_ptr<Tensor>>& ins) {
  return to_scalar(softmax(ins[0]));
}

static std::shared_ptr<Tensor> fwd_log_softmax(const std::vector<std::shared_ptr<Tensor>>& ins) {
  return to_scalar(log_softmax(ins[0]));
}

static std::shared_ptr<Tensor> fwd_embedding(const std::vector<std::shared_ptr<Tensor>>& ins) {
  return to_scalar(embedding(ins[0], ins[1]));
}

static std::shared_ptr<Tensor> fwd_recurrent_scan(const std::vector<std::shared_ptr<Tensor>>& ins) {
  return to_scalar(recurrent_scan(ins[0], ins[1], "parallel", "forward"));
}

static std::shared_ptr<Tensor> fwd_local_attention(const std::vector<std::shared_ptr<Tensor>>& ins) {
  return to_scalar(local_attention(ins[0], ins[1], ins[2], 2));
}

static std::shared_ptr<Tensor> fwd_sparse_lookup(const std::vector<std::shared_ptr<Tensor>>& ins) {
  auto res = sparse_lookup_topk(ins[0], ins[1], 2);
  return to_scalar(res.values);
}

static std::shared_ptr<Tensor> fwd_sparse_update(const std::vector<std::shared_ptr<Tensor>>& ins) {
  return to_scalar(sparse_update(ins[0], ins[1], ins[2], "add"));
}

// ============================================================
// TENSOR TESTS
// ============================================================

TEST_CASE("Tensor: konstruksi dan at", "[tensor]") {
  Tensor t({2, 3}, true);
  REQUIRE(t.numel() == 6);
  REQUIRE(t.shape == std::vector<size_t>({2, 3}));
  t.at({0, 1}) = 4.0f;
  REQUIRE(t.at({0, 1}) == 4.0f);
  REQUIRE(t.at(0) == 0.0f);  // linear index 0 belum di-set
}

TEST_CASE("Tensor: reshape dan transpose view", "[tensor]") {
  Tensor t({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
  auto r = t.reshape({3, 2});
  REQUIRE(r.shape == std::vector<size_t>({3, 2}));
  REQUIRE(r.numel() == 6);
  REQUIRE(r.at({1, 0}) == 3.0f);

  auto tr = t.transpose(0, 1);
  REQUIRE(tr.shape == std::vector<size_t>({3, 2}));
  REQUIRE(tr.at({1, 0}) == 2.0f);
}

TEST_CASE("Tensor: edge case ukuran 0 dan 1", "[tensor][edge]") {
  Tensor t0({0}, true);
  REQUIRE(t0.numel() == 0);
  Tensor t1({1}, std::vector<float>{3.5f});
  REQUIRE(t1.numel() == 1);
  REQUIRE(t1.at(0) == 3.5f);
}

TEST_CASE("Tensor: at indeks di luar jangkauan melempar", "[tensor][edge][negative]") {
  Tensor t({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
  REQUIRE_THROWS_AS(t.at({2, 0}), std::out_of_range);
  REQUIRE_THROWS_AS(t.at(100), std::out_of_range);
}

// ============================================================
// AUTOGRAD TESTS
// ============================================================

TEST_CASE("Autograd: add backward vs numerical", "[autograd]") {
  auto a = std::make_shared<Tensor>(std::vector<size_t>{3},
      std::vector<float>{1.0f, 2.0f, 3.0f}, true);
  auto b = std::make_shared<Tensor>(std::vector<size_t>{3},
      std::vector<float>{4.0f, 5.0f, 6.0f}, true);
  auto res = gradient_check(fwd_add, {a, b});
  REQUIRE(res.size() == 2);
  REQUIRE(res[0].ok);
  REQUIRE(res[1].ok);
}

TEST_CASE("Autograd: mul backward vs numerical", "[autograd]") {
  auto a = std::make_shared<Tensor>(std::vector<size_t>{3},
      std::vector<float>{1.0f, 2.0f, 3.0f}, true);
  auto b = std::make_shared<Tensor>(std::vector<size_t>{3},
      std::vector<float>{4.0f, 5.0f, 6.0f}, true);
  auto res = gradient_check(fwd_mul, {a, b});
  REQUIRE(res.size() == 2);
  REQUIRE(res[0].ok);
  REQUIRE(res[1].ok);
}

TEST_CASE("Autograd: GEMM backward vs numerical", "[autograd][gemm]") {
  auto A = std::make_shared<Tensor>(std::vector<size_t>{2, 2},
      std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, true);
  auto B = std::make_shared<Tensor>(std::vector<size_t>{2, 2},
      std::vector<float>{5.0f, 6.0f, 7.0f, 8.0f}, true);
  auto res = gradient_check(fwd_gemm, {A, B}, 1e-4f, 1e-2f, 1e-2f);
  REQUIRE(res.size() == 2);
  REQUIRE(res[0].ok);
  REQUIRE(res[1].ok);
}

TEST_CASE("Autograd: SiLU backward vs numerical", "[autograd][elementwise]") {
  auto x = std::make_shared<Tensor>(std::vector<size_t>{4},
      std::vector<float>{-2.0f, -0.5f, 0.5f, 2.0f}, true);
  auto res = gradient_check(fwd_silu, {x});
  REQUIRE(res.size() == 1);
  REQUIRE(res[0].ok);
}

TEST_CASE("Autograd: GELU backward vs numerical", "[autograd][elementwise]") {
  auto x = std::make_shared<Tensor>(std::vector<size_t>{4},
      std::vector<float>{-2.0f, -0.5f, 0.5f, 2.0f}, true);
  auto res = gradient_check(fwd_gelu, {x});
  REQUIRE(res.size() == 1);
  REQUIRE(res[0].ok);
}

TEST_CASE("Ops: RMSNorm backward vs numerical", "[autograd][norm]") {
  auto X = std::make_shared<Tensor>(std::vector<size_t>{2, 3},
      std::vector<float>{1, 2, 3, 4, 5, 6}, true);
  auto W = std::make_shared<Tensor>(std::vector<size_t>{3},
      std::vector<float>{1.0f, 1.0f, 1.0f}, true);
  auto res = gradient_check(fwd_rms_norm, {X, W});
  REQUIRE(res.size() == 2);
  REQUIRE(res[0].ok);
  REQUIRE(res[1].ok);
}

TEST_CASE("Ops: softmax backward vs numerical", "[autograd][norm]") {
  auto X = std::make_shared<Tensor>(std::vector<size_t>{2, 3},
      std::vector<float>{1, 2, 3, 4, 5, 6}, true);
  auto res = gradient_check(fwd_softmax, {X});
  REQUIRE(res.size() == 1);
  REQUIRE(res[0].ok);
}

TEST_CASE("Ops: log_softmax backward vs numerical", "[autograd][norm]") {
  auto X = std::make_shared<Tensor>(std::vector<size_t>{2, 3},
      std::vector<float>{1, 2, 3, 4, 5, 6}, true);
  auto res = gradient_check(fwd_log_softmax, {X});
  REQUIRE(res.size() == 1);
  REQUIRE(res[0].ok);
}

TEST_CASE("Ops: embedding backward vs numerical", "[autograd][embedding]") {
  auto W = std::make_shared<Tensor>(std::vector<size_t>{4, 2},
      std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8}, true);
  auto X = std::make_shared<Tensor>(std::vector<size_t>{3},
      std::vector<float>{0.0f, 2.0f, 3.0f}, false);
  auto res = gradient_check(fwd_embedding, {X, W});
  REQUIRE(res.size() == 1);
  REQUIRE(res[0].ok);
}

TEST_CASE("Ops: recurrent_scan backward vs numerical", "[autograd][scan]") {
  auto X = std::make_shared<Tensor>(std::vector<size_t>{1, 3, 2},
      std::vector<float>{1, 2, 3, 4, 5, 6}, true);
  auto D = std::make_shared<Tensor>(std::vector<size_t>{2},
      std::vector<float>{0.5f, 0.5f}, true);
  auto res = gradient_check(fwd_recurrent_scan, {X, D});
  REQUIRE(res.size() == 2);
  REQUIRE(res[0].ok);
  REQUIRE(res[1].ok);
}

TEST_CASE("Ops: local_attention backward vs numerical", "[autograd][attention]") {
  auto Q = std::make_shared<Tensor>(std::vector<size_t>{1, 4, 3},
      std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, true);
  auto K = std::make_shared<Tensor>(std::vector<size_t>{1, 4, 3},
      std::vector<float>{1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1}, true);
  auto V = std::make_shared<Tensor>(std::vector<size_t>{1, 4, 3},
      std::vector<float>{10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120}, true);
  auto res = gradient_check(fwd_local_attention, {Q, K, V});
  REQUIRE(res.size() == 3);
  REQUIRE(res[0].ok);
  REQUIRE(res[1].ok);
  REQUIRE(res[2].ok);
}

TEST_CASE("Ops: sparse_lookup backward vs numerical", "[autograd][sparse]") {
  auto query = std::make_shared<Tensor>(std::vector<size_t>{1, 1, 3},
      std::vector<float>{1, 0, 0}, true);
  auto keys = std::make_shared<Tensor>(std::vector<size_t>{4, 3},
      std::vector<float>{1, 0, 0, 0, 1, 0, 0, 0, 1, 5, 0, 0}, true);
  auto res = gradient_check(fwd_sparse_lookup, {query, keys});
  REQUIRE(res.size() == 2);
  REQUIRE(res[0].ok);
  REQUIRE(res[1].ok);
}

TEST_CASE("Ops: sparse_update backward vs numerical", "[autograd][sparse]") {
  auto slots = std::make_shared<Tensor>(std::vector<size_t>{3, 2},
      std::vector<float>{1, 2, 3, 4, 5, 6}, true);
  auto indices = std::make_shared<Tensor>(std::vector<size_t>{2},
      std::vector<float>{0.0f, 2.0f}, false);
  auto updates = std::make_shared<Tensor>(std::vector<size_t>{2, 2},
      std::vector<float>{10, 20, 30, 40}, true);
  auto res = gradient_check(fwd_sparse_update, {slots, indices, updates});
  REQUIRE(res.size() == 2);  // indices tidak requires_grad
  REQUIRE(res[0].ok);
  REQUIRE(res[1].ok);
}

TEST_CASE("Ops: sum backward vs numerical", "[autograd][reduce]") {
  auto X = std::make_shared<Tensor>(std::vector<size_t>{2, 3},
      std::vector<float>{1, 2, 3, 4, 5, 6}, true);
  auto res = gradient_check(fwd_sum, {X});
  REQUIRE(res.size() == 1);
  REQUIRE(res[0].ok);
}

TEST_CASE("Ops: mean backward vs numerical", "[autograd][reduce]") {
  auto X = std::make_shared<Tensor>(std::vector<size_t>{2, 3},
      std::vector<float>{1, 2, 3, 4, 5, 6}, true);
  auto res = gradient_check(fwd_mean, {X});
  REQUIRE(res.size() == 1);
  REQUIRE(res[0].ok);
}

TEST_CASE("Ops: max backward vs numerical", "[autograd][reduce]") {
  auto X = std::make_shared<Tensor>(std::vector<size_t>{2, 3},
      std::vector<float>{1, 5, 3, 4, 2, 6}, true);
  auto res = gradient_check(fwd_max, {X});
  REQUIRE(res.size() == 1);
  REQUIRE(res[0].ok);
}

// ============================================================
// DETERMINISME (bit-per-bit)
// ============================================================

static void check_determinism_bitwise(
    const std::function<std::shared_ptr<Tensor>(
        const std::vector<std::shared_ptr<Tensor>>&)>& forward,
    const std::vector<std::shared_ptr<Tensor>>& inputs) {
  // Zero gradien sebelum setiap run (backward mengakumulasi ke grad).
  auto zero_grads = [&inputs]() {
    for (auto& in : inputs) {
      if (in->requires_grad) {
        std::fill(in->grad.begin(), in->grad.end(), 0.0f);
      }
    }
  };

  // Run 1
  g_autograd.reset();
  zero_grads();
  auto loss1 = forward(inputs);
  g_autograd.backward(loss1);

  std::vector<float> w1, g1;
  for (auto& in : inputs) {
    if (in->requires_grad) {
      w1.insert(w1.end(), in->ptr, in->ptr + in->numel());
      g1.insert(g1.end(), in->grad.begin(), in->grad.end());
    }
  }

  // Run 2 dengan input dan seed yang sama.
  g_autograd.reset();
  zero_grads();
  auto loss2 = forward(inputs);
  g_autograd.backward(loss2);

  std::vector<float> w2, g2;
  for (auto& in : inputs) {
    if (in->requires_grad) {
      w2.insert(w2.end(), in->ptr, in->ptr + in->numel());
      g2.insert(g2.end(), in->grad.begin(), in->grad.end());
    }
  }

  REQUIRE(w1.size() == w2.size());
  REQUIRE(g1.size() == g2.size());
  REQUIRE(std::memcmp(w1.data(), w2.data(), w1.size() * sizeof(float)) == 0);
  REQUIRE(std::memcmp(g1.data(), g2.data(), g1.size() * sizeof(float)) == 0);
}

// Buat input fresh untuk determinisme (karena backward mengubah grad, bukan ptr)
static std::vector<std::shared_ptr<Tensor>> det_add_inputs() {
  return {
    std::make_shared<Tensor>(std::vector<size_t>{3}, std::vector<float>{1.0f, 2.0f, 3.0f}, true),
    std::make_shared<Tensor>(std::vector<size_t>{3}, std::vector<float>{4.0f, 5.0f, 6.0f}, true),
  };
}

TEST_CASE("Determinisme: add forward+backward bit-per-bit", "[determinism]") {
  auto inputs = det_add_inputs();
  check_determinism_bitwise(fwd_add, inputs);
}

TEST_CASE("Determinisme: GEMM forward+backward bit-per-bit", "[determinism]") {
  auto inputs = std::vector<std::shared_ptr<Tensor>>{
    std::make_shared<Tensor>(std::vector<size_t>{2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6}, true),
    std::make_shared<Tensor>(std::vector<size_t>{3, 2}, std::vector<float>{7, 8, 9, 10, 11, 12}, true),
  };
  check_determinism_bitwise(fwd_gemm, inputs);
}

TEST_CASE("Determinisme: softmax forward+backward bit-per-bit", "[determinism]") {
  auto inputs = std::vector<std::shared_ptr<Tensor>>{
    std::make_shared<Tensor>(std::vector<size_t>{2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6}, true),
  };
  check_determinism_bitwise(fwd_softmax, inputs);
}

TEST_CASE("Determinisme: local_attention forward+backward bit-per-bit", "[determinism]") {
  auto inputs = std::vector<std::shared_ptr<Tensor>>{
    std::make_shared<Tensor>(std::vector<size_t>{1, 4, 3},
      std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, true),
    std::make_shared<Tensor>(std::vector<size_t>{1, 4, 3},
      std::vector<float>{1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1}, true),
    std::make_shared<Tensor>(std::vector<size_t>{1, 4, 3},
      std::vector<float>{10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120}, true),
  };
  check_determinism_bitwise(fwd_local_attention, inputs);
}

// ============================================================
// EDGE CASE
// ============================================================

TEST_CASE("Edge: GEMM shape tidak cocok melempar", "[edge][gemm][negative]") {
  auto A = std::make_shared<Tensor>(std::vector<size_t>{2, 3},
      std::vector<float>{1, 2, 3, 4, 5, 6}, false);
  auto B = std::make_shared<Tensor>(std::vector<size_t>{4, 2},
      std::vector<float>{7, 8, 9, 10, 11, 12, 13, 14}, false);
  REQUIRE_THROWS_AS(gemm(A, B), std::invalid_argument);
}

TEST_CASE("Edge: add broadcast shape tidak cocok melempar", "[edge][elementwise][negative]") {
  auto A = std::make_shared<Tensor>(std::vector<size_t>{2, 3},
      std::vector<float>{1, 2, 3, 4, 5, 6}, false);
  auto B = std::make_shared<Tensor>(std::vector<size_t>{4, 5},
      std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}, false);
  REQUIRE_THROWS_AS(add(A, B), std::invalid_argument);
}

TEST_CASE("Edge: reduce axis di luar jangkauan melempar", "[edge][reduce][negative]") {
  auto X = std::make_shared<Tensor>(std::vector<size_t>{2, 3},
      std::vector<float>{1, 2, 3, 4, 5, 6}, false);
  REQUIRE_THROWS_AS(sum(X, 5), std::invalid_argument);
  REQUIRE_THROWS_AS(mean(X, -3), std::invalid_argument);
  REQUIRE_THROWS_AS(max(X, 10), std::invalid_argument);
}

TEST_CASE("Edge: embedding indeks di luar jangkauan vocab melempar", "[edge][embedding][negative]") {
  auto W = std::make_shared<Tensor>(std::vector<size_t>{4, 2},
      std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8}, false);
  auto X = std::make_shared<Tensor>(std::vector<size_t>{1},
      std::vector<float>{99.0f}, false);
  REQUIRE_THROWS_AS(embedding(X, W), std::out_of_range);
}

TEST_CASE("Edge: sparse_lookup k=0 dan k>n_slot", "[edge][sparse][negative]") {
  auto query = std::make_shared<Tensor>(std::vector<size_t>{1, 1, 3},
      std::vector<float>{1, 0, 0}, false);
  auto keys = std::make_shared<Tensor>(std::vector<size_t>{4, 3},
      std::vector<float>{1, 0, 0, 0, 1, 0, 0, 0, 1, 5, 0, 0}, false);
  // k=0 dan k>n_slot tidak melempar, melambatkan ke n_slot (per implementasi)
  auto res0 = sparse_lookup_topk(query, keys, 0);
  REQUIRE(res0.values->numel() == 1 * 1 * 4 * 3);
  auto resBig = sparse_lookup_topk(query, keys, 100);
  REQUIRE(resBig.values->numel() == 1 * 1 * 4 * 3);
}

TEST_CASE("Edge: sparse_update indeks di luar jangkauan melempar", "[edge][sparse][negative]") {
  auto slots = std::make_shared<Tensor>(std::vector<size_t>{3, 2},
      std::vector<float>{1, 2, 3, 4, 5, 6}, false);
  auto indices = std::make_shared<Tensor>(std::vector<size_t>{1},
      std::vector<float>{5.0f}, false);
  auto updates = std::make_shared<Tensor>(std::vector<size_t>{1, 2},
      std::vector<float>{10, 20}, false);
  REQUIRE_THROWS_AS(sparse_update(slots, indices, updates, "add"), std::out_of_range);
}

TEST_CASE("Edge: recurrent_scan mode tidak valid melempar", "[edge][scan][negative]") {
  auto X = std::make_shared<Tensor>(std::vector<size_t>{1, 3, 2},
      std::vector<float>{1, 2, 3, 4, 5, 6}, false);
  auto D = std::make_shared<Tensor>(std::vector<size_t>{2},
      std::vector<float>{0.5f, 0.5f}, false);
  REQUIRE_THROWS_AS(recurrent_scan(X, D, "invalid", "forward"), std::invalid_argument);
}

TEST_CASE("Edge: input null ke op melempar", "[edge][negative]") {
  REQUIRE_THROWS_AS(add(nullptr, nullptr), std::invalid_argument);
  REQUIRE_THROWS_AS(mul(nullptr, nullptr), std::invalid_argument);
  REQUIRE_THROWS_AS(gemm(nullptr, nullptr), std::invalid_argument);
  REQUIRE_THROWS_AS(softmax(nullptr), std::invalid_argument);
}

TEST_CASE("Edge: softmax/log_softmax dengan NaN/Inf", "[edge][norm][numerik]") {
  auto Xnan = std::make_shared<Tensor>(std::vector<size_t>{1, 3},
      std::vector<float>{std::numeric_limits<float>::quiet_NaN(), 1.0f, 2.0f}, false);
  // Tidak boleh crash; hasil boleh NaN karena input NaN
  auto out_nan = softmax(Xnan);
  REQUIRE(out_nan->numel() == 3);

  auto Xinf = std::make_shared<Tensor>(std::vector<size_t>{1, 3},
      std::vector<float>{std::numeric_limits<float>::infinity(), 1.0f, 2.0f}, false);
  auto out_inf = softmax(Xinf);
  REQUIRE(out_inf->numel() == 3);
}

TEST_CASE("Edge: tensor ukuran 0 forward/backward", "[edge][tensor]") {
  auto X = std::make_shared<Tensor>(std::vector<size_t>{0}, std::vector<float>{}, true);
  auto res = gradient_check(fwd_sum, {X});
  REQUIRE(res.size() == 1);
  REQUIRE(res[0].ok);
  REQUIRE(res[0].n_checked == 0);
}

TEST_CASE("Edge: overflow tidak crash (hasil Inf diperbolehkan)", "[edge][numerik]") {
  auto a = std::make_shared<Tensor>(std::vector<size_t>{2},
      std::vector<float>{1e30f, 1e30f}, false);
  auto b = std::make_shared<Tensor>(std::vector<size_t>{2},
      std::vector<float>{1e30f, 1e30f}, false);
  auto out = mul(a, b);
  REQUIRE(out->numel() == 2);
  // 1e30 * 1e30 = Inf (overflow), tidak boleh crash
  REQUIRE(std::isinf(out->ptr[0]));
  REQUIRE(std::isinf(out->ptr[1]));
}

TEST_CASE("Edge: softmax dengan nilai sangat besar stabil", "[edge][norm][numerik]") {
  auto X = std::make_shared<Tensor>(std::vector<size_t>{1, 3},
      std::vector<float>{1000.0f, 1001.0f, 1002.0f}, false);
  auto out = softmax(X);
  REQUIRE(out->numel() == 3);
  float s = out->ptr[0] + out->ptr[1] + out->ptr[2];
  REQUIRE(s == Approx(1.0f).epsilon(1e-4f));
}

// ============================================================
// REGRESI: AdamW, checkpoint, LRScheduler
// ============================================================

TEST_CASE("Regresi: AdamW step mengubah bobot (perkuat)", "[train][optimizer]") {
  OptimizerConfig cfg;
  cfg.lr = 0.1f;
  cfg.weight_decay = 0.0f;
  AdamW opt(cfg);
  auto p = std::make_shared<Tensor>(std::vector<size_t>{2},
      std::vector<float>{1.0f, 2.0f}, true);
  p->grad.assign({0.5f, 0.5f});
  opt.add_parameter(p);
  float before0 = p->ptr[0];
  float before1 = p->ptr[1];
  opt.step(1);
  REQUIRE(p->ptr[0] != before0);
  REQUIRE(p->ptr[1] != before1);
  // Bobot harus berkurang karena gradien positif
  REQUIRE(p->ptr[0] < before0);
  REQUIRE(p->ptr[1] < before1);
}

TEST_CASE("Regresi: AdamW weight decay mempengaruhi bobot", "[train][optimizer]") {
  OptimizerConfig cfg;
  cfg.lr = 0.1f;
  cfg.weight_decay = 0.1f;
  AdamW opt(cfg);
  auto p = std::make_shared<Tensor>(std::vector<size_t>{1},
      std::vector<float>{1.0f}, true);
  p->grad.assign({0.0f});
  opt.add_parameter(p);
  float before = p->ptr[0];
  opt.step(1);
  REQUIRE(p->ptr[0] != before);
}

TEST_CASE("Regresi: AdamW gradient clipping", "[train][optimizer]") {
  OptimizerConfig cfg;
  cfg.lr = 0.1f;
  cfg.weight_decay = 0.0f;
  AdamW opt(cfg);
  auto p = std::make_shared<Tensor>(std::vector<size_t>{1},
      std::vector<float>{1.0f}, true);
  p->grad.assign({10.0f});
  opt.add_parameter(p);
  float before = p->ptr[0];
  opt.step(1);
  REQUIRE(p->ptr[0] != before);
  // global_grad_norm mengembalikan norma 10.0
  REQUIRE(opt.global_grad_norm() == Approx(10.0f).epsilon(1e-5f));
  // clip_gradients menurunkan norma gradien ke <= max_norm
  opt.clip_gradients(0.5f);
  REQUIRE(opt.global_grad_norm() <= 0.5f + 1e-6f);
}

TEST_CASE("Regresi: LRScheduler warmup + cosine", "[train][scheduler]") {
  LRScheduler sched(0.1f, 2, 10, 0.0f);
  // Warmup: step 0 -> 0, step 1 -> 0.05, step 2 -> 0.1
  REQUIRE(sched.lr_at(0) == Approx(0.0f));
  REQUIRE(sched.lr_at(1) == Approx(0.05f));
  REQUIRE(sched.lr_at(2) == Approx(0.1f));
  // Cosine decay turun monoton hingga min_lr di total_steps
  REQUIRE(sched.lr_at(5) < 0.1f);
  REQUIRE(sched.lr_at(10) == Approx(0.0f));
  REQUIRE(sched.lr_at(11) == Approx(0.0f));  // di atas total_steps tetap min_lr
}

TEST_CASE("Regresi: LRScheduler tanpa warmup (total_steps=0 -> konstan)", "[train][scheduler]") {
  LRScheduler sched(0.05f, 0, 0, 0.0f);
  REQUIRE(sched.lr_at(0) == Approx(0.05f));
  REQUIRE(sched.lr_at(100) == Approx(0.05f));
}

TEST_CASE("Regresi: checkpoint save/load dengan verifikasi CRC", "[train][checkpoint]") {
  Checkpoint ckpt;
  ckpt.global_step = 42;
  ckpt.seed = 1234;
  ckpt.param_names = {"w"};
  ckpt.param_values = {{1.0f, 2.0f, 3.0f}};
  ckpt.param_grads = {{0.1f, 0.2f, 0.3f}};
  ckpt.data_position = 7;
  ckpt.optimizer_state.global_step = 42;
  ckpt.optimizer_state.current_lr = 0.001f;
  ckpt.optimizer_state.step_counts = {1};
  ckpt.optimizer_state.m = {{0.01f, 0.02f, 0.03f}};
  ckpt.optimizer_state.v = {{0.001f, 0.002f, 0.003f}};
  ckpt.rng_state = {1, 2, 3, 4, 5};

  std::string path = "/data/data/com.termux/files/usr/tmp/rak_test_checkpoint.bin";
  save_checkpoint(path, ckpt);
  auto ckpt2 = load_checkpoint(path);
  REQUIRE(ckpt2.global_step == 42);
  REQUIRE(ckpt2.seed == 1234);
  REQUIRE(ckpt2.param_names == std::vector<std::string>({"w"}));
  REQUIRE(ckpt2.param_values[0] == std::vector<float>({1.0f, 2.0f, 3.0f}));
  REQUIRE(ckpt2.param_grads[0] == std::vector<float>({0.1f, 0.2f, 0.3f}));
  REQUIRE(ckpt2.data_position == 7);
  REQUIRE(ckpt2.optimizer_state.global_step == 42);
  REQUIRE(ckpt2.optimizer_state.current_lr == Approx(0.001f));
  REQUIRE(ckpt2.optimizer_state.step_counts == std::vector<int64_t>({1}));
  REQUIRE(ckpt2.rng_state == std::vector<uint8_t>({1, 2, 3, 4, 5}));
  std::remove(path.c_str());
}

TEST_CASE("Regresi: checkpoint CRC terdeteksi saat file rusak", "[train][checkpoint][negative]") {
  Checkpoint ckpt;
  ckpt.global_step = 1;
  ckpt.param_names = {"w"};
  ckpt.param_values = {{1.0f}};
  ckpt.param_grads = {{0.1f}};
  std::string path = "/data/data/com.termux/files/usr/tmp/rak_test_checkpoint_crc.bin";
  save_checkpoint(path, ckpt);
  // Rusak 1 byte di tengah payload (bukan CRC terakhir) -> CRC harus tidak cocok
  {
    std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
    REQUIRE(f.is_open());
    f.seekg(20, std::ios::beg);
    char b = 0;
    f.read(&b, 1);
    b ^= 0xFF;
    f.seekp(20, std::ios::beg);
    f.write(&b, 1);
  }
  REQUIRE_THROWS_AS(load_checkpoint(path), std::runtime_error);
  std::remove(path.c_str());
}

TEST_CASE("Regresi: checkpoint file tidak ditemukan melempar", "[train][checkpoint][negative]") {
  REQUIRE_THROWS_AS(load_checkpoint("/nonexistent/rak_checkpoint.bin"), std::runtime_error);
}

// ============================================================
// TEST NEGATIF: input null, shape invalid, axis di luar jangkauan
// ============================================================

TEST_CASE("Negatif: rms_norm weight ukuran tidak cocok melempar", "[negative][norm]") {
  auto X = std::make_shared<Tensor>(std::vector<size_t>{2, 3},
      std::vector<float>{1, 2, 3, 4, 5, 6}, false);
  auto W = std::make_shared<Tensor>(std::vector<size_t>{4},
      std::vector<float>{1, 1, 1, 1}, false);
  REQUIRE_THROWS_AS(rms_norm(X, W), std::invalid_argument);
}

TEST_CASE("Negatif: sparse_lookup query 2D melempar", "[negative][sparse]") {
  auto query = std::make_shared<Tensor>(std::vector<size_t>{3},
      std::vector<float>{1, 0, 0}, false);
  auto keys = std::make_shared<Tensor>(std::vector<size_t>{4, 3},
      std::vector<float>{1, 0, 0, 0, 1, 0, 0, 0, 1, 5, 0, 0}, false);
  REQUIRE_THROWS_AS(sparse_lookup_topk(query, keys, 2), std::invalid_argument);
}

TEST_CASE("Negatif: sparse_update slots bukan 2D melempar", "[negative][sparse]") {
  auto slots = std::make_shared<Tensor>(std::vector<size_t>{3},
      std::vector<float>{1, 2, 3}, false);
  auto indices = std::make_shared<Tensor>(std::vector<size_t>{1},
      std::vector<float>{0.0f}, false);
  auto updates = std::make_shared<Tensor>(std::vector<size_t>{1, 1},
      std::vector<float>{10}, false);
  REQUIRE_THROWS_AS(sparse_update(slots, indices, updates, "add"), std::invalid_argument);
}

TEST_CASE("Negatif: sparse_update mode tidak valid melempar", "[negative][sparse]") {
  auto slots = std::make_shared<Tensor>(std::vector<size_t>{3, 2},
      std::vector<float>{1, 2, 3, 4, 5, 6}, false);
  auto indices = std::make_shared<Tensor>(std::vector<size_t>{1},
      std::vector<float>{0.0f}, false);
  auto updates = std::make_shared<Tensor>(std::vector<size_t>{1, 2},
      std::vector<float>{10, 20}, false);
  REQUIRE_THROWS_AS(sparse_update(slots, indices, updates, "invalid"), std::invalid_argument);
}

TEST_CASE("Negatif: local_attention shape tidak cocok melempar", "[negative][attention]") {
  auto Q = std::make_shared<Tensor>(std::vector<size_t>{1, 4, 3},
      std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, false);
  auto K = std::make_shared<Tensor>(std::vector<size_t>{1, 4, 4},
      std::vector<float>{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}, false);
  auto V = std::make_shared<Tensor>(std::vector<size_t>{1, 4, 3},
      std::vector<float>{10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120}, false);
  REQUIRE_THROWS_AS(local_attention(Q, K, V, 2), std::invalid_argument);
}

// ============================================================
// TEST FUNGSI PUBLIK LAINNYA (non-tautologis)
// ============================================================

TEST_CASE("Ops: unbroadcast mengurangi grad broadcast dengan benar", "[ops][elementwise]") {
  // grad shape [2,3], target shape [3] (axis terakhir)
  std::vector<float> grad = {1, 2, 3, 4, 5, 6};
  std::vector<size_t> grad_shape = {2, 3};
  std::vector<size_t> target_shape = {3};
  std::vector<size_t> target_stride = {1};
  auto out = unbroadcast(grad, grad_shape, target_shape, target_stride);
  REQUIRE(out.size() == 3);
  REQUIRE(out[0] == Approx(1 + 4));
  REQUIRE(out[1] == Approx(2 + 5));
  REQUIRE(out[2] == Approx(3 + 6));
}

TEST_CASE("Ops: recurrent_scan sequential vs parallel setara", "[ops][scan]") {
  auto X = std::make_shared<Tensor>(std::vector<size_t>{1, 3, 2},
      std::vector<float>{1, 2, 3, 4, 5, 6}, false);
  auto D = std::make_shared<Tensor>(std::vector<size_t>{2},
      std::vector<float>{0.5f, 0.5f}, false);
  auto par = recurrent_scan(X, D, "parallel", "forward");
  auto seq = recurrent_scan(X, D, "sequential", "forward");
  REQUIRE(par->numel() == seq->numel());
  for (size_t i = 0; i < par->numel(); ++i) {
    REQUIRE(par->ptr[i] == Approx(seq->ptr[i]).epsilon(1e-5f));
  }
}

TEST_CASE("Ops: sparse_lookup_topk mengembalikan indeks terbaik", "[ops][sparse]") {
  auto query = std::make_shared<Tensor>(std::vector<size_t>{1, 1, 3},
      std::vector<float>{1, 0, 0}, false);
  auto keys = std::make_shared<Tensor>(std::vector<size_t>{4, 3},
      std::vector<float>{1, 0, 0, 0, 1, 0, 0, 0, 1, 5, 0, 0}, false);
  auto res = sparse_lookup_topk(query, keys, 2);
  REQUIRE(res.values->shape == std::vector<size_t>({1, 1, 2, 3}));
  // key 3 (nilai 5) dan key 0 (nilai 1) adalah top-2 (sort descending)
  REQUIRE(res.indices[0][0][0] == 3);
  REQUIRE(res.indices[0][0][1] == 0);
}

TEST_CASE("Ops: sparse_update mode add dan replace", "[ops][sparse]") {
  auto slots = std::make_shared<Tensor>(std::vector<size_t>{3, 2},
      std::vector<float>{1, 2, 3, 4, 5, 6}, false);
  auto indices = std::make_shared<Tensor>(std::vector<size_t>{1},
      std::vector<float>{0.0f}, false);
  auto updates = std::make_shared<Tensor>(std::vector<size_t>{1, 2},
      std::vector<float>{10, 20}, false);
  auto out = sparse_update(slots, indices, updates, "add");
  REQUIRE(out->ptr[0] == Approx(11.0f));
  REQUIRE(out->ptr[1] == Approx(22.0f));
  // Slot 1 tidak berubah
  REQUIRE(out->ptr[2] == Approx(3.0f));

  auto out2 = sparse_update(slots, indices, updates, "replace");
  REQUIRE(out2->ptr[0] == Approx(10.0f));
  REQUIRE(out2->ptr[1] == Approx(20.0f));
}

TEST_CASE("Tensor: validate konsisten", "[tensor]") {
  Tensor t({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
  REQUIRE_NOTHROW(t.validate());
}
