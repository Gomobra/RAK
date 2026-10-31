// apps/bench_ops.cpp — Benchmark kecepatan dan memori per op RAK.
// Digunakan untuk memenuhi Definition of Done PRD 2.5 (benchmark per op).
// Pemakaian: ./rak_bench (CMake target rak_bench dengan RAK_BUILD_BENCH=ON)

#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <iomanip>
#include <iostream>

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

using namespace rak::core;
using namespace rak::ops;

static double now_s() {
  return std::chrono::duration<double>(
      std::chrono::high_resolution_clock::now().time_since_epoch())
      .count();
}

// Tensor random dengan nilai dalam [lo, hi].
static std::shared_ptr<Tensor> rand_tensor(const std::vector<size_t>& shape,
                                           float lo = -1.0f, float hi = 1.0f) {
  size_t n = 1;
  for (size_t d : shape) n *= d;
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = lo + (hi - lo) * static_cast<float>(rand()) / RAND_MAX;
  return std::make_shared<Tensor>(shape, v, false);
}

struct BenchResult {
  std::string op;
  std::string size;
  double ms;
  size_t bytes;
};

static std::vector<BenchResult> results;

#define BENCH(op_call, size_label, alloc_bytes)                                      \
  do {                                                                               \
    auto t0 = now_s();                                                               \
    op_call;                                                                         \
    auto t1 = now_s();                                                               \
    results.push_back({#op_call, size_label, (t1 - t0) * 1000.0, alloc_bytes});      \
  } while (0)

int main() {
  srand(12345);

  // ---------- GEMM ----------
  for (auto [M, K, N] : std::vector<std::tuple<size_t,size_t,size_t>>{
           {16, 16, 16}, {64, 64, 64}, {128, 128, 128}}) {
    auto A = rand_tensor({M, K});
    auto B = rand_tensor({K, N});
    BENCH(gemm(A, B), "gemm " + std::to_string(M) + "x" + std::to_string(K) + "x" + std::to_string(N),
          (M * K + K * N + M * N) * sizeof(float));
  }

  // ---------- Elementwise ----------
  auto x1 = rand_tensor({1024});
  auto x2 = rand_tensor({1024});
  BENCH(add(x1, x2), "add 1024", 3 * 1024 * sizeof(float));
  BENCH(mul(x1, x2), "mul 1024", 3 * 1024 * sizeof(float));
  BENCH(silu(x1), "silu 1024", 2 * 1024 * sizeof(float));
  BENCH(gelu(x1), "gelu 1024", 2 * 1024 * sizeof(float));

  // ---------- Reduce ----------
  auto r1 = rand_tensor({64, 64});
  BENCH(sum(r1, -1), "sum 64x64->64", 64 * 64 * sizeof(float) + 64 * sizeof(float));
  BENCH(mean(r1, -1), "mean 64x64->64", 64 * 64 * sizeof(float) + 64 * sizeof(float));
  BENCH(max(r1, -1), "max 64x64->64", 64 * 64 * sizeof(float) + 64 * sizeof(float));

  // ---------- Norm ----------
  auto wn = std::make_shared<Tensor>(std::vector<size_t>{64},
      std::vector<float>(64, 1.0f), false);
  BENCH(rms_norm(r1, wn), "rms_norm 64x64", 3 * 64 * 64 * sizeof(float));
  BENCH(softmax(r1), "softmax 64x64", 3 * 64 * 64 * sizeof(float));
  BENCH(log_softmax(r1), "log_softmax 64x64", 3 * 64 * 64 * sizeof(float));

  // ---------- Embedding ----------
  auto Wemb = rand_tensor({100, 16});
  auto Xidx = std::make_shared<Tensor>(std::vector<size_t>{32},
      std::vector<float>{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
                         16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31}, false);
  BENCH(embedding(Xidx, Wemb), "embedding 32->32x16", 100 * 16 * sizeof(float) + 32 * 16 * sizeof(float));

  // ---------- Recurrent scan ----------
  auto Xscan = rand_tensor({4, 10, 8});
  auto Dscan = rand_tensor({8}, 0.5f, 0.9f);
  BENCH(recurrent_scan(Xscan, Dscan, "parallel", "forward"),
        "scan parallel 4x10x8", 2 * 4 * 10 * 8 * sizeof(float));
  BENCH(recurrent_scan(Xscan, Dscan, "sequential", "forward"),
        "scan sequential 4x10x8", 2 * 4 * 10 * 8 * sizeof(float));

  // ---------- Local attention ----------
  auto Q = rand_tensor({2, 16, 16});
  auto K = rand_tensor({2, 16, 16});
  auto V = rand_tensor({2, 16, 16});
  BENCH(local_attention(Q, K, V, 4), "local_attention 2x16x16 w=4",
        3 * 2 * 16 * 16 * sizeof(float));

  // ---------- Sparse lookup ----------
  auto query = rand_tensor({2, 4, 8});
  auto keys = rand_tensor({32, 8});
  BENCH(sparse_lookup_topk(query, keys, 4), "sparse_lookup_topk 2x4 k=4",
        2 * 4 * 4 * 8 * sizeof(float) + 32 * 8 * sizeof(float));

  // ---------- Sparse update ----------
  auto slots = rand_tensor({32, 16});
  auto idx = std::make_shared<Tensor>(std::vector<size_t>{4},
      std::vector<float>{0, 8, 16, 24}, false);
  auto upd = rand_tensor({4, 16});
  BENCH(sparse_update(slots, idx, upd, "add"), "sparse_update add 4x16",
        2 * 32 * 16 * sizeof(float));

  // ---------- Laporan ----------
  std::cout << "\n=== RAK Benchmark per Op ===\n";
  std::cout << std::left << std::setw(32) << "Op" << std::setw(20) << "Ukuran"
            << std::setw(12) << "Waktu (ms)" << std::setw(14) << "Memori (KB)" << "\n";
  std::cout << std::string(78, '-') << "\n";
  double total_ms = 0;
  for (auto& r : results) {
    std::cout << std::left << std::setw(32) << r.op << std::setw(20) << r.size
              << std::setw(12) << std::fixed << std::setprecision(4) << r.ms
              << std::setw(14) << (r.bytes / 1024.0) << "\n";
    total_ms += r.ms;
  }
  std::cout << std::string(78, '-') << "\n";
  std::cout << std::left << std::setw(32) << "TOTAL" << std::setw(20) << ""
            << std::setw(12) << std::fixed << std::setprecision(4) << total_ms << "\n";
  std::cout << "\nBenchmark selesai.\n";
  return 0;
}
