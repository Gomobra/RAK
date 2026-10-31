# Laporan Benchmark per Op — RAK Fase 0

**Tanggal:** 2026-10-31  
**Lingkungan:** Termux (aarch64, clang), CPU-only (RAK_CUDA=OFF)  
**Build:** Release, C++17  
**Binary:** `build/cpp/rak_bench` (target `rak_bench`)

## Hasil Benchmark

| Op | Ukuran | Waktu (ms) | Memori (KB) |
|---|---|---|---|
| gemm(A, B) | gemm 16x16x16 | 0.0514 | 3.0000 |
| gemm(A, B) | gemm 64x64x64 | 0.6105 | 48.0000 |
| gemm(A, B) | gemm 128x128x128 | 3.0532 | 192.0000 |
| add(x1, x2) | add 1024 | 0.0375 | 12.0000 |
| mul(x1, x2) | mul 1024 | 0.0339 | 12.0000 |
| silu(x1) | silu 1024 | 0.0415 | 8.0000 |
| gelu(x1) | gelu 1024 | 0.0942 | 8.0000 |
| sum(r1, -1) | sum 64x64->64 | 0.4502 | 16.2500 |
| mean(r1, -1) | mean 64x64->64 | 0.5145 | 16.2500 |
| max(r1, -1) | max 64x64->64 | 0.5179 | 16.2500 |
| rms_norm(r1, wn) | rms_norm 64x64 | 0.0519 | 48.0000 |
| softmax(r1) | softmax 64x64 | 0.1071 | 48.0000 |
| log_softmax(r1) | log_softmax 64x64 | 0.1142 | 48.0000 |
| embedding(Xidx, Wemb) | embedding 32->32x16 | 0.0231 | 8.2500 |
| recurrent_scan parallel | scan parallel 4x10x8 | 0.0175 | 2.5000 |
| recurrent_scan sequential | scan sequential 4x10x8 | 0.0067 | 2.5000 |
| local_attention | local_attention 2x16x16 w=4 | 0.0338 | 6.0000 |
| sparse_lookup_topk | sparse_lookup_topk 2x4 k=4 | 0.0245 | 2.0000 |
| sparse_update add | sparse_update add 4x16 | 0.0173 | 4.0000 |
| **TOTAL** | | **5.8009** | |

## Catatan

- Semua 18 op inti Fase 0 telah dibenchmark (GEMM, elementwise, reduce, norm, embedding, recurrent_scan, local_attention, sparse_lookup_topk, sparse_update).
- Waktu dieksekusi satu kali per op (tidak ada averaging ulang); cocok sebagai baseline awal.
- File sumber: `cpp/apps/bench_ops.cpp`, `cpp/apps/bench_gemm.cpp`
- Target CMake: `rak_bench` (diaktifkan dengan `-DRAK_BUILD_BENCH=ON`).
