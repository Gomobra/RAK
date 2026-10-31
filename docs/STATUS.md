# Status Modul RAK

Status: **ASLI** (lolos Definition of Done, PRD 2.5, dan diukur di data real) · **RINGAN** · **KERANGKA** · **BELUM DIMULAI**.
Wajib diperbarui setiap modul berubah. Status naik ke ASLI hanya dengan bukti (test, angka mean ± std, log).

| Modul | Layer | Fase / Task | Status | Bukti | Kurang | Diperbarui |
|---|---|---|---|---|---|---|
| core (tensor, autograd) | L7 | F0.2 | ASLI | build OK, gradient check semua op via tests/parity + test_core.cpp, verifikasi numerik lulus (autograd backward bit-benar) | — | 2026-10-31 / T0.2.x |
| ops (op dasar dan khusus RAK) | L7 | F0.3 | ASLI | semua op kompilasi, gradient check per op lolos, benchmark di experiments/reports/bench_report_2026-10-31.md | — | 2026-10-31 / T0.3.x |
| train (optimizer, checkpoint) | L8 | F0.4 | ASLI | AdamW lr_at pakai LRScheduler (warmup linear + cosine decay); checkpoint simpan/muat dengan verifikasi CRC32 + field rng_state untuk resume deterministik; verifikasi numerik lulus (LR warmup+cosine, CRC deteksi kerusakan) | — | 2026-10-31 / T0.4.x |
| cuda | L7 | F0.5 | BELUM DIMULAI | | Kernel CUDA, paritas CPU vs CUDA | |
| bindings Python | L7 | F0.7 | ASLI | _rak.so build dan impor dari Python; buffer protocol, binding op & optimizer; uji paritas Python vs C++ di tests/parity/ | editable install di Termux | 2026-10-31 / T0.7.x |
| data pipeline | L8 | F1.1 | BELUM DIMULAI | | | |
| baseline (Transformer, hybrid) | L1 | F1.2 | KERANGKA | ADR 0009 (spesifikasi arsitektur + estimasi parameter 10M/30M) | Implementasi model C++, test, training pipeline | 2026-10-31 / T1.2.x | |
| eval harness | L11 | F1.3 | BELUM DIMULAI | | | |
| encoders/bytepatch, bpe | L0 | F2.1 | BELUM DIMULAI | | | |
| cortex (blok dasar) | L1 | F2.2 | BELUM DIMULAI | | | |
| decoders/text | L6 | F2.3 | BELUM DIMULAI | | | |
| language tahap A (leksikon, morfologi) | L12 | F2.5 | BELUM DIMULAI | | | |
| cortex/moe, memory_layer | L1 | F3.1 | BELUM DIMULAI | | | |
| cortex/adaptive_loop | L1 | F3.2 | BELUM DIMULAI | | | |
| homeostasis | L5 | F3.3 | BELUM DIMULAI | | | |
| memory (MFM) | L2 | F4 | BELUM DIMULAI | | | |
| neuromod | L4 | F5.1 | BELUM DIMULAI | | | |
| sleep | L3 | F5.2-F5.4 | BELUM DIMULAI | | | |
| audit penghapusan | L2/L3/L11 | F5.5 | BELUM DIMULAI | | | |
| language tahap B-E | L12 | F7 | BELUM DIMULAI | | | |
| suara (codec, pita suara) | L0/L6 | F8 | BELUM DIMULAI | | | |
| gambar | L0/L6 | F9 | BELUM DIMULAI | | | |
| serve (runtime, http, policy) | L9/L10 | F10 | BELUM DIMULAI | | | |
