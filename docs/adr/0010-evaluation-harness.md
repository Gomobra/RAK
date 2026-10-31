# ADR 0010 — Evaluasi Harness Fase 1 (F1.3)

Status: Diterima

## 1. Konteks

Fase 1 (PRD F1.3, T1.3.1–T1.3.5) membutuhkan alat ukur yang dipakai semua fase: bits-per-byte, tugas continual (belajar domain A lalu B, ukur lupa di A), forgetting dan forward transfer, recall dari memori, dan kerangka audit penghapusan (diisi di Fase 5). Hasil selalu dalam mean ± std dari beberapa seed, dan laporan dibuat otomatis.

Harness ini mengukur baseline Fase 1 (Transformer dan hybrid) dan menjadi acuan untuk Fase 2+.

## 2. Keputusan

Dibangun evaluasi harness di Python (orchestrasi) yang memanfaatkan engine RAK C++17 via binding `_rak` (Tensor, ops, autograd, optimizer, checkpoint). Harness mencakup:

### 2.1 Metrik Bits-per-Byte (BPB) dan Perplexity

- BPB = −log₂(P(data)) / n_byte = cross_entropy / ln(2) / n_byte
- Perplexity = exp(cross_entropy) per token
- Cross-entropy dihitung dari log_softmax logits pada token target berikutnya
- Padding diabaikan (mask)
- Implementasi di Python via binding `_rak.log_softmax` dan `_rak.mean`

### 2.2 Protokol Continual Learning

- Urutan domain: Domain A (data pertama) → Domain B (data kedua)
- Model dilatih pada A hingga konvergen, simpan snapshot (checkpoint)
- Model dilanjutkan (resume) pada B
- Ukur:
  - Akurasi/loss pada A sebelum dan setelah belajar B → forgetting
  - Akurasi pada B → forward transfer
- Matriks skor: sel (domain_test × domain_train), nilai BPB/perplexity
- Lupa (forgetting) = BPB_A_setelah_B − BPB_A_sebelum_B
- Forward transfer = BPB_B_dengan_A − BPB_B_tanpa_A

### 2.3 Metrik Recall (exact@k) untuk Memori

- Simpan entri (kunci, nilai) dalam memori slot bank (sparse_lookup_topk)
- Query: cari entri berdasarkan kemiripan kunci
- exact@k = fraksi entri yang kunci-nya muncul dalam top-k hasil pencarian
- Diuji pada N entri (default 1000), k variatif (1, 5, 10)

### 2.4 Kerangka Audit Penghapusan (Stub)

- Stub untuk Fase 5: struktur fungsi audit yang akan diisi nanti
- Tidak mengubah state model saat read (hash sebelum = sesudah)
- Read-only verifikasi, tidak ada operasi write

### 2.5 Format Laporan Otomatis

- Tabel mean ± std, minimal 3 seed
- Disimpan di `experiments/reports/`
- Format: Markdown + JSON (untuk parsing otomatis)
- Setiap laporan mencatat: config hash, seed, versi kode, timestamp, metrik per seed

### 2.6 Integrasi dengan Engine RAK

- Model diinisialisasi di Python via binding `_rak` (Tensor, ops)
- Forward/backward inner-loop berjalan di C++ (via `_rak`)
- Evaluasi (BPB, perp, recall) dihitung di Python dari output tensor
- Checkpoint/resume via `save_checkpoint`/`load_checkpoint` (CRC32 + RNG state)

## 3. Spesifikasi Teknis Ringkas

| Komponen | Detail |
|---|---|
| Bahasa implementasi | Python (orchestrasi), C++ (inner loop via `_rak`) |
| Metrik utama | BPB, perplexity, exact@k, forgetting (δ BPB), forward transfer |
| Seed minimal | 3 seed per eksperimen (42, 123, 777) |
| Laporan | `experiments/reports/eval_<eksperimen>_<timestamp>.md` + `.json` |
| Audit penghapusan | Stub (Fase 5) |
| Determinisme | Seed dicatat, split deterministik berdasarkan doc_hash |

## 4. Langkah Selanjutnya (DoD F1.3)

- Implementasi skrip evaluasi di `src/rak/eval/` dan `scripts/experiments/`
- Uji pada baseline Fase 1 (Transformer + hybrid, 10–30M, ≥3 seed)
- Validasi determinisme (dua kali run dengan seed sama → hasil bit-benar)
- Gradient check, test lengkap, benchmark, dan data real ≥ 3 seed sebelum naik status ke ASLI

## 5. Konsekuensi

- Semua eksperimen Fase 1 harus menggunakan harness ini untuk konsistensi
- Hasil negatif tetap dilaporkan (PRD "Jujur")
- Ambang gate G2–G6 dikalibrasi setelah baseline Fase 1 tersedia (mean ± std)
