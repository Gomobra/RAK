# ADR 0011 — Anggaran Komputasi Fase 1.4 (T1.4.1–T1.4.4)

Status: Diajukan

## 1. Konteks

Fase 1.4 (PRD F1.4, T1.4.1–T1.4.4) membutuhkan anggaran komputasi realistis agar rencana pelatihan tidak mustahil dijalankan, mengingat keterbatasan: satu orang + AI, anggaran mendekati nol, Kaggle GPU gratis terbatas (~30 jam/minggu, sesi maksimal ~9 jam), GitHub Actions CPU terbatas (biasanya ≤ 6 jam per job), dan Termux CPU dengan RAM terbatas (~5.5 GiB, swap aktif).

Engine Fase 0 sudah ASLI (GEMM, autograd reverse-mode, AdamW, checkpoint CRC32 + state RNG untuk resume deterministik, binding Python). Baseline Fase 1 (Transformer decoder dan hybrid rekuren + attention lokal, ukuran 10M dan 30M, ADR 0009) dan evaluation harness (ADR 0010: BPB, forgetting, exact@k) menyediakan basis untuk ambang gate G2–G6.

Dokumen ini mencakup lima hal yang diminta Tugas:
1. Pengukuran kecepatan latihan (token/detik) di Termux CPU, GitHub Actions CPU, Kaggle GPU.
2. Perhitungan kebutuhan latihan 10–30M dan 100M parameter dalam jam di tiap lingkungan.
3. Jadwal sesi Kaggle dengan checkpoint dan resume (menggunakan checkpoint engine T0.4.3/T0.4.4).
4. Ambang gate G2–G6 berdasarkan baseline (BPB threshold, forgetting threshold, dll.).
5. Anggaran compute per fase agar rencana tidak mustahil.

## 2. Metode Pengukuran (T1.4.1)

Pengukuran dilakukan dengan `rak_bench` (C++ engine langsung, tanpa overhead Python orchestration) untuk mendapatkan lower-bound microbenchmark, lalu dikonversi ke estimasi token/detik latihan dengan asumsi efektif.

Rumus konversi (asumsi yang dicatat, akan diverifikasi oleh T1.4.1):

- Forward per token ≈ 2 × params FLOP (matmul-dominan).
- Forward + backward (backprop) ≈ 6 × params FLOP per token.
- token_per_sec = GFLOP_s_efektif × 10⁹ / (6 × params).

Microbenchmark yang sudah dijalankan (Termux CPU, aarch64 clang, single-thread, 2026-10-31):

| Op | Ukuran | Waktu | Efektif |
|---|---|---|---|
| GEMM | 128×128×128 | 7,85 ms | ~534 MFLOP/s |

Asumsi GFLOP/s efektif untuk loop latihan lengkap (forward+backward + overhead Python + memory bandwidth, belum multithread):

| Lingkungan | GFLOP/s efektif (asumsi) | Alasan |
|---|---|---|
| Termux CPU | 1,0 | Microbenchmark 0,534; matmul lebih besar + batch naikkan sedikit, RAM terbatas |
| GitHub Actions CPU | 5,0 | x86-64, 2 core, multi-thread ringan, terbatas CI |
| Kaggle GPU | 1000,0 | T4/P100 fp32, efektif setelah overhead data |

Estimasi token/detik latihan (asumsi, akan diukur T1.4.1):

| Model | params | Termux | Actions | Kaggle |
|---|---|---|---|---|
| 10M | 1×10⁷ | ~17 tok/s | ~83 tok/s | ~16.700 tok/s |
| 30M | 3×10⁷ | ~5,6 tok/s | ~28 tok/s | ~5.556 tok/s |
| 100M | 1×10⁸ | ~1,7 tok/s | ~8,3 tok/s | ~1.667 tok/s |

## 3. Kebutuhan Latihan dalam Jam (T1.4.2)

Asumsi jumlah token (belum diukur pada data real, akan disesuaikan setelah F1.1):

- 10M prototipe: 10⁸ token (100M token), 1 seed
- 30M prototipe: 2×10⁸ token, 1 seed
- 100M utama: 10⁹ token per seed, 3 seed

Jam = total_token / (token_per_sec × 3600).

| Model | Termux (jam) | Actions (jam) | Kaggle per seed (jam) | Kaggle 3 seed (jam) |
|---|---|---|---|---|
| 10M | ~1.667 | ~333 | ~1,7 | ~5,0 |
| 30M | ~9.921 | ~2.000 | ~10,0 | ~30,0 |
| 100M | ~166.667 | ~33.333 | ~166,7 | ~500,0 |

Baca dengan catatan penting:
- **Termux** tidak bisa melatih 10–30M secara penuh (ribuan jam). Cocok hanya untuk smoke test, gradient check, dan model kecil (≤ 3M params, ≤ 10⁶ token).
- **Actions** melebihi batas CI (≤ 6 jam/job) untuk 10M penuh. Cocok hanya untuk build/test dan validasi kecil.
- **Kaggle** adalah lokasi pelatihan utama. 10M (≈5 jam, 3 seed) dan 30M (≈30 jam, 3 seed ≈ 1 minggu gratis) masuk anggaran. 100M (≈500 jam, 3 seed ≈ 17 minggu Kaggle gratis) **tidak memuat** dalam kuota gratis — perlu dipotong token, dikurangi seed, atau diakui sebagai stretch yang butuh dana/compute tambahan. Ini temuan jujur, bukan pembalikan hipotesis.

## 4. Jadwal Sesi Kaggle dengan Checkpoint dan Resume (T1.4.3)

Checkpoint engine yang sudah ada (T0.4.3/T0.4.4): simpan bobot + state optimizer + state RNG + posisi data, verifikasi CRC32, resume deterministik (bit-sama pada mesin yang sama). Setiap sesi Kaggle diakhiri dengan checkpoint; sesi berikutnya melanjutnya tanpa menghilangkan progres.

Asumsi kuota Kaggle gratis: ~30 jam GPU/minggu, sesi maksimal ~9 jam.

### Fase 1 — Baseline 10M dan 30M (3 seed)

| Sesi | Jam | Pekerjaan | Checkpoint akhir |
|---|---|---|---|
| 1 | 9 | Inisialisasi 10M seed 1,2,3 (≈5 jam total) | ckpt_10m_s{1,2,3}_final |
| 2 | 9 | Inisialisasi 30M seed 1 (≈10 jam, terpotong) | ckpt_30m_s1_partial |
| 3 | 9 | Lanjut 30M seed 1 → selesai; mulai 30M seed 2 | ckpt_30m_s1_final, ckpt_30m_s2_partial |
| 4 | 3 | Selesaikan 30M seed 2,3 | ckpt_30m_s{2,3}_final |

Total ≈ 30 jam = 1 minggu Kaggle gratis. Semua hasil disimpan di `experiments/runs/` dan ringkasan di `experiments/reports/` (hanya ringkasan yang di-commit, PRD 6.2).

### Fase 6 — 100M (stretch, melebihi kuota gratis)

Checkpoint setiap epoch (atau setiap N token, mis. 10⁷ token) agar setiap sesi 9 jam dapat dilanjutkan. Estimasi 500 jam untuk 3 seed → ~56 sesi Kaggle ≈ 17 minggu. Jangan jadwalkan ini dalam satu sesi; gunakan resume otomatis dan valut ulang apakah 100M masih layak setelah hasil 30M (gate G6).

### Prosedur resume (setiap sesi Kaggle)

1. Muat checkpoint terakhir via `rak::train::Checkpoint::load` (verifikasi CRC32).
2. Pulihkan state optimizer AdamW + state RNG + posisi data/shard.
3. Lanjutkan loop pelatihan dari posisi data yang tersimpan.
4. Akhir sesi: simpan checkpoint baru, verifikasi CRC32, catat hash config + seed + versi kode.

## 5. Ambang Gate G2–G6 (T1.4.4)

PRD 2.4 menyatakan ambang angka ditetapkan setelah baseline Fase 1 selesai. Karena baseline belum dilatih, ambang di bawah diformulasikan dan diberi **nilai contoh yang harus dikalibrasi ulang** setelah kurva baseline tersedia (mean ± std, 3 seed).

### G2 (Fase 2) — RAK-core vs hybrid baseline, 10–30M
- BPB RAK-core ≤ BPB hybrid + δ_bpb, dengan δ_bpb = 0,15 (contoh, dikalibrasi).
- Morfologi: akurasi kata berimbuhan/typo RAK-core + modul bahasa > tanpa modul, tanpa menurunkan BPB umum lebih dari δ_bpb.
- Minimal 3 seed, laporan mean ± std.

### G3 (Fase 3) — Slot FFN vs FFN biasa
- Satu varian slot FFN (MoE atau memory layer) mencapai BPB ≤ FFN baseline − δ_slot (δ_slot = 0,05 contoh) pada compute sama, ATAU sama BPB dengan waktu nyata lebih rendah.
- Loop adaptif menghemat **waktu nyata** (wall clock), bukan hanya kedalangan pada kertas.
- Router tidak collapse (pemakaian expert terpantau).

### G4 (Fase 4) — Memory File Manager
- exact@k recall ≥ 0,90 pada k = 5, N = 1.000 entri (contoh, dikalibrasi).
- Latensi `search` < 100 ms pada N = 10.000 entri di Termux (contoh).
- `read` terbukti tidak mengubah state (hash sebelum = sesudah).

### G5 (Fase 5) — Neuromodulator + Tidur
- Forgetting pada domain A setelah belajar B lebih rendah dari baseline pada compute sama (δ_forget = 0,05 akurasi, contoh).
- Verifikasi tidur lulus dan rollback bekerja saat verifikasi gagal.
- Audit penghapusan lulus empat lapis pada uji skala kecil.

### G6 (Fase 6, gate besar) — 100M continual learning
- Forgetting RAK < baseline pada tugas continual yang sama dengan compute sama.
- Kualitas umum (BPB pada test_fixed) tidak turun lebih dari δ_gen = 0,10 (contoh).
- Keputusan resmi: lanjut ke Fase 7, perbaiki Fase 6, atau ubah hipotesis. Hasil negatif tetap dilaporkan.

Semua ambang akan ditimpa oleh nilai kalibrasi setelah baseline Fase 1 selesai dan dicatat ulang di dokumen ini.

## 6. Anggaran Compute per Fase (agar tidak mustahil)

| Fase | Lingkungan utama | Sinkupan anggaran | Catatan |
|---|---|---|---|
| Fase 0 (Engine) | Termux + Actions | Smoke test, gradient check, build/test CI | Tidak pelatihan model besar |
| Fase 1 (Data + Baseline) | Kaggle GPU | 10M ≈ 5 jam, 30M ≈ 30 jam (3 seed) | Termux/Actions hanya validasi kecil |
| Fase 2 (Inti teks + bahasa A) | Kaggle GPU | ~30–60 minggu? | Prototipe kecil, 3 seed |
| Fase 3–5 (Korteks, MFM, tidur) | Kaggle GPU | Skala kecil, 10–30M | Checkpoint/resume wajib |
| Fase 6 (100M) | Kaggle GPU | ~500 jam (3 seed) — **melebihi gratis** | Kurangi token/seed atau cari dana; jangan paksa dalam satu sesi |
| Fase 7–11 | Kaggle / Termux | Sesuai hasil G6 | Disetujui setelah gate G6 |

Aturan keuangan: tidak memaksa 100M pada kuota gratis Kaggle. Jika G6 membutuhkan 100M penuh, usulkan alternatif (kurangi seed, kurangi token, atau dorong ke lingkungan berbayar) dan laporkan sebagai temuan, tidak tersembunyi.

## 7. Konsekuensi

- Pengukuran aktual (T1.4.1) harus dilakukan sebelum angka di atas dipakai untuk komitmen jadwal; angka ini adalah estimasi awal dengan asumsi tercatat.
- 10–30M dapat diselesaikan dalam kuota Kaggle gratis (~1–2 minggu). 100M penuh tidak dapat — anggaran per fase harus mencerminkannya agar tidak mustahil.
- Checkpoint dan resume deterministik (T0.4.3/T0.4.4) adalah syarat agar sesi Kaggle yang terpotong tetap bernilai.
- Ambang gate G2–G6 akan dikalibrasi ulang setelah baseline Fase 1 tersedia; dokumen ini diperbarui pada saat itu.

## 8. Terkait

- PRD F1.4 (T1.4.1–T1.4.4), PRD 1.7 risiko compute, PRD 2.4 metrik dan gate, PRD 2.5 Definition of Done.
- ADR 0009 (baseline model), ADR 0010 (evaluation harness).
- T0.4.3/T0.4.4 (checkpoint CRC32 + resume deterministik).
