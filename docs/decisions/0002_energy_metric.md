# ADR-0002: Metrik energi = SynOps, bukan joule

- Status: Accepted
- Date: 2026-10-15
- Affects: NFR-04, T7.3

## Konteks

PRD v1.0 menulis `NFR-04: Energy per inference < 10x backprop baseline`.

Dua masalah:

**1. Spike count bukan joules.** Hitungan spike hanya proxy yang longgar.
Tidak ada konversi spike → joule yang valid tanpa asumsi hardware.

**2. Di CPU, SNN tidak otomatis lebih hemat.** Klaim "SNN hemat energi"
benar **hanya di hardware neuromorf**. Fact:

| Platform |Efisiensi |
|---|---|
| Loihi 2 vs Jetson Nano B01 | 10–35× lebih hemat |
| Loihi 2 vs Jetson Orin Nano | 1.6–3.7× lebih hemat |
| SpikingJelly CPU, T=8 vs ANN | ~2× **lebih mahal** |
| SpikingJelly CPU, T=1–2 vs ANN | 24–52% operasi (lebih murah) |

Jadi di Codespaces (CPU murni) klaim `< 10x backprop` itu **meaningless**
dan kemungkinan besar **gagal** kalau diukur dengan benar.

## Keputusan

Ganti NFR-04 dengan:

> **NFR-04-Rev: SynOps per inference < 0.5× ANN FLOPs pada akurasi setara.**

- **SynOps** (sparsity-aware synaptic operations) = jumlah operasi
  synaptic yang benar-benar terjadi. Ini proxy yang jujur: kalau
  spike density 5%, SynOps turun ~20×.
- Joule absolut (RAPL) tetap diukur sebagai **metrik sekunder**, tapi
  WAJIB dilabeli **"estimated"**.

## Kenapa RAPL harus dilabeli "estimated"

Intel RAPL membaca **counter energi** di domain CPU + DRAM. Yang
dapat diukur langsung:

1. RAPL mengukur seluruh domain, bukan per-process.
2. Atribusi ke process harus lewat model estimasi (CPU time, perf
   counters).
3. Riset menunjukkan CPU + DRAM rata-rata ~80% konsumsi, sisanya
   (motherboard, fan, peripheral) hampir tidak bisa diatribusikan.

Jadi angka per-process adalah **turunan**, bukan pengukuran langsung.
Reviewer akan menanyakan ini duluan — jadi kita nyatakan duluan.

## Metrik turunan (Phase 2+)

Ditambahkan sebagai guard supaya homeostasis benar-benar bekerja:

| Metrik | Target | Cara ukur |
|---|---|---|
| Firing rate uniformity | std/mean < 0.1 setelah convergence | Homeostasis aktif |
| Homeostatic Θ | > 1 (Zenke 2013) | Sweep τ_detector |

## Konsekuensi

- Klaim energi di paper: **tidak boleh** menulis "RAK lebih hemat energi"
  tanpa qualify "SynOps pada CPU, simulasi software".
- Kalau RAK punya spike density rendah tapi T besar, SynOps mungkin
  tidak tercapai — itu temuan yang valid, laporkan.
- Tools: `perf stat -e power/energy-pkg/`, `turbostat`, PowerJoular.

## Alternatif yang ditolak

- **Hapus klaim energi sepenuhnya** — ditolak, kehilangan satu
  kontribusi paper yang legit (SynOps adalah metrik standar di
  benchmarking SNN).
- **Jumlah spike saja** — ditolak, bukan joules dan reviewer akan
  langsung menolak.
- **Ukur dengan GPU T4** — ditolak, GPU T4 adalah GPU, bukan
  neuromorf. Spike di GPU tidak hemat energi, dan T4 bukan
  representatif deployment RAK.