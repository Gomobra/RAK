# ADR-0001: Omniglot split = Lake original

- Status: Accepted
- Date: 2026-10-15
- Affects: Phase 6 (T6.2), semua klaim few-shot

## Konteks

Omniglot punya **dua split yang beredar**, dan keduanya dipakai literatur
secara legit. Angka dari paper yang berbeda sering **tidak bisa
dibandingkan** karena itu sebabnya.

Tabel yang beredar (5-way 1-shot):

| Model | Lake split | Rotation split |
|---|---|---|
| Baseline nearest neighbor | 28.86% | — |
| Matching Networks | 43.40% | 98.1% |
| Meta-Learner LSTM | 43.44% | — |
| MAML | 48.70% | 98.7% |
| **Prototypical Networks** | **49.42%** | **98.8%** |

Model yang sama, angka beda ~2×, karena split-nya beda.

Konsekuensi untuk RAK:

- PRD v1.0 menulis target 70%. Kalau ternyata dipakai rotation split,
  angka itu **di bawah baseline ProtoNet** (98.8%) — artinya klaim
  "menang" jadi palsu.
- Kalau dipakai Lake split, 70% berarti **+20 poin di atas ProtoNet**,
  yang tidak realistis untuk satu orang dalam 2 tahun dengan local
  learning saja.

## Keputusan

**Pakai Lake et al. (2015) original split.** Target **55–60%** 5-way 1-shot.

Alasan:

1. Lake split adalah split yang dipakai paper original dan bisa
   dibandingkan ke hampir semua paper lama.
2. Rotation split Angkanya terlalu tinggi dan sulit dit believable —
   banyak reviewer curiga data leak di split itu.
3. Target 55–60% adalah klaim yang jujur: **lebih baik dari ProtoNet
   dengan margin yang masuk akal**, achieved with purely local
   learning. Itu cerita yang publishable.

## Konsekuensi

- Setiap laporan WAJIB menyebut "Lake original split" + LINK ke split file.
- Baseline ProtoNet harus **direimplementasi** (T6.2.5), bukan diambil
  dari angka paper. Kalau implementasi kita sendiri dapat 55%, itu
  berarti pipeline kita benar dan target RAK harus di-upgrade.
- Split file di-commit ke repo supaya reviewer bisa verifikasi persis
  karakter mana yang masuk train vs test.
- Angka dari paper lain **tidak boleh** ditulis tanpa menyebutkan split mana yang dipakai.

## Alternatif yang ditolak

- **Rotation split** — ditolak, angka tidak trustworthy.
- **Belum diputuskan** — ditolak, karena memblokir semua baseline
  comparison di Phase 6 dan bisa meleset di bulan ke-15.