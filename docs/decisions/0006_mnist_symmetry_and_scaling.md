# ADR-0006: Keruntuhan simetri, enam bug terukur, dan penskalaan neuron

- Status: Accepted
- Date: 2026-10-22
- Affects: Phase 3 (T6.x), `src/rak/network/competitive.py`, `src/rak/learning/*`

## Konteks

Target proyek adalah akurasi MNIST setinggi mungkin tanpa backprop.
Acuan Diehl & Cook 2015 melaporkan ~89% pada 400 neuron / 3 epoch dan
~95% pada 6400 neuron / 15 epoch. Target yang diminta 98% — di atas
plafon acuan, jadi angka itu hanya mungkin kalau ada perbaikan di luar
acuan, dan tidak boleh diklaim sebelum terukur.

Selama Phase 3, akurasi berhenti di ~10% (setara tebakan) meskipun
semua kriteria yang kami punya hijau: 0 neuron mati, semua kelas
terwakili, dan bobot berubah selama training.

## Gejala

Run 100 neuron, 59.968 presentasi, 189 menit:

```
korelasi antar kolom            : 1.000
jarak L1 neuron 0 vs neuron 99  : 0.00
neuron per kelas                : [100, 0, 0, 0, 0, 0, 0, 0, 0, 0]
akurasi                         : ~10%
```

Semua neuron memiliki bobot identik. Jaringan tidak belajar apa pun.

## Enam bug, masing-masing dengan bukti pengukuran

### 1. Evaluasi tidak memakai intensitas adaptif

`evaluate()` memanggil `present_batch` dengan intensitas awal, bukan
`present_adaptive`. Pada intensitas 2, 98 dari 100 neuron tidak pernah
mencapai threshold setelah theta naik.

Bukti: 98/100 neuron sunyi saat evaluasi, sedangkan training pada
intensitas yang sama membuat neuron fire.

Perbaikan: evaluasi memakai `present_adaptive(learn=False)`.

### 2. STDP batch memakai rata-rata, bukan jumlah

`dw` dihitung dari rata-rata kontribusi 64 gambar dalam batch. Satu
presentasi gambar memberi sinyal 64x lebih lemah dari yang dimaksud.

Bukti: bobot bergerak ~64x lebih lambat dari perkiraan analitik;
setelah 59.968 presentasi bobot masih di sekitar nilai awal.

Perbaikan: pakai jumlah kontribusi, bukan rata-rata.

### 3. `outer(E[a], E[b])` menghapus sinyal koinsidensi

Versi pertama memperbarui bobot dengan outer product dari ekspektasi
trace, yaitu `E[a] x E[b]`. Itu sama dengan mengalikan rata-rata, bukan
rata-rata perkalian: neuron yang fire pada waktu berbeda tetap
menghasilkan produk besar.

Bukti: bobot menjadi hampir seragam (rata-rata 0.0995 = 78/784, persis
normalisasi kolom), tanpa struktur digit.

Perbaikan: satu matmul `pre^T @ (post2 * fired)` yang mempertahankan
koinsidensi per timestep.

### 4. `theta_batch` selalu mulai dari `theta_init`

`theta` di akhir presentasi ditimpa dengan rata-rata lintasan batch
yang dimulai dari 20 mV, sehingga `theta` tidak pernah terakumulasi
antar presentasi.

Bukti: `theta` terukur 20.0 +- 0.7 setelah 59.968 presentasi — sama
dengan `theta_offset`, artinya homeostasis tidak pernah bekerja.

Perbaikan: `theta_batch` dimulai dari `theta` yang sudah dipelajari.

### 5. Evaluasi mengubah bobot dan theta

`present_batch` dipanggil tanpa mematikan pembelajaran saat assignment
dan evaluasi. Assignment lalu dihitung dari jaringan yang sudah berubah
oleh data evaluasi itu sendiri.

Bukti: assignment `[99, 0, ...]` berubah menjadi `[72, 18, 3, ...]`
setelah `learn=False` ditambahkan.

Perbaikan: flag `learn` pada `present_batch`, `False` untuk assignment
dan evaluasi.

### 6. `theta_plus` tidak cocok dengan laju fire kita

`theta_eq = theta_plus * laju * tau_theta`. Acuan berjalan pada laju
~0.0125 spike/neuron/gambar (5 spike untuk 400 neuron). Nilai
`theta_plus = 0.05` pada laju kita menghasilkan `theta_eq` 32-115 mV,
sehingga evaluasi menuntut intensitas 10-12.

Bukti: run 100 neuron berakhir dengan intensitas evaluasi 10-12 dan
akurasi 14.8%.

Perbaikan: `theta_plus = 0.0125`, dan `theta_max = 40` sebagai batas
atas.

## Tiga perbaikan struktural

Enam bug di atas menjelaskan mengapa *tidak ada* pembelajaran. Setelah
semuanya diperbaiki, akurasi mencapai 28.64% pada 100 neuron — tapi
penskalaan ke 400 neuron justru turun ke 9.20%. Penyebabnya struktural.

### A. I -> E acak, dinormalisasi per kolom

Versi lama all-to-all seragam: setiap neuron eksitasi menerima inhibisi
yang identik. Kompetisi menjadi simetris sempurna, dan simetri itu
tidak bisa dipecahkan oleh STDP mana pun.

Acuan memakai matriks acak (`random/AiAe.npy`). Sekarang: konektivitas
acak `p = 0.5`, dan setiap **kolom** (satu neuron eksitasi) dinormalisasi
supaya jumlah bobotnya tepat `inhibition_strength`. Total inhibisi jadi
konstan terhadap ukuran jaringan dan terhadap jumlah neuron inhibitor
yang menyala (dalam ekspektasi).

Bukti: `col_corr` 0.975 (gagal) -> 0.141 (400 neuron, sweep 30 batch).

### B. `theta_max`

`theta` tumbuh mengikuti laju fire. Saat laju per neuron naik 2x
(0.103 -> 0.22 pada 400 neuron), `theta_eq` naik ke ~79 mV, evaluasi
menuntut intensitas 9.3, dan pada intensitas itu input menjadi seragam.

Bukti: run 400 neuron sebelum batas: intensitas 9.3, **semua 400 neuron
menjadi kelas 0**, akurasi 9.20%. Sesudah batas: intensitas 2.0,
assignment min 16 neuron/kelas, akurasi 44.40%.

`theta_max = 40` adalah penyimpangan dari acuan dan disengaja: acuan
tidak membutuhkannya karena laju fire-nya 17x lebih rendah.

### C. Laju fire per neuron harus konstan terhadap `n_e`

Sweep 400 neuron dengan konektivitas acak:

| strength | spike/presentasi | kelas | col_corr |
|---|---|---|---|
| 1600 | 83 | 10/10 | 0.141 |
| 2400 | 83 | 10/10 | 0.155 |
| 3200 | 82 | 10/10 | 0.146 |
| 4800 | 32.134 | 1/10 | 1.000 |

Inhibisi jenuh di 1600-3200 dan meledak di 4800. Nilai 1600 dipakai
untuk training penuh.

## Hasil terukur

| konfigurasi | akurasi | intensitas eval | assignment |
|---|---|---|---|
| 100n, 1 epoch, sebelum perbaikan | 14.80% | 10-12 | `[72,18,3,2,2,1,1,1]` |
| 100n, 1 epoch, sesudah perbaikan | **28.64%** | 3.7 | `[72,18,3,2,2,1,1,1]` |
| 400n, 1 epoch, sebelum A+B | 9.20% | 9.3 | `[400]` (semua kelas 0) |
| 400n, 1 epoch, sesudah A+B | **44.40%** | 2.0 | min 16/kelas |

## Konsekuensi

- Semua laporan harus menyertakan intensitas evaluasi. Akurasi tanpa
  intensitas tidak bisa dibandingkan: intensitas tinggi menghapus
  informasi bentuk dan membuat assignment runtuh.
- `inhibition_strength` hanya bermakna bersama `n_e`, `theta_max`, dan
  `inhib_connectivity`. Sweep lama yang tidak mencatat ini tidak bisa
  direproduksi.
- Target 98% tetap belum tercapai. Angka 89% (400n/3ep) adalah target
  berikutnya yang bisa diuji; 98% baru boleh diklaim kalau ada bukti
  pengukuran.
