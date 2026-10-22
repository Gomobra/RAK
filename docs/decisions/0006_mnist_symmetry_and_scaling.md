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

## Readout: assignment keras membuang informasi

`assign_classes` memaksa setiap neuron memilih satu kelas. Neuron
yang mirip 4 dan 9 hanya menyumbang ke salah satunya. Lima (lalu
delapan) aturan baca dibandingkan pada bobot yang SAMA, dipilih di
split validasi data TRAIN, lalu test diukur sekali:

| readout | validasi | test |
|---|---|---|
| mean (rata-rata anggota kelas) | 45.80 | 42.18 |
| max (neuron paling aktif) | 47.88 | 44.92 |
| margin (berbobot kemurnian) | 52.40 | 47.46 |
| topk (rata-rata 3 teratas) | 55.36 | 50.82 |
| binary (proporsi yang fire) | 42.96 | 39.50 |
| matched (dot product prototipe) | 55.92 | 51.76 |
| matched_cos (prototipe dinormalisasi) | 57.80 | 55.18 |
| **matched_cosz** (z-score gambar + prototipe) | **63.68** | **59.02** |

Dengan 5.000 gambar kalibrasi dan 10.000 test, readout yang sama
memberi **61.59%** (naik dari 45.94% dengan `mean`).

Varian "column-centered" yang secara teori lebih rapi (blob berbobot
nol) justru KALAH (55.18 vs 63.68 di validasi), jadi yang dipakai
adalah varian yang menang di validasi, bukan yang terdengar lebih
benar.

## Receptive fields setelah 3 epoch

`experiments/exp_003_mnist/receptive_fields.png` (400 neuron, urut
kelas lalu selektivitas) menunjukkan dua populasi:

- **Template digit bersih** — 0, 1, 6, 9 sangat jelas; 3 dan 7
  bercampur dengan 8/9.
- **Pola checkerboard** di ekor tiap kelas — neuron yang tidak pernah
  memenangkan kompetisi. Kelas 5 hampir seluruhnya di ekor ini, dan
  recall-nya 25%.

Ini menjelaskan mengapa akurasi masih jauh dari 89%: bukan karena
tidak ada fitur, tapi karena sebagian neuron tidak terpakai.

## Hasil terukur

| konfigurasi | akurasi | intensitas eval | assignment |
|---|---|---|---|
| 100n, 1 epoch, sebelum perbaikan | 14.80% | 10-12 | `[72,18,3,2,2,1,1,1]` |
| 100n, 1 epoch, sesudah perbaikan | **28.64%** | 3.7 | `[72,18,3,2,2,1,1,1]` |
| 400n, 1 epoch, sebelum A+B | 9.20% | 9.3 | `[400]` (semua kelas 0) |
| 400n, 1 epoch, sesudah A+B | **44.40%** | 2.0 | min 16/kelas |
| 400n, 3 epoch, readout mean | 45.94% | 2.0 | min 10/kelas |
| 400n, 3 epoch, readout topk | 53.06% | 2.0 | min 11/kelas |
| 400n, 3 epoch, readout matched_cosz | 61.59% | 2.0 | min 11/kelas |
| 400n, 1 epoch, theta_max=80 | 62.76% | 2.0 | min 21/kelas |
| 400n, 1 epoch, theta_max=80, 5 trials | **71.68%** | 2.0 | min 21/kelas |
| 400n, 3 epoch, theta_max=80 | 59.82% | 2.0 | min 14/kelas |

## theta_max: plafon yang mengikat

Dengan `theta_max=40`, 31% neuron menempel di plafon. Homeostasis
tidak bisa menekan pemenang lagi, jadi neuron yang sudah kuat terus
menang dan neuron di ekor tidak pernah belajar (pola checkerboard di
gambar receptive field).

Menaikkan `theta_max` ke 80:

- spike/presentasi turun 50.4 -> 32.2 (lebih dekat ke acuan),
- assignment jauh lebih merata: `[85,85,37,36,33,28,28,25,22,21]`
  versus `[115,51,50,...]`,
- recall kelas 5 naik 25.0% -> 50.7%,
- akurasi 1 epoch **62.76%** (mengalahkan 3 epoch theta_max=40).

Tetapi 3 epoch dengan `theta_max=80` justru turun ke 59.82%: theta
terus naik, laju turun ke 22.6 spike/presentasi, dan sebagian kelas
kehilangan neuron (kelas 9 hanya 14). Jadi 1 epoch adalah titik
terbaik yang terukur; menambah epoch tidak otomatis menambah akurasi.

## Encoder Poisson: satu sampel itu berisik

`encode_poisson` membangkitkan spike secara stokastik. Satu presentasi
per gambar berarti pola spike yang dipakai classifier mengandung
varians sampling, bukan hanya informasi bentuk.

Menjumlahkan 5 presentasi untuk gambar yang sama (tanpa menyentuh
bobot, tanpa label) menaikkan akurasi 62.76% -> **71.68%** pada test
10.000 gambar. Ini bukan trik evaluasi yang menyembunyikan kelemahan:
classifier tetap sama, hanya estimasi pola yang lebih stabil. Biaya
waktunya linear terhadap jumlah percobaan.

Recall masih timpang: kelas 9 terendah (49.2%), kelas 0/1/2 tertinggi
(~87%).

## Readout linear: prototipe bukan batas atas

Prototipe kelas adalah classifier linear dengan bobot tulisan tangan.
Ridge regression pada spike count yang sama, dilatih di TRAIN,
hyperparameter dipilih di validasi TRAIN, diukur sekali di TEST:

| model | matched_cosz | ridge |
|---|---|---|
| 400n, 3ep, theta40 | 67.43 | **78.30** |
| 400n, 1ep, theta80 | 62.82 | 77.33 |
| 400n, 1ep, theta80 (fit 10k) | 62.74 | 77.86 |
| 800n, 1ep, theta80 | 56.70 | 74.61 |

Ridge menambah ~10 poin. Ini tetap sah: SNN tidak menerima label
sama sekali; label hanya dipakai untuk fitting readout, persis
seperti assignment kelas pada acuan. Yang berubah hanya cara
mengubah spike count menjadi label.

Temuan penting: model yang menang di matched filter (theta80 1ep)
KALAH dari model 3 epoch theta40 begitu readout-nya ridge. Urutan
model berubah menurut readout, jadi memilih model dari satu readout
saja bisa menyesatkan.

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
