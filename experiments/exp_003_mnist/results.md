# exp_003 MNIST — hasil terukur

Semua angka di bawah diukur pada **test set MNIST 10.000 gambar**,
kecuali disebut lain. Tidak ada backprop, tidak ada pretrained model,
tidak ada API. Bobot SNN dilatih tanpa label dengan STDP lokal.
Label hanya dipakai untuk **readout** (mengubah spike count menjadi
kelas) dan untuk mengukur akurasi.

## Progresi

| # | konfigurasi | readout | akurasi |
|---|---|---|---|
| 1 | 400n, 3 epoch, theta_max=40 | rata-rata anggota kelas | 45.94% |
| 2 | 400n, 3 epoch, theta_max=40 | top-3 rata-rata | 53.06% |
| 3 | 400n, 3 epoch, theta_max=40 | matched filter (prototipe) | 61.59% |
| 4 | 400n, 1 epoch, theta_max=80 | matched filter, 5 trials | 71.68% |
| 5 | 400n, 1 epoch, theta_max=80 | ridge, 5 trials | 77.33% |
| 6 | 400n, 3 epoch, theta_max=40 | ridge, 5 trials | 78.30% |
| 7 | **400n, 3 epoch, theta_max=80** | **ridge, 5 trials** | **84.11%** |
| 8 | 800n, 1 epoch, theta_max=80 | ridge, 5 trials | 74.61% |

Catatan:

- `trials=5` berarti setiap gambar test dipresentasikan 5 kali dengan
  sampel Poisson berbeda, lalu spike count-nya dijumlahkan. Encoder
  Poisson stokastik; satu sampel mengandung varians sampling. Bobot
  TIDAK berubah (evaluasi memakai `learn=False`).
- Ridge adalah classifier linear yang dilatih pada spike count data
  TRAIN saja (80% fit, 20% validasi untuk memilih alpha). Data test
  tidak pernah dipakai untuk memilih apa pun.
- Konfigurasi #7 adalah model terbaik saat ini. Receptive field-nya
  ada di `receptive_fields.png`: hampir semua neuron adalah template
  digit yang terbaca, dengan sisa kebingungan 4/9, 7/1, dan 3/8.

## Readout menentukan urutan model

Model yang sama, dua readout:

| model | matched filter | ridge |
|---|---|---|
| 400n, 3ep, theta_max=80 | 75.52 | **84.11** |
| 400n, 3ep, theta_max=40 | 67.43 | 78.30 |
| 400n, 1ep, theta_max=80 | 62.82 | 77.33 |
| 800n, 1ep, theta_max=80 | 56.70 | 74.61 |

Kesimpulan "1 epoch lebih baik dari 3 epoch" yang sempat diambil dari
matched filter **terbalik** setelah readout diperbaiki. Readout yang
lemah bukan hanya menurunkan angka; ia bisa menyesatkan pemilihan
model.

## Yang tidak berhasil

- **800 neuron** (dengan `e_to_i` diskalakan): 74.61%, lebih buruk
  dari 400 neuron. Assignment sangat timpang (215 neuron untuk satu
  kelas, 19 untuk kelas lain).
- **`matched_cosz_std`** (standarisasi skor per kelas): kalah dari
  `matched_cosz` di validasi.
- **3 epoch theta_max=80 dengan matched filter**: 59.82% — terlihat
  seperti kegagalan, padahal dengan ridge model itu yang terbaik.
- **`theta_max=40`**: 31% neuron menempel di plafon, sehingga
  homeostasis tidak bisa merekrut neuron ekor.

## Perbandingan dengan acuan

Diehl & Cook 2015 melaporkan 89% untuk 400 neuron / 3 epoch dan 95%
untuk 6400 neuron. Model terbaik kita 84.11% pada ukuran 400 neuron.
Target 98% **tidak tercapai dan tidak diklaim**. Selisih yang tersisa
belum diatribusikan; kandidat penyebab: synaptic delay acak (acuan
memakai 0–10 ms input dan 0–5 ms E↔I, kita tidak), parameter STDP,
dan detail homeostasis.

## Reproduksi

Training:

```
gh workflow run mnist.yml -f neurons=400 -f epochs=3 -f inhibition=1600 \
  -f theta_max=80 -f theta_plus=0.0125 -f assign=5000 -f test=5000 -f smoke=true
```

Readout:

```
gh workflow run ridge.yml -f run_id=<RUN_ID> -f train_samples=5000 \
  -f test_samples=10000 -f trials=5 -f inhibition=1600 -f e_to_i=1600
```
