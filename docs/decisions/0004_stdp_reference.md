# ADR-0004: STDP acuan — 2-trace Norse di 2A, 3-trace Diehl-Cook di Phase 3

- Status: Accepted
- Date: 2026-10-15
- Affects: T2.2 (Phase 2A), T6.1 (Phase 3)

## Konteks

Ada **dua keluarga STDP** yang berbeda dan keduanya dipakai di literatur.
Rancangan awal Phase 2 hanya menyebut "trace-based STDP" tanpa
menentukan yang mana.

### Keluarga 1 — Norse, 2 trace

```
dW_ij = A_plus(W_ij)  * x_pre_i  * z_post_j
      - A_minus(W_ij) * z_pre_i  * x_post_j
```

Trace: `x_pre` (n_pre,), `x_post` (n_post,) — **per-neuron**.
Empat varian A: additive · additive_step · multiplicative_pow ·
multiplicative_relu.

### Keluarga 2 — Diehl & Cook 2015, 3 trace

```
on pre-spike :  pre = 1;  w -= nu_pre  * post1
on post-spike:  post2before = post2;  w += nu_post * pre * post2before
                post1 = 1;  post2 = 1;
```

Trace: `pre` (τ=20ms) · `post1` (τ=20ms, untuk LTD) · `post2` (τ=40ms,
untuk LTP) · ditambah `post2before` sebagai variabel bayangan.

Mengapa butuh `post2before`: kalau `post2` di-set 1 **sebelum**rumus LTP
dihitung, maka `A_plus` selalu pakai `post2 = 1.0` dan dependensi
`τ_post = 40ms` hilang total. Variabel bayangan itu yang membuatnya benar.

## Perbedaan yang penting

| | Norse (2-trace) | Diehl-Cook (3-trace) |
|---|---|---|
| Trace | 2, per-neuron | 3, per-neuron + 1 bayangan |
| τ LTP | `tau_pre` | `tau_pre` |
| τ LTD | `tau_post` | `tau_post_2` |
| η | `eta_plus`, `eta_minus` | `nu_ee_pre=1e-4`, `nu_ee_post=1e-2` |
| Batas W | clipping opsional | hard bound [0, 1] + normalisasi kolom |
| Varian | 4 (A_plus/A_minus) | 4 (exponential/power-law/etc, via parameter) |

Keduanya **bukan** satu hal yang bisa ditukar. Diehl-Cook memisahkan
time constant LTP dan LTD; Norse memakai satu `tau_post` untuk keduanya
(meski bisa dibedakan dengan `tau_pre_ms` ≠ `tau_post_ms`).

## Keputusan

**Phase 2A: implementasi 2-trace gaya Norse** (4 varian), seperti
rancangan PRD.

Alasan:

1. **Surface area lebih kecil.** 3 trace + variabel bayangan = lebih
   banyak tempat bug. Dan Phase 1 sudah taught kita bahwa "sederhana"
   pun bisa salah.
2. **Nilai leak yang sama.** 2-trace sudah bisa mereproduksi window
   asimetris hanya dengan `tau_pre_ms != tau_post_ms`.-tested di
   `test_window_asymmetric_when_tau_differ`.
3. **Phase 3 tinggal menambah**, bukan mengganti. Kita sudah butuh
   bentuk yang punya `tau_pre`/`tau_post` terpisah sebelum pindah ke
   Diehl-Cook.

**Phase 3: tambah varian 3-trace sebagai backend kedua**, di balik
satu interface yang sama (`stdp_step`).

Konsekuensi yang diterima: ada dua implementasi STDP di repo. Itu
memang duplikasi, tapi kalau keduanya di-flag eksplisit dan diuji
dengan window analitik yang sama, bedanya justru jadi data — bukan
kerusakan.

## Konvensi tanda yang di-pin

```
dt > 0   post TERLAMBAT  -> potensiasi  (daerah positif window)
dt < 0   post LEBIH AWAL  -> depresi     (daerah negatif window)
dt = 0   simultan        -> nol
```

Sebagian besar plot STDP di literatur memakai tanda kebalikan (sumbu
y dibalik). Yang berbeda adalah konvensi, bukan rumusnya. Ours
sejalan dengan `stdp_step` supaya test analitik bisa membandingkan
window dengan hasil simulasi tanpa flipping.

## Urutan internal timestep (WAJIB dipatuhi)

```
untuk setiap timestep t:
    1. trace *= gamma            (decay DULU)
    2. dW dari trace yang sudah meluruh, dikali z_pre[t] / z_post[t]
    3. W = clamp(W + dW)
    4. trace += z               (bump SESUDAH)
```

Dua aturan ini mengunci correctness dan keduanya sudah punya test:

- **decay sebelum update.** Kalau bump-dulu, trace baru meluruh
  `dt - 1` kali dan window diskretnya menyimpang dari rumus eksponensial
  sebesar satu faktor gamma. Diperiksa oleh
  `test_stdp_window_matches_empirical` (toleransi 2e-4).
- **bump setelah update.** Kalau bump-dulu, spike pre dan post di
  timestep yang sama saling "melihat" spike-nya sendiri, dan
  kredit palsu diberikan. Diperiksa oleh
  `test_simultaneous_spike_is_noop`.

## Trace per-neuron, bukan per-synapse

Trace disimpan sebagai vektor `(n_pre,)` dan `(n_post,)`, **bukan**
matriks `(n_pre, n_post)`.

Alasannya: `dW_ij` hanya butuh `x_pre_i` dan `x_post_j` — tidak ada
nilai yang berbeda untuk tiap pasangan synapse. Menyimpan per-synapse
memakai memori n_pre × n_post (bukan n_pre + n_post) tanpa menambah
informasi apa pun.

Ini diverifikasi secara ketat: `stdp_step` **menolak** trace berbentuk
matriks dengan pesan error eksplisit, supaya bug ini tidak bisa
disembunyikan di kemudian hari
(`test_trace_shape_validation_is_strict`).

## Referensi

- Norse: `norse/torch/functional/stdp.py` — `STDPState`, `STDPParameters`
- Diehl & Cook 2015: `peter-u-diehl/stdp-mnist`, `Diehl&Cook_spiking_MNIST.py`
- Marker STDP asli: Bi & Poo, J Neurosci 1998