# exp_002 — Learning Rules + Homeostasis (Phase 2)

Verifikasi T2.1, T2.2, T1B.1, T1B.2b, T1B.2, T1B.3, dan gate test
homeostasis. PRD v1.2 §8 Phase 2A + 2B.

## Yang diuji

| Task | Test | Jumlah |
|---|---|---|
| T2.1.1-T2.1.5 Hebbian / BCM / Oja | `test_hebbian.py` | 23 |
| T2.2 Trace-based STDP (4 varian) | `test_stdp.py` | 49 |
| Guard konfigurasi STDP + integritas figure | `test_stdp_config.py` | 12 |
| **GATE: STDP bisa belajar asosiasi** | `test_stdp_learning.py` | 12 |
| T1B.2b Refractory period | `test_refractory.py` | 21 |
| T1B.1 Intrinsic plasticity | `test_homeostasis.py` | 19 |
| T1B.2 Synaptic scaling (hard + soft) | `test_scaling.py` | 23 |
| T1B.3 Lateral inhibition | `test_inhibition.py` | 10 |
| **GATE: homeostatic gain > 1** | `test_homeostatic_gate.py` | 11 |
| **Subtotal Phase 2** | **9 file** | **180** |

### Total suite: 281 test dari 16 file

Angka "246" yang tercatat sebelumnya menghitung file Phase 2 saja (9
file), padahal suite-nya 16 file. Rinciannya:

| Fase | File | Test |
|---|---|---|
| Phase 1 — LIF, Adaptive, Izhikevich, SpikeLayer, trace | 6 | 95 |
| Phase 0 — seeds, edge case numerik | 2 | 22 |
| Phase 2A — refractory, Hebbian, STDP | 3 | 93 |
| Phase 2B — intrinsic, scaling, inhibition | 3 | 52 |
| Gate — homeostasis, konfigurasi STDP, belajar asosiasi | 3 | 35 |
| **Total** | **16** | **281** |

## Cara reproduce

```bash
pip install -e ".[dev]"
pytest                                  # 281 test, ~4 menit
pytest tests/test_stdp_learning.py      # gate: STDP bisa belajar
pytest tests/test_homeostatic_gate.py     # gate: homeostasis menahan drift
```

## Hasil gate test

Arsitektur: 64 input → 32 eksitasi, LIF + inhibisi lateral + STDP.
`eta_plus = 2e-3`, `eta_minus = 1e-4` (rasio 20×).

| Konfigurasi | Drive akhir | w_max | Status |
|---|---|---|---|
| STDP saja | **2047.6** | 1.00 | semua synapse saturasi |
| STDP + homeostasis | **135.3** | 0.15 | tertahan |

Drive tanpa homeostasis: 328 → 467 → 678 → 1049 → 1611 → **2048**
(plafon keras `n_in × n_out × w_max`).

**Ini reproduksi langsung temuan Zenke et al. 2013:** Hebbian plasticity
tanpa homeostatic control menghasilkan positive feedback yang tidak
tertangani.

## Temuan utama Phase 2

### 1. η asimetris itu WAJIB, bukan opsional

Dengan `eta_plus == eta_minus`, STDP punya **fixed point** — LTP dan LTD
semenyakalkan sebelum bobot mencapai batas. Drive stabil di ~331.

Konsekuensi berbahaya: eksperimen homeostasis dengan η simetris akan
menyimpulkan **"homeostasis tidak diperlukan"**, dan itu salah.

Diehl-Cook memakai rasio ~100×. Ada test khusus yang mengunci ini
(`test_equal_eta_stdp_is_self_limiting`) supaya tidak terulang.

### 2. Tiga mekanisme, tiga skala berbeda

| Mekanisme | Yang dikontrol | BUKAN |
|---|---|---|
| theta (intrinsic) | DAYA — laju total populasi | Kesamaan antar neuron |
| column scaling | DRIVE per neuron | Batas atas |
| lateral inhibition | KOMPETISI | Rate absolut |

Bukti: pada input acak yang sudah seimbang, CV laju justru **naik** dari
0.014 (LIF) jadi 0.036 (Adaptive). Theta adalah rem global, bukan
penyamarataan. Yang menyamaratkan adalah column scaling.

Lihat ADR-0005.

### 3. θ_ss = β_θ × rate × τ_θ

Rumus tuning utama Phase 3. Yang penting menjaga **produknya** kecil,
bukan salah satu saja — dibuktikan di
`test_matched_beta_times_tau_gives_matched_rate`.

Konsekuensi praktis: τ_θ panjang harus dikompensasi dengan β_θ kecil.
Kalau tidak, θ menembus jangkauan v dan neuron mati permanen
(`test_theta_can_kill_neurons_if_beta_too_big`).

### 4. Bug yang ditemukan saat implementasi

| Bug | Gejala |
|---|---|
| `0.0 * inf = nan` di masking aritmetika | Neuron diam **selamanya** setelah spike pertama (refractory) |
| Trace STDP per-synapse | `(n_in, n_out)` — boros, tidak dipakai rumus mana pun. Sekarang per-neuron, dan **ditolak** dengan pesan eksplisit |
| Trace decay di timestep salah | Decay setelah bump → window menyimpang satu faktor γ dari rumus eksponensial |
| `soft_normalize` tanpa target | `W /= mean^p` justru **memperbesar** neuron yang kurang aktif |
| Neuron inhibitor di-init v=0.0 | v_th = −0.4 → inhibitor fire tiap timestep, matikan seluruh jaringan |
| Skala inhibisi `1/√N` | Total inhibisi per neuron tumbuh √N, bukan konstan. Seharusnya `1/N` |

### 5. Klaim yang dikoreksi

**"Varian multiplicative tidak bisa melewati batas"** — salah. Satu
langkah dengan η besar masih bisa lewat. Yang benar: bobot **berosilasi
di sekitar** `w_max`, bukan konverge monoton (karena trace jenuh membuat
langkah update ikut membesar).

**"STDP meledak tanpa normalisasi"** — nuansa. Kalau pre dan post fire
**frekuensi sama**, LTP dan LTD saling menyeimbangkan dan w mencapai fixed
point. Justru itu stabilizing role STDP yang sesungguhnya. Divergensi
butuh asimetri.

## Figures

Regenerasi semua dengan `python -m rak viz-phase2`.

| File | Isi |
|---|---|
| `figures/01_stdp_window.png` | Keempat varian, tau simetris |
| `figures/02_stdp_window_asym.png` | tau_pre=50 ms, tau_post=10 ms — window asimetris |
| `figures/03_stdp_window_variants.png` | w = 0.1 / 0.5 / 0.9 — perbedaan varian multiplicative |
| `figures/04_theta_adaptation.png` | Peluruhan theta (3 tau) + efek beta ke laju firing |
| `figures/05_homeostatic_gate.png` | **Figure utama** — drive dengan vs tanpa homeostasis |
| `figures/06_column_normalization.png` | Kolom diratakan sebelum/sesudah + heatmap bobot |
| `figures/07_association_learning.png` | **GATE** — kurva belajar asosiasi A->B |

Konvensi tanda window: `dt > 0` = post terlambat = **potensiasi**,
`dt < 0` = post lebih awal = depresi. Sebagian literatur memakai tanda
kebalikan; yang berbeda konvensinya, bukan rumusnya.

Catatan: dengan `w_ref` tunggal dan `mu = 0`, keempat varian IDENTIK —
`(w_max - w)^0 = 1` jadi ekuivalen additive. Figure 03 memakai tiga
nilai w supaya perbedaannya terlihat.

### Figure 03 — pernah menampilkan algoritma yang salah

Figure ini mengklaim menunjukkan perbedaan varian multiplicative pada
tiga nilai bobot. Kenyataannya **ketiga kurvanya identik persis**.

Penyebabnya dua bug bertumpuk:

1. **CLI mengirim `STDPConfig(mu=1.0)` tanpa `algorithm=`.** Default-nya
   `"additive"`, jadi `mu` dibuang diam-diam. Tiga panel yang dimaksud
   "multiplicative" sebenarnya additivesemu.
2. **`plot_stdp_window` salah memakai variabel loop.** `for ax, algo in
   zip(axes, ALGORITHMS)` memakai `algo` CUMA sebagai label, lalu
   meneruskan `cfg` apa adanya ke `stdp_window` — yang mendispatch ke
   `cfg.algorithm`. Jadi keempat panel menampilkan algoritma yang SAMA
   dengan empat nama berbeda.

Implementasi `A_plus`/`A_minus`-nya sendiri **benar** sepanjang waktu
(`A_plus` = 0.0009 / 0.0005 / 0.0001 untuk w = 0.1 / 0.5 / 0.9). Yang
bohong adalah figurenya.

Perbaikan:

- `STDPConfig.__post_init__` sekarang **menolak** `mu != 0` pada varian
  yang tidak membacanya. Bug ini adalah no-op senyap; kalau hanya
  jadi warning, akan terulang.
- `viz.panel_configs(cfg)` jadi seam yang bisa dites: satu `STDPConfig`
  turunan per varian. `plot_stdp_window` memakainya.
- `test_stdp_config.py` (12 test) mengunci keduanya.

Figure sekarang jauh lebih jujur: panel `additive` dan `additive_step`
memang **tidak** bergantung pada w (kurvanya identik — itu kebenaran),
sedangkan dua panel multiplicative benar-benar terpisah.

### GATE baru — STDP bisa belajar asosiasi

Sebelum ini Phase 2 hanya membuktikan dua hal: window STDP benar secara
matematis, dan STDP + homeostasis tidak ledak. Keduanya soal **mekanisme**.
Tidak ada test yang menyatakan "jaringan berubah karena pengalaman".

`test_stdp_learning.py` (12 test) mengajari jaringan 2x2 asosiasi
A -> B dan B -> A:

| | A memicu B | B memicu B (salah) |
|---|---|---|
| belum diajar | 0/60 | 0/60 |
| diajar A -> B | **60/60** | 0/60 |

Yang paling penting bukan ini, tapi kontrolnya:

- **Anti-causal tidak mengajar apa pun.** Kalau post fire lebih dulu,
  bobot tidak naik (terverifikasi: 0.05000 -> 0.04992, LTD). Ini yang
  membuktikan aturannya soal *waktu*, bukan sekadar "dua neuron yang
  sama-sama aktif jadi makin kuat".
- **`eta_plus == eta_minus` tidak bisa mengajar sama sekali.** Fixed
  point ada di w ~ 0.83, dan dengan satu kanal input `v_ss = w` — jadi
  bobot itu tidak pernah menembus `v_th = 1.0`. Konsekuensi ADR-0005
  yang baru terlihat di level task, bukan di level window.

### Temuan tambahan — residual trace meloloskan causality

Uji anti-causal pertama **gagal**: bobot naik ke 2.03 malah, bukan
turun. Penyebabnya bukan bug STDP, tapi sifat nyata trace:

    trial sebelumnya menyisakan trace_pre[0] = 0.142
    trial berikutnya t=0, z_post[1] menyala
      -> LTP = 5e-2 * 0.142 = 7.1e-3
    sementara LTD di t=5 cuma 1e-4 * 0.78 = 7.8e-5

LTP residual **9x lebih besar** dari LTD yang seharusnya menetralkannya.
Trace yang bocor lintas trial sama sekali tidak melihat causality — dia
cuma tahu "ada aktivitas pre yang baru saja terjadi".

Ini perilaku yang benar secara biologis (justru itulah cara STDP menangkap
pola temporal), tapi sebagai kontrol causality hasilnya kabur. Kontrol
anti-causal sekarang memakai `reset_traces=True`.

Konsekuensi untuk Phase 3: **jarak antar presentasi bukan parameter
kosmetik.** Kalau pola MNIST disajikan tanpa jeda yang memadai,
presynaptic trace dari frame sebelumnya akan menyalakan LTP pada
kombinasi input yang salah.

### Figure 04 — dua eksperimen terpisah, sengaja tidak digabung

Percobaan pertama menggabungkan peluruhan theta dan efek beta dalam satu
run. Hasilnya tidak informatif: karena `beta` diskalakan dengan `1/tau`,
produk `beta * tau` tetap sama, jadi ketiga kurvanya identik.

Sekarang dipisah:

- **Peluruhan theta** — theta mulai dari 1.0, arus **nol**, jadi tidak
  ada spike sama sekali. Kurvanya jadi murni laju peluruhan, dan skala
  log membuat perbedaan tau terlihat jelas (5 s meluruh ke 0.02; 1 jam
  dan 2.8 jam nyaris datar).
- **Efek beta** — tau tetap, beta divariasikan. Ini bukti ADR-0005:
  beta=0 → 571 spike, beta=0.05 → 22, beta=0.2 → 7.

Raster di panel 4 di-zoom ke 1.5 detik pertama. Pada 20 detik penuh,
beta=0 fire tiap ~35 ms dan rasternya jadi blok solid yang tidak
membandingkan apa pun — sumbu biner 0/1 jadi tidak berguna kalau
seluruh panel terisi.

### Figure 06 — matriks sengaja dibuat tidak seimbang

Figure ini **tidak** memakai bobot dari run STDP sungguhan, dan itu
keputusan yang butuh penjelasan.

Awalnya dipakai. Hasilnya membingungkan: CV hanya 0.06, dan justru
**mengecil** kalau input dibuat heterogen (0.014). Setelah dicek, itu
bukan salah plotting:

1. STDP di jaringan **1 lapis** dengan input homogen sangat seragam —
   tiap neuron postsynaptic dapat total input yang sama, jadi kolomnya
   memang sudah seimbang sejak awal.
2. Input heterogen justru menyembuhkan: neuron yang dapat lebih banyak
   fire lebih sering, theta-nya naik, lalu rate-nya ditekan balik.
   Persis ADR-0005 — theta bekerja sebagai pembatas daya.

Imbalance yang realistis butuh struktur **berlapis**: kolom layer
pertama tidak seimbang kalau bobot ke layer kedua tidak seimbang.
Belum ada di Phase 2.

Jadi figure ini memakai matriks dengan faktor lognormal per kolom
(sigma = 0.6) untuk menunjukkan operasi normalisasinya secara langsung:

| | CV | w_max | w_min |
|---|---|---|---|
| sebelum | 0.724 | 2.344 | 0.015 |
| sesudah | 0.000 | 0.673 | 0.063 |

Heatmap memakai `vmax` yang sama di kedua panel supaya perbandingan
kompresinya jujur — kolom yang blaze sebelum normalisasi terlihat
menerang, dan sesudahnya melebar rata.

Satu koreksi yang perlu dicatat: dengan `target_sum` = drive rata-rata,
`w_max` **hampir tidak berubah**. Yang diratakan adalah tinggi tiap
kolom, bukan besarannya. Menulis "skala turun" di judul figure
awalnya adalah klaim yang salah.

## Yang belum dikerjakan

Sengaja ditunda ke Phase 3+:

- Conductance-based synapse (Diehl-Cook pakai ini — bukan current-based)
- STDP 3-trace ala Diehl-Cook (ADR-0004)
- Integrasi homeostasis ke `SpikeLayer` sebagai loop training utuh
- Konsolidasi / replay

Yang sudah siap dipakai Phase 3: `scaling.target_sum_for()` (default 78),
`intrinsic.TIME_SCALES['diehl_cook_2.8h']`, dan inhibitory factory dengan
parameter yang benar.