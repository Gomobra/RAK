# exp_001 — Neuron Models (Phase 1)

Verifikasi T1.1–T1.5 dari PRD v1.1.

## Yang diuji

| Task | Status | Test |
|---|---|---|
| T1.1.1 LIF di PyTorch | ✅ | `test_lif.py::test_lif_runs_and_shapes` |
| T1.1.2 Input konstan → rate stabil | ✅ | `test_constant_input_gives_stable_rate` |
| T1.1.3 Input bervariasi → pattern berubah | ✅ | `test_varying_input_changes_spike_pattern` |
| T1.1.4 Plot V(t) + spike train | ✅ | `rak viz` → `figures/01_lif.png` |
| T1.1.5 Parameter tuning τ, v_th, v_reset | ✅ | `test_smaller_tau_gives_higher_rate`, dst. |
| T1.2.1 Variabel adaptasi θ | ✅ | `test_adaptive_lif.py::test_adaptive_state_has_theta` |
| T1.2.2 θ naik saat spike, lalu decay | ✅ | `test_theta_increases_on_spike`, `test_theta_decays_without_spike` |
| T1.2.3 Rate menurun saat input konstan | ✅ | `test_rate_declines_under_sustained_input` |
| T1.2.4 Bandingkan dengan LIF biasa | ✅ | `test_adaptive_rate_lower_than_plain_lif` → `figures/03_comparison.png` |
| T1.3.1 Persamaan Izhikevich | ✅ | `test_izhikevich.py` |
| T1.3.2 Mode firing biologis | ✅ | `test_each_firing_mode_produces_spikes`, `test_firing_modes_differ` |
| T1.3.3 Kapan pakai LIF vs Izhikevich | ✅ | `docs/theory.md` §7 |
| T1.4.1 SpikeLayer | ✅ | `test_layer.py` |
| T1.4.2 State V, θ, spike_history, trace | ✅ | `test_state_contains_traces` |
| T1.4.3 Forward T timestep | ✅ | `test_forward_shapes` |
| T1.4.4 Reset state antar batch | ✅ | `test_reset_state_zeroes` |
| T1.4.5 Benchmark throughput | ✅ | `rak bench` (di bawah) |
| T1.5 Spike trace + Poisson | ✅ | `test_trace.py` (17 test) |

## Cara reproduce

```bash
pip install -e ".[dev]"
pytest                      # 101 test
rak selfcheck               # smoke test
rak viz                     # regenerate figures/
rak bench --n-in 100 --n-out 100 --steps 100 --repeats 5
```

## Hasil benchmark (T1.4.5)

Di Termux/Android aarch64, torch 2.14.1, 8 thread, **tanpa GPU**.
Input: `--input mnist-like` (statistik MNIST sintetis, bukan data asli).

```
$ rak bench --n-in 784 --n-out 400 --steps 350 --repeats 3

config       : 784in -> 400out, 350 steps, batch 1, input=mnist-like
elapsed      : 0.827s over 3 repeats
neuron-upd/s : 507,639
spikes/s     : 15,777
spike density: 0.0311  (SynOps per cell = 0.0311)
neurons fired: 400/400
synapses     : 313,600
```

Density 3.11% itu angka yang **memang dipakai** untuk klaim sparsity:
1 dari 32 sel neuron aktif. Bandingkan dengan ANN dense yang 100%
aktif di setiap timestep.

**Catatan penting:** benchmark ini memakai input **sintetis** dengan
statistik menyerupai MNIST (~20% piksel menyala, intensitas [0.05, 1.0]),
BUKAN MNIST asli — lingkungan ini tidak punya akses jaringan untuk
mengunduhnya. Untuk benchmark MNIST sungguhan, pakai `rak bench` dari
Codespaces dengan dataset asli (Phase 3).

Angka throughput absolut **tidak boleh** dibandingkan dengan mesin lain.
Yang penting adalah urutan besaran: ~5×10⁵ neuron-update/detik di CPU
tanpa GPU. Itu yang membuat NFR-01 (4 GB, CPU) realistis.

## Temuan penting fase ini

### 1. Bentuk implisit LIF bukan pilihan gaya

Bentuk Euler eksplisit yang biasa ditulis di literatur tidak stabil
kalau `dt > tau`. Bentuk implisit (`alpha = 1 - dt/tau`) selalu `< 1`
dan aman untuk semua timestep. Detail: `docs/theory.md` §1.

### 2. Izhikevich: bentuk diskret yang beredar bukan hasil integrasi

Bentuk `0.96*v + 0.40*w - v^3 + 0.556*w + I` yang banyak di-copy blog
memiliki suku `-v³` yang **tidak bisa muncul** dari Euler atas ODE
kuadratik Izhikevich. Kami memakai bentuk ODE kanonik (Brian2):

```
dv/dt = 0.04v^2 + 5v + 140 - w + I
dw/dt = a(bv - w)
```

Bonus: titik istirahatnya bisa dihitung analitik (`v = -70`,
`w = -14` — titik **coupled**, bukan akar `0.04v²+5v+140=0` yang
yang biasa diambil orang secara keliru), dan `c = -65` adalah nilai **reset**,
bukan resting.

Mode firing yang tervalidasi:

| Mode | a | b | c | d | I | ISI |
|---|---|---|---|---|---|---|
| Regular spiking | 0.02 | 0.20 | -65 | 8 | 10 | 65 ms (15 Hz) |
| Chattering | -0.02 | 0.20 | -50 | 2 | 15 | 3.5 ms |

### 3. Kalibrasi bobot SpikeLayer itu wajib, bukan opsional

Untuk LIF implisit, steady-state membrane = `v_leak + I` **persis**
(`v_ss = v_leak + β(I + v_leak)/(1−α)`). Kalau bobot di-random tanpa
kalibrasi, `v_ss` jatuh di bawah threshold dan **layer diam total** —
density 0. Temuan ini terlihat dari figure pertama: raster kosong.

Kalibrasi sekarang: bobot discale agar `v(calib_value) ≈ 1.5 × v_th`.
`calib_value` default **0.12** — itu intensitas **rata-rata seluruh
piksel termasuk yang nol** (MNIST asli ≈ 0.13), bukan intensitas per-piksel
menyala (~0.5) dan bukan full scale (1.0).

Kenapa mean dan bukan per-piksel: current per timestep dijumlahkan dari
SEMUA piksel yang spike pada timestep itu. Yang menentukan apakah neuron
fire adalah rata-ratanya. Versi pertama kalibrasi ke 0.25 (ngira itu
intensitas per-piksel) menghasilkan density 0.0001 di input MNIST-like —
hanya 7 dari 400 neuron yang fire. Salah 20×.

Hasil setelah dikoreksi (784→400, T=350, input MNIST-like):

| calib_value | E[w] | density | neuron fire |
|---|---|---|---|
| 0.25 | 0.1200 | 0.0001 | 7/400 |
| 0.15 | 0.2001 | 0.0160 | 381/400 |
| **0.12** | **0.2501** | **0.0311** | **400/400** |
| 0.10 | 0.3001 | 0.0442 | 400/400 |
| 0.08 | 0.3752 | 0.0629 | 400/400 |

Respons **gradasi** (784→100, tiap piksel di-isi nilai konstan):

| Intensitas | density | neuron fire |
|---|---|---|
| 0.05 | 0.0000 | 0/100 |
| 0.13 | 0.0019 | 42/100 |
| 0.25 | 0.0437 | 100/100 |
| 0.50 | 0.1153 | 100/100 |
| 1.00 | 0.2406 | 100/100 |

### 4. Adaptasi θ: mekanismenya threshold yang naik, bukan tegangan yang naik

Figure `03_adaptation_detail.png` memisahkan empat kuantitas, dan itu
yang membuat mekanismenya jelas:

```
V          : 0.12 → 1.00 → reset → ... → 1.56      (dari atas)
θ          : 0.00 → 0.560, naik tiap spike, meluruh pelan
v_eff      : 1.00 → 1.560   (garis putus-putus = v_th = 1.00)
spike      : 16 spike, jaraknya MAKIN LAMBAT (LIF biasa: 25 spike)
```

Yang sebenarnya dinaikkan adalah **threshold efektif** `v_eff = v_th + θ`,
bukan tegangan membran. Kalau hanya melihat panel V, orang akan salah
mengira adaptasi terjadi karena "V makin tinggi". Panel `v_eff` yang
membuktikannya: puncak V di panel 1 **mengikuti** garis v_eff di panel 3
sampai ke digit terakhir.

Mekanismenya:

| Timestep | Yang terjadi |
|---|---|
| 1 | θ decay sedikit (α = 1 − dt/τ_adapt ≈ 0.998) |
| 2 | v dihitung: `v = α·v + β·(I + v_leak)` |
| 3 | spike kalau `v ≥ v_th + θ` |
| 4 | kalau spike: v di-reset ke v_reset, **lalu θ += β_θ** |
| 5 | θ meluruh ke arah θ₀ selama ~τ_adapt |

Urutannya penting: θ dinaikkan **setelah** spike, bukan sebelumnya.
Kalau dinaikkan dulu, neuron yang baru fire akan langsung menaikkan
threshold-nya sendiri di timestep yang sama — coupling yang tidak
diinginkan. Ada test yang mengunci urutan ini.

**Hubungan dengan Zenke 2013:** τ_adapt adalah "seberapa lama neuron
ingat bahwa dia terlalu aktif". Diuji:

| τ_adapt | spike akhir per 1000 timestep (input 2.0) |
|---|---|
| 500 ms | 18 |
| 2000 ms | 10 |
| 5000 ms | 4 |

(Rasio 4.5× dari τ_adapt 12× lebih panjang — penJARANGAN tidak proporsional
langsung karena θ juga ikut meluruh, tapi arahnya konsisten dengan
prediksi Zenke.)

Makin panjang τ_adapt, makin kuat penJARANGAN — persis prediksi paper.
Implikasi Phase 2: default homeostasis harus **detik-ke-menit**, bukan
detik. Sudah dicatat di docstring `AdaptiveLIF`.

## Figures

Regenerate dengan `rak viz --steps 350`. Semua input memakai statistik
MNIST sintetis yang sama dengan `rak bench`, supaya angka density di
figure konsisten dengan benchmark.

| File | Isi | Angka kunci |
|---|---|---|
| `01_lif.png` | V(t) + spike train LIF | 25 spike, ISI **tetap** |
| `02_adaptive_lif.png` | V(t) + spike train Adaptive LIF | 16 spike, ISI **makin panjang** |
| `03_adaptation_detail.png` | **4 panel: V, θ, v_eff, spike** | θ 0.000 → 0.560 |
| `04_comparison.png` | Overlay LIF vs Adaptive | 25 vs 16 spike |
| `05_izhikevich.png` | V(t) + spike train Izhikevich | 32 spike, ISI 63.8 ms |
| `06_layer_raster_dim.png` | Raster 784→400, input mean 0.13 | **density 5.15%** |
| `06_layer_raster_sedang.png` | Raster, input mean 0.30 | **density 14.61%** |
| `06_layer_raster_terang.png` | Raster, input mean 1.00 | **density 46.35%** |

### Tiga raster itu adalah argumen untuk `calib_value`

Rasio density **9×** antara input redup dan input terang:

```
mean 0.13  →   5.15%   ← realistis untuk MNIST
mean 0.30  →  14.61%
mean 1.00  →  46.35%   ← sparsity HANCUR
```

Jadi kalibrasi ke full-scale itu bukan hanya keliru secara analitik
(§3), tapi **menghancurkan sparsity secara empirik**. Di full-scale
hampir setengah dari setiap sel neuron aktif di setiap timestep —
MCU hanya bisa memb_itkan klaim "dikit neuron aktif" kalau encoder
memakai statistik input yang realistis, bukan nilai maksimal.

Figure 03 adalah bukti visual paling jelas dari adaptasi — lihat §4.

Semua panel spike memakai sumbu **biner** (0/1, dengan label
"no spike"/"spike"). Versi pertama mewarisi skala tegangan dari panel V
sehingga menampilkan angka -60/-40 di panel spike — membingungkan dan
secara semantik salah.

---

## Temuan dari review Phase 1 (yang sudah diperbaiki)

Enam poin dari review, dan tiga di antaranya ternyata bug nyata, bukan
sekadar permintaan dokumentasi:

| # | Reviewer bilang | Status |
|---|---|---|
| P1 | Density benchmark 0.00, klaim sparse invalid | ✅ **Bug nyata.** README salah (angka basi) + kalibrasi 20× keliru. Input sekarang `mnist-like`, density 3.11% |
| P2 | V naik 1.0→1.6 tidak dijelaskan, plot V dan θ dipisah | ✅ **Bug nyata.** Yang naik adalah threshold efektif, bukan V. Plot 4 panel baru (`03_adaptation_detail.png`) |
| P3 | Edge case: input negatif, dt>tau, neuron mati, overflow Izhikevich | ✅ **Bug nyata.** `dt > tau` membuat α < 0 → membran berosilasi, neuron fire tiap timestep. Sekarang di-raise |
| P4 | Seed bobot init tidak disebut | ✅ Sudah di-set (via `torch.Generator` eksplisit), sekarang didokumentasikan + 3 test |
| P5 | `tau_detector` → `tau_adapt` lebih jelas | ✅ Di-rename seluruh repo, alasannya ditulis di docstring |
| P6 | Sumbu spike Izhikevich displaying tegangan | ✅ **Bug nyata.** Panel spike sekarang biner 0/1 |

Tambahan yang ditemukan sendiri saat investigasi:

- **Semantik `v_leak` salah.** Nol input menghasilkan v → 0, bukan v → `v_leak`. Sekarang `v_ss = v_leak + I` dan nol input → `v_leak`. Ada test yang mengunci ini.
- **Guard stabilitas Izhikevich terikat ke `v_threshold`.** `v_threshold = 1e30` membuat batas dt jadi 0.00 dan config sah ikut ditolak. Sekarang guard memakai batas tervalidasi empiris (`dt_max_validated_ms = 1.0`) yang lepas dari `v_threshold`.

Test count: **83 → 101**.

## Yang TIDAK dikerjakan di fase ini

Sengaja ditunda ke Phase 2+: Hebbian/BCM/Oja (T2.1), STDP (T2.2),
sparse connectivity (T3.1), homeostasis enforcement (T1B). Trace sudah
dibangun sekarang supaya tidak perlu membongkar `SpikeLayer` nanti.