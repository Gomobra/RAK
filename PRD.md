# PRD RAK v1.2 — Arsitektur AI Murni

Nama Proyek: RAK (Rekam, Akses, Komposisi)
Versi: 1.2 — Pure Architecture (revisi dari v1.1)
Status: Riset Fundamental
Target: 2 tahun, bertahap
Perubahan dari v1.1: 7 koreksi lanjutan berbasis data dari implementasi Phase 1 dan sumber acuan Phase 3.

---

## 0. Changelog v1.0 → v1.1

Semua koreksi berbasis hasil riset terhadap paper dan benchmark nyata.

| ID | Lokasi v1.0 | Masalah | Koreksi v1.1 |
|---|---|---|---|
| E1 | NFR-04 | "Energy < 10x backprop" tidak bermakna dan tidak bisa diukur di CPU | Ganti metrik: SynOps vs ANN FLOPs |
| E2 | FITUR 1 | Homeostasis tidak ada sebagai requirement | FITUR 1B baru, prioritas P0 |
| E3 | FITUR 5.2 | Lateral inhibition baru bulan 13 | Tarik ke Phase 2 (bulan 5) |
| E4 | FITUR 6.2 | Target Omniglot 70% tanpa declare split | Lake split, target 55–60% |
| E5 | FITUR 6.1 | MNIST spec kosong | Spec Diehl-Cook 2015 lengkap |
| E6 | §5.2 | "Loss tidak perlu" salah untuk benchmark | Task signal wajib ada |
| E7 | NFR-06 vs T7.1.2 | "Autograd untuk test bukan training" bertentangan dengan baseline surrogate gradient | Surrogate gradient = baseline only |
| E8 | FITUR 3.1 | Sparse connectivity tanpa acuan | Pakai small-world, bukan random |
| E9 | §6 | Params dan State tercampur | Pisahkan ke dataclass berbeda |
| E10 | Roadmap | Izhikevich sebelum homeostasis | Urutan dibalik |
| E11 | FITUR 4.4 | Immutable Fact Store disamakan dengan Hopfield | Pisahkan: Hopfield vs SDM |

Tambahan baru:
- **FITUR 1B**: Homeostatic Plasticity [P0] — syarat mutlak STDP converge
- **FITUR 3B**: Reservoir Computing [P1] — satu-satunya "no backprop" yang terbukti bekerja
- **§7 Aturan Validasi** — disiplin evaluasi dari NAS Best Practices (JMLR 2020)

## 0. Changelog v1.1 → v1.2

Tujuh koreksi ini berdasar pada: (a) implementasi nyata Phase 1, (b) sumber
kode asli Diehl & Cook 2015, (c) hasil pengujian numerik.

| ID | Lokasi v1.1 | Masalah | Koreksi v1.2 |
|---|---|---|---|
| C1 | Sub-Fitur 1B, §9 | τ_θ ditulis "detik–menit" | **~2.8 jam**. Diehl & Cook pakai `tc_theta = 1e7 ms`. Zenke menyebut detik–menit sebagai *batas bawah*, bukan target realistis |
| C2 | Metrik §9 | MNIST ≥ 90% | **85–89%**. Repo resmi achieve 89% from-scratch dengan 400 neuron; 95% butuh 6400 neuron + tuning conductance |
| C3 | Roadmap | Phase 2 tidak dipecah | **2A (learning rules) → 2B (homeostasis)**. Homeostasis tidak bisa dibuktikan tanpa plastisitas yang harus distabilkan |
| C4 | Sub-Fitur 1B.2 | Hanya synaptic scaling soft | Tambah **hard column normalization** (Diehl-Cook) sebagai jaring pengaman |
| C5 | — | Refractory period tidak ada | **Requirement eksplisit**. Diehl-Cook pakai 5 ms. Tanpa itu neuron bisa fire beruntun dan STDP kehilangan makna |
| C6 | — | Acuan STDP tidak dipin | **ADR-0004**: 2-trace Norse di 2A, 3-trace Diehl-Cook di Phase 3 |
| C7 | Sub-Fitur 1B.2 | Normalisasi "per-row" ambigu | Pin definisi: `W[k,j]`, **kolom = neuron postsynaptic**. Hard target Σ = 78 untuk 784 input |

### 0.0 Koreksi v1.1 → v1.2 (ringkas)

Detector homeostasis gw turunkan dari "detik–menit" ke "detik ke ~jam".
Bacaan gw di v1.1 terlalu optimistis: Zenke et al. 2013 memang menyebut
"detik–menit" sebagai kondisi *harus dipenuhi* supaya stabilitas tidak
hilang, tapi angka yang dipakai Diehl-Cook untuk MNIST adalah
`1e7 ms ≈ 2.8 jam`. Menit terlalu agresif.


### 0.1 Catatan Environment (penting)

v1.0 mengasumsikan lingkungan dev = GitHub Codespaces. Kenyataannya tidak selalu begitu.

- Di Codespaces: `pip install torch` ✅
- Di Termux/Android (aarch64): TIDAK ada wheel PyPI untuk torch. Harus via `apt install python-torch`.

Konsekuensi: **kode harus portabel**. Tidak boleh bergantung pada ekstensi torch khusus (custom C++/CUDA). Hanya pakai operasi torch standar yang tersedia di kedua backend. Test knobs via env var.

---

## 1. Overview

### 1.1 Latar Belakang

AI modern dibangun di atas tiga pilar yang sudah berumur 40+ tahun:

- Backpropagation (Rumelhart, 1986)
- Transformer (Vaswani, 2017)
- Gradient descent (Cauchy, 1847)

Pilar-pilar ini terbukti bekerja, tapi punya keterbatasan fundamental:

- Butuh data raksasa
- Tidak bisa belajar online (terus-menerus)
- Tidak punya memori jangka panjang yang stabil
- Tidak bisa menjelaskan keputusan
- Boros energi (GPU besar)

Otak manusia tidak pakai ketiga pilar itu. Otak pakai:

- Spike (event-driven, bukan continuous)
- Local learning (STDP, Hebbian, bukan backprop)
- Sparsity (sedikit neuron aktif, bukan semua)
- Memory-compute co-location (bukan von Neumann)

Proyek RAK v1.1 bertujuan membangun arsitektur AI dari nol yang meniru prinsip-prinsip otak, bukan membungkus AI yang sudah ada.

### 1.2 Prinsip Fundamental

| Prinsip | Arti |
|---|---|
| Spike-based Neuron | komunikasi lewat spike diskrit, bukan angka kontinu |
| Event-driven | Komputasi hanya saat ada spike, bukan setiap step |
| Local learning | Update bobot pakai aturan lokal, bukan backprop global |
| Sparse | Sedikit neuron aktif di satu waktu |
| Memory-compute co-located | Memori dan komputasi di tempat yang sama |
| Online | Belajar terus-menerus, bukan batch |
| Explainable | Setiap keputusan bisa dilacak |
| Ringan | Jalan di CPU, hemat energi |

### 1.3 Yang Dibangun dari Nol

| Komponen | Yang Ada Sekarang | Yang RAK Bangun |
|---|---|---|
| Neuron | nn.Linear + activation | Model neuron biologis (LIF, Adaptive LIF, Izhikevich) |
| Learning | Backpropagation | STDP, R-STDP, Hebbian, BCM, Oja, e-prop, predictive coding |
| Homeostasis | — (tidak ada padanannya) | Adaptive threshold, synaptic scaling, lateral inhibition |
| Arsitektur | Transformer, CNN, SSM | Sparse modular graph + reservoir computing |
| Memori | Vector DB eksternal | Attractor network, Sparse Distributed Memory, episodic buffer |
| Inference | Forward pass penuh | Time-stepped sparse inference, sparse activation |
| Training | Batch, offline | Online, streaming, continual |

### 1.4 Non-Goals

- ❌ Saingi GPT-4 di skala umum
- ❌ Bikin AGI dalam 2 tahun
- ❌ Deployment produksi
- ❌ Pakai API LLM sebagai "otak"
- ❌ Pakai backpropagation sebagai learning utama
- ❌ Klaim hemat energi di hardware non-neuromorf

### 1.5 Klaim yang Mau Dibuktikan

**Klaim utama (tidak diubah):** Arsitektur spike-based dengan local learning bisa mencapai performa kompetitif dengan backprop pada task tertentu, dengan lebih sedikit parameter.

**Klaim kedua (baru, lebih kuat & realistis):** Arsitektur dengan local learning + homeostasis bisa **online learning tanpa catastrophic forgetting**, sesuatu yang batch backprop Indicators sulit.

**Klaim ketiga (baru):** Modern Hopfield memory secara matematik identik dengan attention mechanism. Kita bisa mengimplementasikan attention via attractor memory, bukan softmax dot-product.

Kalau berhasil, kontribusi nyata. Kalau gagal, tetap ada pelajaran.

### 1.6 Realitas Roadmap (koreksi penting)

Riset menunjukkan batas keras dari klaim "no backprop":

- **Local learning scale test (arxiv 1807.04587)**: "Our results are largely negative. None of the tested algorithms are capable of effectively scaling up to training large networks on ImageNet." Tapi MNIST/CIFAR kecil tetap jalan.
- **Predictive coding ≈ backprop (Millidge 2023)**: "the assumptions we have to make to get an approximation to backprop basically massively reduce the biological plausibility of the scheme such that it is not really any better than backprop directly."

Konsekuensi: **RAK tidak akan menang di skala besar.** Klaim menang di (a) sample efficiency, (b) online learning, (c) explainability, (d) parameter efficiency. Scope eksperimen disesuaikan.

---

## 2. Requirements

### 2.1 Functional Requirements

| ID | Requirement | Prioritas | Phase |
|---|---|---|---|
| FR-01 | Model neuron sendiri (bukan nn.Linear) | P0 | 1 |
| FR-02 | Learning rule sendiri (bukan backprop) | P0 | 2 |
| FR-03 | Arsitektur jaringan sendiri | P0 | 4 |
| FR-04 | Memori jangka panjang yang stabil | P0 | 5 |
| FR-05 | Inference sparse, event-driven diukur | P1 | 7 |
| FR-06 | Online learning (streaming) | P1 | 4, 6 |
| FR-07 | Explainability (trace spike) | P1 | 7 |
| FR-08 | Port ke 2+ task | P1 | 3, 6 |
| FR-09 | Benchmark vs backprop | P1 | 7 |
| FR-10 | Energy measurement (SynOps, bukan joules) | P2 | 8 |
| FR-11 | Homeostatic plasticity | P0 | 2 |
| FR-12 | Reservoir computing | P1 | 4 |

### 2.2 Non-Functional Requirements

| ID | Requirement | Target |
|---|---|---|
| NFR-01 | Jalan di CPU tanpa GPU | 4 GB RAM |
| NFR-02 | Portabel lintas backend | torch PyPI (Codespaces) DAN torch apt (Termux/Android) |
| NFR-03 | Parameter count | < 10M |
| NFR-04-Rev | **SynOps per inference** | < 0.5× ANN FLOPs pada akurasi setara |
| NFR-05 | Reproducible | Seed di-set, direkam, mean±std over seeds 1..10 |
| NFR-06 | Pure Python + PyTorch low-level | Tidak ada API eksternal, tidak ada pre-trained model |
| NFR-06a | Backprop | **Hanya untuk baseline benchmark.** Bukan learning rule utama RAK |

**Catatan NFR-04-Rev:** SynOps (sparsity-aware synaptic operations) adalah jumlah operasi synaptic yang benar-benar terjadi. Ini proxy yang jujur, bukan spike count (yang bukan joules).

Energi absolut (joule) diukur dengan Intel RAPL (`perf stat -e power/energy-pkg/`) dikurangi idle power. Karena RAPL hanya membaca counter dan atribusi per-process adalah **estimasi**, semua angka joule WAJIB dilabeli "estimated" di laporan. Spike count TIDAK boleh dipakai sebagai klaim energi.

Referensi: Loihi 2 menghasilkan 10–35× efisiensi energi dibanding Jetson Nano, TAPI itu karena hardware neuromorfik. Di CPU digital, spike-based inference tidak otomatis lebih hemat. Benchmark SpikingJelly menunjukkan SNN di T=8 hampir 2× lebih mahal dari ANN; hanya T=1–2 yang menghasilkan 24–52% operasi.

### 2.3 Constraints

- Budget: $0
- Hardware: Codespaces (utama) + Termux/Android (portabel, opsional)
- Tim: 1 orang
- Waktu: 2 tahun, bertahap
- Background: Pemula, belajar sambil jalan
- Fokus: Riset fundamental, bukan produk

---

## 3. Core Features

### FITUR 1: Neuron Model (dari Nol) — Phase 1

**Idea:** Model neuron biologis yang mensimulasikan dinamika membran.

#### Sub-Fitur 1.1: Leaky Integrate-and-Fire (LIF)

Rumus (formulasi implisit timestep — stabil, tanpa ODE solver):

```
V[t] = (1 - α) · V[t-1] + β · I[t]      dengan α = 1 - dt/τ_m
z[t] = 1 jika V[t] ≥ v_threshold, else 0
V[t] = (1 - z[t]) · V[t] + z[t] · v_reset
```

Memakai bentuk implisit (bukan `V += dt/τ·(V_rest - V + R·I)`) karena pembulatan floating point dari bentuk eksplisit bisa membuat neuron "macet" membeku forever — bug klasik SNN.

Tasks:
- [x] T1.1.1: Implementasi LIF di PyTorch, tensor-based, batch + time dims
- [x] T1.1.2: Test: input konstan → firing rate stabil (t-invariant)
- [x] T1.1.3: Test: input bervariasi → firing pattern berubah
- [x] T1.1.4: Visualisasi: plot V(t) dan spike train
- [x] T1.1.5: Parameter tuning: τ_m, v_threshold, v_reset (hasil dicatat di docs/theory.md)

#### Sub-Fitur 1.2: Adaptive LIF

Rumus:

```
θ += β_θ · z       (setiap spike, threshold naik)
θ -= (θ - θ_0) · α_θ  (decay tiap timestep)
v_threshold_eff = v_threshold + θ
```

Tasks:
- [x] T1.2.1: Tambah variabel adaptasi θ ke LIF
- [x] T1.2.2: Implementasi: setiap spike, θ += β_θ, lalu θ decay
- [x] T1.2.3: Test: firing rate menurun saat input konstan
- [x] T1.2.4: Bandingkan dengan LIF biasa, plot overlay

#### Sub-Fitur 1.3: Izhikevich Model

Rumus:

```
V[t+1] = 0.96·V[t] + 0.40·w[t] - V[t]³ + 0.556·w[t]   (pada τ = 10 step)
w[t+1] = 0.92·w[t] - 0.08·V[t]
if V[t+1] ≥ 30: V[t+1] = c, w[t+1] = d
```

Tasks:
- [x] T1.3.1: Implementasi persamaan Izhikevich
- [x] T1.3.2: Test: reproduksi minimal 6 mode firing (regular, bursting, chattering, adapting, spike-frequency adaptation, mixed)
- [x] T1.3.3: Dokumentasi: kapan pakai LIF vs Izhikevich (docs/theory.md)

#### Sub-Fitur 1.4: Neuron Layer

Tasks:
- [x] T1.4.1: Implementasi SpikeLayer (banyak neuron)
- [x] T1.4.2: State: V, θ, spike_history, **trace_pre/post**
- [x] T1.4.3: Forward: simulasi T timesteps, return spike tensor [T, batch, N]
- [x] T1.4.4: Reset state antar batch (`.reset_state()`)
- [x] T1.4.5: Benchmark: 100 neuron × 100 timestep, ukur spike/detik dan neuron-update/detik

**Kenapa trace ada sejak Phase 1:** spike trace (`x[t] = γ·x[t-1] + z[t]`) adalah fondasi STDP. Kalau tidak dibangun sekarang, Phase 2 akan perlu membongkar SpikeLayer.

#### Sub-Fitur 1.5: Spike Trace & Coding

Rumus trace:

```
x[t] = γ · x[t-1] + z[t]        γ = exp(-dt/τ_tr)
```

Tasks:
- [x] T1.5.1: Implementasi trace decay + increment
- [x] T1.5.2: Trace ada di state neuron, di-reset bareng
- [x] T1.5.3: Test: trace monoton turun setelah spike berhenti
- [x] T1.5.4: Poisson spike train generator (untuk input MNIST)

---

### FITUR 1B: Homeostatic Plasticity [BARU — P0] — Phase 2

**Idea (paling kritis di seluruh PRD):** Tanpa homeostasis, STDP diverge. Bobot tumbuh tanpa batas, semua neuron fire bersamaan, jaringan mati.

Referensi wajib: Zenke, Hennequin & Gerstner, "Synaptic Plasticity in Neural Networks Needs Homeostasis", PLoS Comput Biol 2013.

**Aturan desain dari Zenke 2013 (jadi assertion, bukan tebakan):**
- τ_θ (rate detector homeostasis) harus **detik atau lebih panjang**. Zenke 2013: kalau terlalu lambat, stabilitas hilang. Diehl & Cook memakai 1e7 ms ≈ **2.8 jam** — lihat ADR-0004.
- τ_scaling harus **sebanding** dengan τ_plasticity. Terlalu besar → osilasi. Terlalu kecil → drift.
- Bobot berubah dalam detik–menit; homeostasis bereaksi dalam jam–hari. Ini yang bikin homeostasis "inert" terhadap perubahan cepat.

#### Sub-Fitur 1B.1: Intrinsic Plasticity (Adaptive Threshold)

```
θ += β_θ · z
θ -= (θ - θ_0) · α_θ      dengan 1/α_θ ≫ 1/α_plasticity (detector lambat)
```

Tasks:
- [ ] T1B.1.1: Reuse θ dari Adaptive LIF (Sub-Fitur 1.2)
- [ ] T1B.1.2: Parameter τ_θ dipisah dari τ_plasticity, default ~1 jam (lihat C1)
- [ ] T1B.1.3: TEST: setelah 100 epoch input konstan, firing rate semua neuron ≈ sama
- [ ] T1B.1.4: TEST: stabilitas — Θ homeostatic > 1 (jinko dengan Zenke 2013)

#### Sub-Fitur 1B.2: Synaptic Scaling (HARD + SOFT)

**Definisi shape yang dipin (C7):** `W[k, j]` dengan k = input neuron,
j = neuron postsynaptic. **Kolom** (`sum over k`) = total drive masuk ke
satu neuron postsynaptic. Jadi "per-row" di versi v1.1 ambigu — yang
dimaksud adalah per-neuron, yaitu per-kolom di layout ini.

**HARD — column normalization ke konstanta (Diehl & Cook 2015):**
```
col_sums = W.sum(dim=0)                       # (n_out,)
W *= target_sum / (col_sums + ε)              # target_sum = 78 untuk 784 input
```
Dipanggil **periodik** (setiap N contoh presentasi), bukan tiap timestep.
`target_sum = 78.0` persis seperti di `Diehl&Cook_spiking_MNIST.py`
(`weight['ee_input'] = 78.`).

**SOFT — Oja-style kontinu:**
```
W /= ( mean_k(|W_kj|) + ε )^p                 # per-neuron, tiap timestep
```
Lebih halus dan per-synapse, tapi butuh tuning `p` dan belum tentu konverge
dengan sendirinya.

Keduanya dipakai bersamaan: hard jadi jaring pengaman yang menjamin
drive masuk per neuron tetap, soft jadi mekanisme learning yang halus.
Ini pembagiannya juga yang dipakai di biologi — scaling Traub & Millar
mencakup keduanya.

Tasks:
- [ ] T1B.2.1: Implementasi hard column normalization ke `target_sum`
- [ ] T1B.2.2: Implementasi soft Oja-style per-neuron
- [ ] T1B.2.3: τ_scaling ~ τ_plasticity
- [ ] T1B.2.4: TEST: Σ_k W_kj == target_sum ± 1e-6 setelah normalisasi
- [ ] T1B.2.5: TEST: tidak ada osilasi pada τ_scaling besar
- [ ] T1B.2.6: TEST: tidak ada drift pada τ_scaling kecil

#### Sub-Fitur 1B.2b: Refractory Period (C5)

**Kenapa wajib:** tanpa refractory, neuron bisa fire di beberapa timestep
berurutan. Spike jadi tidak punya "waktu" yang jelas, dan STDP kehilangan
makna — kita tidak bisa bilang spike mana yang mendahului yang mana.
Diehl & Cook memakai `refrac_e = 5 ms` untuk neuron eksitasi.

Saat neuron dalam periode refraktori: `v` di-clamp ke `v_reset` dan spike
diblokir.

Tasks:
- [ ] T1B.2b.1: Timer per-neuron (bukan scalar bersama)
- [ ] T1B.2b.2: `v` di-clamp + spike diblokir selama periode
- [ ] T1B.2b.3: TEST: ISI ≥ refrac_ms
- [ ] T1B.2b.4: TEST: refractory menurunkan firing rate
- [ ] T1B.2b.5: TEST: tidak ada NaN setelah spike pertama (0*inf guard)

#### Sub-Fitur 1B.3: Lateral Inhibition

```
I_inh = W_inh @ z_exc        W_inh = -ρ · I_ones  (ρ > 0, no self-connection)
```

Tasks:
- [ ] T1B.3.1: Implementasi inhibition global-to-global (WTA)
- [ ] T1B.3.2: ρ (inhibition strength) di-tune per network size
- [ ] T1B.3.3: TEST: inhibisi memunculkan kompetisi antar neuron eksitasi

---

### FITUR 2: Learning Rule (dari Nol) — Phase 2

**Idea:** Aturan belajar lokal.

#### Sub-Fitur 2.1: Hebbian Learning

```
Δw_ij = η · x_i · x_j      (x = spike trace, BUKAN spike biner)
```

Tasks:
- [ ] T2.1.1: Implementasi Hebbian update berbasis trace
- [ ] T2.1.2: Test: asosiasi 2 pola → bobot menguat
- [ ] T2.1.3: Oja variant (normalisasi): Δw = η(a_i a_j - β·a_j²·w_ij) — lebih stabil daripada plain Hebbian
- [ ] T2.1.4: Test: BCM variant Δw = η·a_j·(a_j - θ_j)·a_i
- [ ] T2.1.5: Visualisasi: bobot berubah seiring waktu

Referensi implementasi: Nengo punya BCM (`Δω = κ·a_j·(a_j − θ_j)·a_i`) dan Oja (`Δω = κ·(a_i·a_j − β·a_j²·ω_ij)`). Copy persis, sudah battle-tested.

#### Sub-Fitur 2.2: STDP (Spike-Timing-Dependent Plasticity)

**Trace-based, bukan pair-based.** Pair-based butuh spike list eksplisit dan
lambat. Trace-based cukup dua tensor decay.

**Acuan: 2-trace gaya Norse di Phase 2A.** Backend 3-trace ala Diehl & Cook
ditambahkan di Phase 3. Lihat **ADR-0004** untuk konvensi tanda dan urutan
internal timestep yang wajib dipatuhi.

Rumus additive:

```
Δw_ij = η_plus · x_pre_i[t] · z_post_j[t]
      - η_minus · x_post_j[t] · z_pre_i[t]
```

Rumus multiplicative power (dipakai Diehl-Cook, performanya lebih baik):

```
A_plus(w)  = η_plus · (w_max - w)^μ
A_minus(w) = η_minus · (w - w_min)^μ
```

Tasks:
- [ ] T2.2.1: Implementasi STDP berbasis trace
- [ ] T2.2.2: 4 varian: additive, additive-step, multiplicative-power, multiplicative-ReLU
- [ ] T2.2.3: Test: urutan spike (pre sebelum post → naik; sebaliknya → turun)
- [ ] T2.2.4: Plot STDP window
- [ ] T2.2.5: Hard bound w_min/w_max
- [ ] T2.2.6: Trace **per-neuron** (n_pre,) dan (n_post,) — bukan per-synapse
- [ ] T2.2.7: Urutan internal: decay trace → update W → bump trace
- [ ] T2.2.8: TEST: window implementasi cocok rumus eksponensial (toleransi 2e-4)
- [ ] T2.2.9: TEST: pre & post di timestep sama tanpa histori → Δw = 0

Referensi: Norse `STDPState` (t_pre, t_post + decay) dan `STDPParameters` (a_pre, a_post, tau_pre_inv, tau_post_inv, w_min, w_max, eta_plus, eta_minus, stdp_algorithm, mu, hardbound). Default Norse: tau = 50ms, w ∈ [0, 1], eta = 1e-3.

#### Sub-Fitur 2.3: Reward-Modulated STDP (R-STDP) — Phase 6

```
e[t] = γ_e · e[t-1] + z_pre ⊗ z_post        (eligibility trace)
Δw[t] = reward[t] · e[t]
```

Tasks:
- [ ] T2.3.1: Tambah variabel eligibility trace
- [ ] T2.3.2: Update: Δw = reward · eligibility
- [ ] T2.3.3: Test: task sederhana (pilih kiri/kanan)
- [ ] T2.3.4: Bandingkan dengan Q-learning standar (bukan RL library)

#### Sub-Fitur 2.4: e-prop — **[BARU, P1]**

**Kenapa ditambah:** e-prop (Lillicrap et al.) adalah three-factor rule yang spiking-compatible dan **kompetitif dengan BPTT** di TIMIT dengan LSNN. Ini "STDP yang sudah di-upgrade". RAK selama ini stuck di R-STDP murni yang jauh lebih lemah.

Rumus:

```
L[t]   = ŷ - y*                                    (error output, per-neuron)
z̄[t]   = ∂z/∂v · (L[t] · W_out)                   (feedback, fixed random OK)
eligibility e_ij[t] = λ · e_ij[t-1] + z̄_i[t] · s_j[t]
Δw_ij  = η · L[t] · e_ij[t]
```

Tasks:
- [ ] T2.4.1: Implementasi feedback alignment (W_out fixed random, bukan transpose)
- [ ] T2.4.2: Implementasi eligibility trace dengan λ decay
- [ ] T2.4.3: Test: LSNN (spiking RNN) di time-series regression
- [ ] T2.4.4: Bandingkan: e-prop vs R-STDP vs BPTT (baseline) di task yang sama

#### Sub-Fitur 2.5: Predictive Coding — Stretch, reposition

**Reposisi (hasil riset Millidge 2023):** PC bukan untuk "beat backprop". PC kuat di **low-data, online learning, continual learning, conditioning fleksibel**. Itu persis klaim RAK. Jadi PC tetap masuk, tapi posisinya bukan "menang akurasi", melainkan "mekanisme online learning".

Arsitektur (Whittington & Bogacz 2017, Rao & Ballard 1999): 3 layer, tiap layer punya expectation unit + error unit, lateral connections, weight update via Hebbian lokal.

Tasks:
- [ ] T2.5.1: Implementasi PC layer (expectation unit + error unit)
- [ ] T2.5.2: Inference loop (relaxasi sampai energi konvergen)
- [ ] T2.5.3: Test: prediksi sequence sederhana
- [ ] T2.5.4: Dokumentasi: PC untuk apa, dan kenapa TIDAK untuk beat BP

---

### FITUR 3: Arsitektur (dari Nol) — Phase 4

#### Sub-Fitur 3.1: Sparse Connectivity

**Acuan biologis (bukan random — koreksi E8):**

Jaringan biologis itu **small-world**: high clustering + short path length (Bassett & Bullmore 2016, Network Neuroscience). Random sparse bukan representasi biologis yang benar.

Tiga mode sparsity yang akan dibandingkan:
- `random`: connectivity acak pada degree tetap
- `small_world`: Watts-Strogatz, high clustering + short path
- `local`: koneksi lokal seperti receptive field (Chavlis & Poirazi 2025)

Referensi parameter: dendritic ANN (Nature Communications, Jan 2025) menunjukkan sparse + structured connectivity + restricted receptive-field sampling menghasilkan **1–3 orde of magnitude lebih sedikit parameter** dengan accuracy setara atau lebih baik.

Tasks:
- [ ] T3.1.1: Bikin adjacency generator (3 mode di atas)
- [ ] T3.1.2: Ukur: % koneksi aktif, clustering coefficient, average shortest path
- [ ] T3.1.3: Bandingkan dense vs sparse (parameter, speed, accuracy)
- [ ] T3.1.4: Visualisasi graph

#### Sub-Fitur 3.2: Modular Architecture

Tasks:
- [ ] T3.2.1: Definisikan modul (sensor, asosiasi, memori, motor)
- [ ] T3.2.2: Interface antar modul (spike train in/out, bukan dense tensor hidden state)
- [ ] T3.2.3: Test: 3 modul, komunikasi lewat spike
- [ ] T3.2.4: Visualisasi: informasi mengalir antar modul

#### Sub-Fitur 3.3: Reservoir Computing **[BARU — P1]**

**Kenapa wajib ada (koreksi besar terhadap v1.0):**

Reservoir computing / Echo State Network adalah **satu-satunya arsitektur "no backprop" yang terbukti bekerja di dunia nyata**:
- Random recurrent reservoir **fixed** (tidak dilatih)
- Output linear dilatih via **least squares** (bukan backprop, bukan reward)
- Danish Vowel: **0% error** (previous best 1.8%)
- Chaotic time series: **3 orde of magnitude** lebih akurat dari RNN trained
- Spoken digits: 0.2% WER

Ini memenuhi 4 prinsip RAK sekaligus: sparse ✅, recurrent ✅, local learning ✅, online ✅.

Kelemahan yang harus didokumentasikan (bukan disembunyikan):
- Perlu tuning **spectral radius ρ(W)**. Klasik: ρ(W) < 1. Tapi paper Lukosevicius & Jaeger (2019) koreksi misconception umum: ρ(W) < 1 **bukan** syarat perlu-cukup untuk echo state property.
- Memory capacity terbatas (Lin & Kanerva)
- Fading memory — jejak stimuli lama meluruh

Rumus:

```
x[t+1] = (1-a)·x[t] + a·tanh(W_res·x[t] + W_in·u[t])
y[t]   = W_out · x[t]                    (W_out dilatih, x[t] tidak)
```

Tasks:
- [ ] T3.3.1: Reservoir LIF spiking (bukan tanh — ini versi spike)
- [ ] T3.3.2: Spektral radius: cara ukur (`scipy.sparse.linalg.eigs`), cara tune
- [ ] T3.3.3: Readout dilatih via **RLS** (Recursive Least Squares) → online, bukan batch
- [ ] T3.3.4: Test: MNIST via reservoir, bandingkan dengan STDP network
- [ ] T3.3.5: Dokumentasi batas: memory capacity, fading memory, ρ(W)

Referensi RLS (Nengo):
```
g = P · r
P ← P - (g·gᵀ)/(1 + rᵀ·P·r)
Δω = κ·δ/n · e·(P·r)
```

#### Sub-Fitur 3.4: Recurrent Loops

Tasks:
- [ ] T3.4.1: Implementasi recurrent connection
- [ ] T3.4.2: Cegah explosion: (a) bobot inhibitory, (b) trace-based inhibition, (c) λ untuk plasticity, (d) **spectral radius < 1**
- [ ] T3.4.3: Test di task sequence
- [ ] T3.4.4: Bandingkan dengan RNN/LSTM

#### Sub-Fitur 3.5: Sparse Attention

**Justifikasi kuat:** Modern Hopfield network **secara matematik identik dengan attention mechanism** (Transformer's softmax dot-product attention). Jadi mengimplementasikan attention via attractor memory adalah claim yang legitimate dan publishable — bukan gimmick.

Tasks:
- [ ] T3.5.1: Implementasi top-k sparse attention
- [ ] T3.5.2: Bandingkan: dense vs sparse attention (SynOps, accuracy)
- [ ] T3.5.3: Implementasi attention sebagai modern Hopfield retrieval
- [ ] T3.5.4: Test: apakah spike-based attention mengsocket konvergensi lebih cepat?

---

### FITUR 4: Memory Module (dari Nol) — Phase 5

#### Sub-Fitur 4.1: Attractor Network (Hopfield)

Energi:
```
E = -½ Σ w_ij · s_i · s_j
```

**Kapasitas (jawaban pasti untuk T4.1.4):**
- Hopfield klasik: **K_max ≈ 0.138 N** (Krotov & Hopfield, "Dense Associative Memory for Pattern Recognition", NeurIPS 2016)
- Modern Hopfield: **P = exp(αN)** — eksponensial (Dalla Chiara et al., PRL 132:077301, 2024)

Tasks:
- [ ] T4.1.1: Implementasi Hopfield network (bi-/tri-stable)
- [ ] T4.1.2: Simpan 10 pola
- [ ] T4.1.3: Test: recall pola dari noise
- [ ] T4.1.4: **Ukur capacity** — sweep N, cari titik gagal, bandingkan dengan prediksi 0.138N
- [ ] T4.1.5: Visualisasi energy landscape
- [ ] T4.1.6: Implementasi modern Hopfield (polynomial energy, n≠2), ukur capacity eksponensial

#### Sub-Fitur 4.2: Sparse Distributed Memory (SDM) — **[PISAH DARI HOPFIELD, koreksi E11]**

**Ini yang jadi "Immutable Fact Store"** (Sub-Fitur 4.4), BUKAN Hopfield.

Referensi: Kanerva 1988 "Sparse Distributed Memory"; Furber et al. (Manchester) implementasi N-of-M codes; sudah pernah diimplementasikan dengan **spiking neurons di Nengo** (arxiv 2109.03111).

Arsitektur:
```
ADDRESS (N-of-M code, sparse)
    ↓ address decoder (fixed random sparse weights)
ADDRESS SPACE (W >> A)
    ↓
DATA MEMORY (content-addressable matrix)
```

Tasks:
- [ ] T4.2.1: Implementasi N-of-M sparse coding
- [ ] T4.2.2: Address decoder dengan fixed random sparse weights
- [ ] T4.2.3: Data memory: content-addressable read/write
- [ ] T4.2.4: **Ukur capacity**: berapa pasangan (address, data) bisa disimpan?
- [ ] T4.2.5: Test: read dengan noise di address
- [ ] T4.2.6: Test: dua fakta_addr yang mirip → apakah tabrakan?

#### Sub-Fitur 4.3: Episodic Buffer

Tasks:
- [ ] T4.3.1: Buffer spike pattern (ring buffer, bukan vector DB)
- [ ] T4.3.2: Write: setiap kejadian masuk
- [ ] T4.3.3: Read: query by similarity (dot product atas spike train)
- [ ] T4.3.4: Capacity: berapa bisa disimpan?
- [ ] T4.3.5: Consolidation: pindah ke jangka panjang

#### Sub-Fitur 4.4: Immutable Fact Store **[REPOSISI, koreksi E11]**

**Basis: SDM (Sub-Fitur 4.2), bukan Hopfield.** Karena Hopfield attractor bisa overlap (pola A dan B berdekatan bisa menyatu jadi satu attractor) — persis masalah yang tidak boleh terjadi di fact store.

Tasks:
- [ ] T4.4.1: Representasi fakta sebagai N-of-M spike pattern (dari SDM)
- [ ] T4.4.2: Simpan di memori non-overlapping (alamat ter-disjoint)
- [ ] T4.4.3: Query by content, bukan by address
- [ ] T4.4.4: **Test: simpan 100 fakta, recall dengan akurasi > 95%**
- [ ] T4.4.5: Test: 100 fakta yang sengaja dipilih mirip-plicate → ukur interference

#### Sub-Fitur 4.5: Consolidation (Stretch)

Tasks:
- [ ] T4.5.1: Trigger consolidation (setiap N step, atau saat idle)
- [ ] T4.5.2: Replay: main ulang pengalaman
- [ ] T4.5.3: Update bobot jangka panjang pakai STDP
- [ ] T4.5.4: Test: apakah konsolidasi meningkatkan recall?

---

### FITUR 5: Inference Engine — Phase 7

**Reposisi penting:** Event-driven di CPU **biasanya lebih lambat** dari dense timestep simulation. Event-driven adalah abstraction yang benar untuk hardware neuromorf, **bukan** speedup CPU.

Evidence:
- SparseProp (arxiv 2312.17216) mencapai O(N)→O(log N) dengan **binary heap priority queue** — belum ada di PyTorch
- PyTorch sparse ops di CPU (`index_add_`, COO scatter) tidak lebih cepat dari dense `mm` pada sparsity moderate
- Python event-queue loop akan kalah dari vectorized timestep

Jadi T5.1.3 tetap dilakukan, tapi **harapan hasilnya dibalik**: ekspektasinya dense menang, dan itu temuan yang valid untuk dipublikasikan.

#### Sub-Fitur 5.1: Event-Driven Simulation

Tasks:
- [ ] T5.1.1: Event queue (heap)
- [ ] T5.1.2: Process: tiap event → update neuron target
- [ ] T5.1.3: Benchmark: event-driven vs timestep. **Ekspektasi: timestep menang di CPU. Catat honestly.**
- [ ] T5.1.4: Ukur % neuron idle

#### Sub-Fitur 5.2: Sparse Activation

Tasks:
- [ ] T5.2.1: Winner-take-all (k=1)
- [ ] T5.2.2: k-WTA (top-k)
- [ ] T5.2.3: Test: akurasi vs sparsity trade-off

#### Sub-Fitur 5.3: Temporal Coding

**Data ekspektatif (dari benchmark encoding MNIST, model surrogate-gradient kecil):**

| Encoding | Spikes/sample | Akurasi |
|---|---|---|
| repeat_binary | 2.429 | 91.9% |
| direct | 2.396 | 91.2% |
| rate (Bernoulli) | 2.396 | 90.4% |
| **latency / TTFS** | **142** | **88.1%** |
| burst | 1.055 | 84.7% |
| rank-order | 142 | 67.1% |
| phase | 3.011 | 62.1% |

Dengan local learning (bukan surrogate gradient), expect **lebih rendah**. Rank-order butuh recurrent network; di feedforward dia poor.

Tasks:
- [ ] T5.3.1: Rate encoding (Poisson/Bernoulli)
- [ ] T5.3.2: TTFS / latency coding
- [ ] T5.3.3: Rank-order coding
- [ ] T5.3.4: Benchmark semua encoding di task yang sama, plot accuracy vs spike count

---

### FITUR 6: Learning Tasks (Testbed)

#### Sub-Fitur 6.1: MNIST (Baseline) — Phase 3

**Target: replikasi Diehl & Cook 2015.**

Spec wajib (koreksi E5):
```
Arsitektur : 6400 excitatory neuron, 784 input neuron
Neuron     : LIF
Input      : Poisson spike train, 350 ms, rate 0–63.75 Hz
             (max pixel intensity 255 dibagi 4)
             repeat sampai excitatory neuron fire ≥ 5 spike per presentasi
Synapse    : CONDUCTANCE-based (bukan current-based) — wajib
Learning   : STDP unsupervised (TIDAK ADA teaching signal)
                + adaptive threshold (intrinsic plasticity)
                + lateral inhibition
Training   : 15× repeat of full MNIST train set
Target     : ≥ 85% (stretch: 89%, replikasi Diehl & Cook)
```

Baseline conversion (rate-based ANN → SNN) Diehl et al. 2015: **99.1%**.

Tasks:
- [ ] T6.1.1: Download MNIST via torchvision
- [ ] T6.1.2: Konversi MNIST jadi Poisson spike train
- [ ] T6.1.3: Bangun LIF layer 784→6400
- [ ] T6.1.4: **Conductance-based synapse** (T retenu, bukan langsung ke V)
- [ ] T6.1.5: Trace-based STDP
- [ ] T6.1.6: Intrinsic plasticity (adaptive threshold)
- [ ] T6.1.7: Lateral inhibition
- [ ] T6.1.8: Loop: present sample → cek neuron count → repeat
- [ ] T6.1.9: Inference: spike count output neuron = prediksi
- [ ] T6.1.10: Evaluasi 10.000 test digit, ≥ 85%
- [ ] T6.1.11: Ablasi: matikan satu per satu {conductance, inhibition, adaptive threshold} → tabulas dampaknya
- [ ] T6.1.12: Bandingkan dengan MLP backprop (baseline)

#### Sub-Fitur 6.2: Omniglot (Few-Shot) — Phase 6

**Split WAJIB dideklarasikan (koreksi E4).**

Menggunakan **Lake et al. original split**. Pada split ini:
- Baseline nearest neighbor: 28.86%
- Matching Networks: 43.40%
- Meta-Learner LSTM: 43.44%
- MAML: 48.70%
- **Prototypical Networks: 49.42%**

**Target RAK: 55–60%** (bukan 70%).

Referensi tambahan: paper hippocampal meta-learner berbasis R-STDP (arxiv 2306.04410) melaporkan hasil sangat tinggi, TAPI pada split yang longgar dan dengan protokol berbeda. **Angka dari paper itu TIDAK bisa dibandingkan** dengan split Lake. Catat eksplisit di laporan.

Tasks:
- [ ] T6.2.1: Download Omniglot (via GitHub raw / torchvision punya)
- [ ] T6.2.2: Setup protokol **Lake split** eksplisit + episode sampler (N-way K-shot)
- [ ] T6.2.3: Encode gambar → spike train
- [ ] T6.2.4: Latih dengan R-STDP
- [ ] T6.2.5: Reimplementasi Prototypical Networks sebagai baseline (bukan cuma SITASING angka paper)
- [ ] T6.2.6: Bandingkan di **protokol identik**

#### Sub-Fitur 6.3: Associative Memory — Phase 5

Tasks:
- [ ] T6.3.1: Bikin dataset: pasangan (A, B)
- [ ] T6.3.2: Simpan di SDM
- [ ] T6.3.3: Test: recall dengan noise
- [ ] T6.3.4: Ukur capacity vs interference

#### Sub-Fitur 6.4: Sequence Memory — Stretch

Tasks:
- [ ] T6.4.1: Bikin dataset: urutan angka, recall
- [ ] T6.4.2: Latih dengan recurrent + STDP
- [ ] T6.4.3: Bandingkan dengan LSTM

#### Sub-Fitur 6.5: Language Mini — Stretch, last

Tasks:
- [ ] T6.5.1: Dataset: kalimat pendek, klasifikasi
- [ ] T6.5.2: Encoding: kata → spike pattern (SDR)
- [ ] T6.5.3: Latih dengan STDP
- [ ] T6.5.4: Bandingkan dengan LSTM kecil

---

### FITUR 7: Evaluation & Analysis — Phase 7

#### Sub-Fitur 7.1: Aturan Validasi (WAJIB, baru di v1.1)

Referensi: Li & Talwalkar, "Best Practices for Scientific Research on Neural Architecture Search", JMLR 21(55), 2020.

Aturan yang WAJIB dipatuhi, bukan opsional:

1. **Eval protocol sama untuk semua method.** Tanpa ini tabel benchmark cuma noise.
2. **Baseline = random search**, bukan cuma MLP/CNN/LSTM. DNAS (Sciuto et al. 2019) dan Li & Talwalkar (2019) menunjukkan random search sudah sangat kompetitif.
3. **Multiple seeds, report mean ± std.** Seed 1..10, direkam di config. STDP variance-nya sangat tinggi.
4. **Ablation tiap komponen.** T7.4 ada di v1.0 tapi harus jalan **sepanjang project**, bukan hanya di Phase 7. Ablation yang cuma jalan di akhir = ablation yang tidak dilakukan.
5. **Pilih arsitektur di validation split, JANGAN test split.** Kesalahan paling umum dan tidak bisa dipulihkan.
6. **Simpan spike log + checkpoint** setiap run (NFR-05).

#### Sub-Fitur 7.2: Benchmark

Tasks:
- [ ] T7.2.1: Baseline: MLP, CNN, LSTM (backprop)
- [ ] T7.2.2: Baseline: SNN + surrogate gradient (Hanya sebagai pembanding, BUKAN learning rule RAK)
- [ ] T7.2.3: Jalankan semua di task + protocol identik
- [ ] T7.2.4: Tabel perbandingan dengan mean ± std

#### Sub-Fitur 7.3: Energy Measurement — Phase 8

Tasks:
- [ ] T7.3.1: **SynOps** counter — sparsity-aware synaptic operations
- [ ] T7.3.2: Bandingkan SynOps vs ANN FLOPs pada akurasi setara
- [ ] T7.3.3: RAPL joule: `perf stat -e power/energy-pkg/`, kurangi idle power
- [ ] T7.3.4: **WAJIB** label angka joule sebagai "estimated" (atribusi per-process bukan pengukuran langsung)
- [ ] T7.3.5: Visualisasi: energy vs akurasi
- [ ] T7.3.6: Eksplisit catat: spike count BUKAN joules

#### Sub-Fitur 7.4: Explainability

Tasks:
- [ ] T7.4.1: Log spike train per inferensi
- [ ] T7.4.2: Visualisasi: neuron mana yang fire?
- [ ] T7.4.3: Analisis: pola apa yang dikenali?
- [ ] T7.4.4: Trace keputusan end-to-end (input spike → neuron mana → output spike)

#### Sub-Fitur 7.5: Ablation Study

Tasks:
- [ ] T7.5.1: Matikan homeostasis → apa yang terjadi?
- [ ] T7.5.2: Matikan sparsity → apa yang terjadi?
- [ ] T7.5.3: Matikan recurrent → apa yang terjadi?
- [ ] T7.5.4: Matikan conductance-based synapse → apa yang terjadi?
- [ ] T7.5.5: Tabel ablation dengan mean ± std

---

### FITUR 8: Documentation & Publication — Phase 8

#### Sub-Fitur 8.1: Kode & Repo

Tasks:
- [ ] T8.1.1: Setup repo + LICENSE (MIT)
- [ ] T8.1.2: Dokumentasi setiap komponen
- [ ] T8.1.3: Tutorial: cara reproduce hasil
- [ ] T8.1.4: Pyproject.toml agar bisa `pip install -e .`

#### Sub-Fitur 8.2: Blog Series

Tasks:
- [ ] T8.2.1: Post 1: kenapa arsitektur murni (termasuk **koreksi roadmap v1.0**)
- [ ] T8.2.2: Post 2: neuron model
- [ ] T8.2.3: Post 3: learning rule — kenapa homeostasis wajib
- [ ] T8.2.4: Post 4: hasil benchmark
- [ ] T8.2.5: Post 5: pelajaran yang didapat

#### Sub-Fitur 8.3: Paper (Stretch)

Tasks:
- [ ] T8.3.1: Pilih venue
- [ ] T8.3.2: Tulis draft
- [ ] T8.3.3: Submit

---

## 4. User Flow

### 4.1 Flow: Eksperimen Baru

```
Ide → Tulis hipotesis → Desain eksperimen → Kode
    ↓
Test di task kecil (MNIST subset, < 1 menit)
    ↓
Kalau gagal → debug → iterasi
Kalau berhasil → scale up
    ↓
Ablation + analisis → dokumentasi → publikasi
```

### 4.2 Flow: Training

```
Data → Encode jadi spike train
    ↓
Inisialisasi (sparse small-world, homeostasis aktif)
    ↓
Loop:
    ├─ Forward: timestep simulation (dense, bukan event — lihat FITUR 5.1)
    ├─ Homeostasis: adaptive threshold + synaptic scaling
    ├─ Learning: STDP / R-STDP / e-prop (LOKAL)
    ├─ Consolidation (setiap N step)
    └─ Log: spike count, SynOps, wall time
    ↓
Evaluasi di test set (seed 1..10)
    ↓
Simpan checkpoint + spike log + config
```

### 4.3 Flow: Analisis

```
Load checkpoint + spike log + config
    ↓
Visualisasi: neuron aktif, koneksi, energy
    ↓
Ablation: matikan komponen, ukur efek (mean ± std)
    ↓
Bandingkan dengan baseline (protocol identik)
    ↓
Tulis insight
```

---

## 5. Architecture

### 5.1 High-Level

```
┌──────────────────────────────────────────────┐
│              INPUT ENCODER                   │
│  (gambar → spike train, rate/TTFS)           │
└──────────────────┬───────────────────────────┘
                   │
┌──────────────────▼───────────────────────────┐
│         SPIKING NEURAL NETWORK               │
│                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ Sensor   │→ │ Asosiasi │→ │ Motor    │  │
│  │ Module   │  │ Module   │  │ Module   │  │
│  └──────────┘  └────┬─────┘  └──────────┘  │
│                     │                       │
│                ┌────▼─────┐                 │
│                │ Memory   │                 │
│                │ Module   │                 │
│                │(SDM/Hop) │                 │
│                └────┬─────┘                 │
│                     │                       │
│                ┌────▼─────┐                 │
│                │Reservoir │                 │
│                │ Module   │                 │
│                └──────────┘                 │
│                                              │
│  Learning: STDP / R-STDP / e-prop (LOKAL)    │
│  Homeostasis: adaptive θ + scaling + inhib   │
│  Activation: sparse, small-world             │
└──────────────────┬───────────────────────────┘
                   │
┌──────────────────▼───────────────────────────┐
│              OUTPUT DECODER                  │
│  (spike train → klasifikasi/aksi)            │
└──────────────────────────────────────────────┘
```

### 5.2 Komponen Detail

| Komponen | Fungsi | Dibangun dari Nol? |
|---|---|---|
| Neuron | LIF / Adaptive LIF / Izhikevich | ✅ Ya |
| Learning | STDP, R-STDP, Hebbian, BCM, Oja, e-prop | ✅ Ya |
| Homeostasis | Adaptive threshold, synaptic scaling, lateral inhibition | ✅ Ya |
| Arsitektur | Sparse small-world modular + reservoir | ✅ Ya |
| Memory | Attractor (Hopfield) + SDM + episodic buffer | ✅ Ya |
| Inference | Timestep simulation (event-driven sebagai ablation) | ✅ Ya |
| Encoder | Rate / TTFS / rank-order | ✅ Ya |
| Decoder | Spike count → output | ✅ Ya |
| Optimizer | — | ❌ Tidak perlu (local learning) |
| Loss | **Task signal / reward** | ⚠️ ** Koreksi E6 |

**Koreksi E6 — "Loss tidak perlu" adalah KESALAHAN.** Untuk benchmark task, loss/task signal wajib ada:
- Omniglot few-shot butuh class label → reward signal untuk R-STDP
- Baseline MLP/CNN/LSTM butuh loss untuk training
- MNIST unsupervised STDP memang tidak butuh label, tapi butuh **assessment signal** (cek apakah neuron cukup fire) untuk mekanisme repeat-presentasi

Yang benar: RAK tidak butuh **global differentiable loss**. Yang tetap ada: **local task signal**.

### 5.3 Prinsip Desain

- No backprop sebagai learning utama
- No global differentiable loss — task signal lokal
- No dense layer — sparse small-world connectivity
- No GPU untuk inference — CPU cukup
- No API — semua di lokal
- **Homeostasis bukan opsional** — tanpa itu STDP diverge
- **Portable** — tidak bergantung pada backend spesifik

---

## 6. State Representation

**Koreksi E9:** Params dan State dipisahkan. Parameter = yang di-set sebelum training dan tetap. State = yang berubah setiap timestep.

### 6.1 Params (dikoncfigurasi, tetap)

```python
@dataclass(frozen=True)
class LIFParams:
    tau_mem: float        # time constant membran (ms)
    tau_syn: float        # time constant synapse (ms)
    tau_trace: float      # time constant spike trace (ms)
    v_leak: float         # resting potential / leak (mV)
    v_th: float           # threshold (mV)
    v_reset: float        # reset potential (mV)
```

```python
@dataclass(frozen=True)
class STDPParams:
    eta_plus: float       # learning rate potentiation
    eta_minus: float      # learning rate depression
    tau_pre: float        # trace pre (ms)
    tau_post: float       # trace post (ms)
    w_min: float          # bobot minimum
    w_max: float          # bobot maksimum
    mu: float             # exponent multiplicative STDP
    algorithm: str        # additive | additive_step | multiplicative_pow | multiplicative_relu
```

```python
@dataclass(frozen=True)
class HomeostasisParams:
    tau_theta: float   # rate detector homeostasis — detik ke JAM (Zenke 2013 + Diehl-Cook 1e7 ms)
    theta_inc: float      # threshold increment per spike
    tau_scaling: float    # synaptic scaling (~ tau_plasticity)
    scaling_power: float  # exponent synaptic scaling
    inhibition_strength: float
```

### 6.2 State (berubah tiap timestep)

```python
@dataclass
class LIFState:
    v: torch.Tensor        # membran potential [N]
    i: torch.Tensor        # synaptic current [N]  (conductance-based)
    theta: torch.Tensor    # adaptive threshold [N]
    trace_pre: torch.Tensor
    trace_post: torch.Tensor
    eligibility: torch.Tensor | None   # hanya untuk R-STDP
    last_spike_t: torch.Tensor | None
```

```python
@dataclass
class SynapseState:
    w: torch.Tensor        # bobot [N_pre, N_post]
    g: torch.Tensor | None # conductance (conductance-based synapses)
    delay: torch.Tensor    # ← PARAMETER, bukan state. Pindah ke SynapseParams
```

```python
@dataclass
class MemoryState:
    hopfield_weights: torch.Tensor | None
    sdm_address_decoder: torch.Tensor | None
    sdm_data_memory: torch.Tensor | None
    episodic_buffer: list            # list of SpikePattern
    consolidated: list
    reservoir_state: torch.Tensor | None
    reservoir_readout: torch.Tensor | None
    reservoir_rls_P: torch.Tensor | None
```

### 6.3 Persistence

```python
torch.save({
    "version": "1.1",
    "params": ...,        # semua Params
    "state": ...,         # semua State
    "config": ...,        # seed, dataset split, protocol
}, "checkpoint.pt")
```

Wajib include `config` berisi seed dan split — supaya NFR-05 (reproducible) benar-benar bisa dipertanggungjawabkan.

---

## 7. Tech Stack

### 7.1 Development

| Layer | Tools | Alasan |
|---|---|---|
| Bahasa | Python 3.10+ | Standar |
| Numerik | NumPy | Dasar |
| Tensor | PyTorch (low-level) | Autograd **hanya** untuk baseline benchmark |
| Sparse | scipy.sparse | Koneksi sparse, spectral radius |
| Visualisasi | matplotlib, networkx | Plot spike, graph |
| Test | pytest | Standar |
| Logging | Python logging | Simple |

**Yang TIDAK dipakai:**
- ❌ HuggingFace Transformers
- ❌ API LLM
- ❌ Pre-trained model
- ❌ nn.Transformer, nn.LSTM, nn.Conv2d
- ❌ snnTorch, Norse, Brian2, Nengo, Lava (untuk implementasi — **boleh** dipakai sebagai referensi baca)
- ❌ Custom CUDA/C++ extension (portabilitas)

### 7.2 Backend Portability (koreksi baru)

| Backend | Install | Catatan |
|---|---|---|
| Codespaces (x86_64) | `pip install torch` | Standar |
| Termux (aarch64 Android) | `apt install python-torch` | Tidak ada wheel PyPI |
| Kaggle T4 | `pip install torch` | Untuk baseline backprop saja |

Konsekuensi kode:
- Hanya `torch.nn.Module`, operasi tensor standar, `torch.sparse`
- Tidak pakai `torch.compile`, custom op, atau ekstensi native
- Dependensi neural network **tidak boleh** hard-fail kalau backend berubah

### 7.3 Referensi & Library untuk Belajar

| Library | Untuk | Link |
|---|---|---|
| Norse | Implementasi LIF + STDP referensi | github.com/norse/norse |
| snnTorch | API neuron module + surrogate gradient baseline | github.com/jeshraghian/snntorch |
| Brian2 | Arsitektur simulator, codegen | briansimulator.org |
| Nengo | **Learning rules lengkap + RLS** (referensi terbaik) | nengo.ai |
| Lava | Arsitektur hardware-aware | lava-nc.github.io |
| SpikingJelly | Benchmarking SynOps | spikingjelly.readthedocs.io |

**Nengo adalah referensi terbaik untuk Phase 2** — punya 5 learning rule lengkap dengan update equation:
```
PES : Δω_ij = (κ/n)·e_j·a_i                    (butuh error signal)
RLS : recursive least squares                   (butuh error signal)
BCM : Δω_ij = κ·a_j·(a_j - θ_j)·a_i              (NO error signal)
Oja : Δω_ij = κ·(a_i·a_j - β·a_j²·ω_ij)         (NO error signal)
Voja: encoder update                            (NO error signal)
```

Library ini untuk **belajar**, bukan dipakai. Kita bikin sendiri supaya paham.

### 7.4 Struktur Folder

```
rak/
├── pyproject.toml
├── pytest.ini
├── src/
│   ├── neuron/
│   │   ├── trace.py            # spike trace + Poisson generator
│   │   ├── lif.py              # T1.1
│   │   ├── adaptive_lif.py     # T1.2
│   │   ├── izhikevich.py       # T1.3
│   │   └── layer.py            # T1.4
│   ├── homeostasis/
│   │   ├── intrinsic.py        # T1B.1
│   │   ├── scaling.py          # T1B.2
│   │   └── inhibition.py       # T1B.3
│   ├── learning/
│   │   ├── hebbian.py          # T2.1 (Hebbian, BCM, Oja)
│   │   ├── stdp.py             # T2.2
│   │   ├── r_stdp.py           # T2.3
│   │   ├── eprop.py            # T2.4
│   │   └── predictive_coding.py# T2.5 (stretch)
│   ├── architecture/
│   │   ├── sparse.py           # T3.1 (random/small_world/local)
│   │   ├── modular.py          # T3.2
│   │   ├── reservoir.py        # T3.3
│   │   ├── recurrent.py        # T3.4
│   │   └── attention_sparse.py # T3.5
│   ├── memory/
│   │   ├── attractor.py        # T4.1
│   │   ├── sdm.py              # T4.2 (SDM, N-of-M)
│   │   ├── episodic.py         # T4.3
│   │   ├── immutable_store.py  # T4.4 (based on SDM)
│   │   └── consolidation.py    # T4.5 (stretch)
│   ├── inference/
│   │   ├── event_driven.py     # T5.1
│   │   ├── sparse_activation.py# T5.2
│   │   └── temporal_coding.py  # T5.3
│   ├── encoder/
│   │   └── poisson.py
│   ├── decoder/
│   │   └── spike_count.py
│   ├── tasks/
│   │   ├── mnist.py            # T6.1
│   │   ├── omniglot.py         # T6.2
│   │   ├── associative.py      # T6.3
│   │   ├── sequence.py         # T6.4
│   │   └── language_mini.py    # T6.5
│   └── eval/
│       ├── seeds.py            # T7.1 — seed management
│       ├── benchmark.py        # T7.2
│       ├── energy.py           # T7.3 — SynOps + RAPL
│       ├── explainability.py   # T7.4
│       └── ablation.py         # T7.5
├── experiments/
│   ├── exp_001_lif/
│   ├── exp_002_homeostasis/
│   ├── exp_003_mnist/
│   └── ...
├── notebooks/
│   ├── 01_neuron_basics.ipynb
│   ├── 02_learning_rules.ipynb
│   ├── 03_memory.ipynb
│   └── ...
├── docs/
│   ├── theory.md
│   ├── decisions/              # ADR — Architecture Decision Records
│   │   ├── 0001_omniglot_split.md
│   │   ├── 0002_energy_metric.md
│   │   └── 0003_mnist_spec.md
│   ├── tutorials/
│   └── papers/
├── tests/
│   ├── test_lif.py
│   ├── test_adaptive_lif.py
│   ├── test_izhikevich.py
│   ├── test_trace.py
│   ├── test_homeostasis.py
│   └── test_stdp.py
├── configs/
│   ├── mnist_stdp.yaml
│   └── mnist_baseline.yaml
└── README.md
```

---

## 8. Roadmap 2 Tahun

**Perubahan fundamental:** semua_phase punya output yang bisa diverifikasi. Homeostasis dipindah ke Phase 2 (bukan implicit). Reservoir computing masuk Phase 4.

### Phase 0 — Fondasi (Bulan 1) — ✅ SELESAI
| Fokus | Deliverable | Status |
|---|---|---|
| Repo, tooling, konfigurasi | `pyproject.toml`, `pytest.ini`, `.gitignore` | ✅ |
| Seed management | `src/rak/eval/seeds.py`, default seeds 1–10 | ✅ |
| Test harness | 83 test hijau, `rak selfcheck` | ✅ |
| Docs | `docs/decisions/0001-0003`, `docs/theory.md` | ✅ |
| CLI | `rak info / selfcheck / bench / viz` | ✅ |
| **Output** | `pytest` hijau, `python -m rak` jalan | ✅ |

### Phase 1 — Neuron Models (Bulan 2) — ✅ SELESAI
| Fokus | Deliverable | Status |
|---|---|---|
| Spike trace + Poisson generator | `trace.py`, 17 test | ✅ |
| LIF | T1.1.1–T1.1.5 | ✅ |
| Adaptive LIF | T1.2.1–T1.2.4 | ✅ |
| Izhikevich | T1.3.1–T1.3.3 | ✅ |
| SpikeLayer | T1.4.1–T1.4.5 | ✅ |
| Visualisasi | `rak viz` → 5 figure | ✅ |
| **Output** | 2 mode firing tervalidasi, benchmark ~10⁵ neuron-update/s di CPU | ✅ |

**Temuan Phase 1** (detail di `experiments/exp_001_neuron/README.md`):
1. Bentuk implisit LIF wajib — Euler eksplisit tidak stabil untuk `dt > tau`
2. Bentuk diskret Izhikevich yang beredar bukan hasil integrasi ODE kuadratik; titik istirahat yang benar adalah coupled `(v=-70, w=-14)`
3. `c = -65` adalah nilai **reset**, bukan resting
4. SpikeLayer **wajib** kalibrasi bobot — tanpa itu v_ss < threshold dan layer diam total
5. `tau_adapt` panjang → penJARANGAN lebih kuat (terverifikasi 5→1 spike), sesuai Zenke 2013

### Phase 2A — Learning Rules (Bulan 3–4) — ✅ SELESAI
| Fokus | Deliverable | Status |
|---|---|---|
| Refractory period | 5 ms, default opt-in | ✅ |
| Hebbian + BCM + Oja | T2.1.1–T2.1.5 | ✅ |
| Trace-based STDP (4 varian) | T2.2.1–T2.2.9 | ✅ |
| STDP window plot | analitik vs implementasi, toleransi 2e-4 | ✅ |
| **Output** | Aturan belajar lokal tervalidasi terhadap rumus analitik | ✅ |
| **Test** | 183 test (dari 101) | ✅ |

### Phase 2B — Homeostasis (Bulan 5–6) — ✅ SELESAI
| Fokus | Deliverable | Status |
|---|---|---|
| Intrinsic plasticity (τ_θ ~ jam, dual-mode) | T1B.1.1–T1B.1.4 | ✅ |
| Synaptic scaling **hard** (kolom → Σ=78) | T1B.2.1–T1B.2.4 | ✅ |
| Synaptic scaling **soft** (target-based) | T1B.2.5–T1B.2.6 | ✅ |
| Lateral inhibition (E→I 1:1, I→E all-to-all) | T1B.3.1–T1B.3.3 | ✅ |
| **Homeostatic gain > 1** | GATE — **LULUS** | ✅ |
| **STDP bisa belajar asosiasi** | GATE — **LULUS** | ✅ |
| **Output** | Drive 2047.6 (tanpa homeo, saturasi) vs 135.3 (dengan homeo) | ✅ |
| **Output 2** | Asosiasi A->B: 0/60 -> **60/60** trial, selektif (B->B tetap 0/60) | ✅ |
| **Test** | 281 test total (dari 183) | ✅ |

**Temuan Phase 2** (ADR-0005, `experiments/exp_002_learning/README.md`):
1. η asimetris itu WAJIB — dengan η+ = η− STDP punya fixed point dan homeostasis kelihatan tidak berguna (yang salah)
2. Tiga mekanisme berbeda skala: θ = pembatas DAYA · scaling = penyamarataan DRIVE · inhibition = KOMPETISI
3. Rumus tuning: `θ_ss = β_θ × rate × τ_θ` — yang penting menjaga produknya
4. **Residual trace lintas trial meloloskan causality** — `trace_pre` dari presentasi sebelumnya bisa memicu LTP 9x lebih besar dari LTD yang seharusnya menetralkannya. Jarak antar presentasi bukan parameter kosmetik
5. **Guard `mu`**: menyetel `mu` pada varian STDP yang tidak membacanya kini error, bukan no-op senyap — bug ini sempat membuat figure 03 menampilkan `additive` 3x sambil mengklaim multiplicative

> **Kenapa dipecah (C3):** homeostasis tidak bisa dibuktikan tanpa
> plastisitas yang harus distabilkan. Zenke 2013 justru analyzes
> *interaksi* keduanya. Kalau homeostasis dibangun lebih dulu, satu-satunya
> cara membuktikannya tetap dengan STDP — jadi tidak ada keuntungan urutan.

### Phase 3 — MNIST Baseline (Bulan 5–6)
| Fokus | Deliverable |
|---|---|
| Poisson input encoder | T6.1.1–T6.1.2 |
| 784→6400 LIF layer | T6.1.3 |
| Conductance-based synapse | T6.1.4 |
| Full STDP training loop | T6.1.5–T6.1.9 |
| Evaluasi + ablations | T6.1.10–T6.1.12 |
| **Output** | **≥ 85% MNIST** (stretch 89%, replikasi Diehl & Cook) |
| **Bukti** | Tabel ablations: matikan {conductance, inhibition, adaptive θ} |

### Phase 4 — Arsitektur + Reservoir (Bulan 7–8)
| Fokus | Deliverable |
|---|---|
| Sparse connectivity (3 mode) | T3.1.1–T3.1.4 |
| Modular architecture | T3.2.1–T3.2.4 |
| Reservoir computing + RLS | T3.3.1–T3.3.5 |
| Recurrent loops | T3.4.1–T3.4.4 |
| **Output** | Reservoir beating atau match STDP network di MNIST |

### Phase 5 — Memory (Bulan 9–10)
| Fokus | Deliverable |
|---|---|
| Hopfield + capacity study | T4.1.1–T4.1.6 |
| SDM (N-of-M) | T4.2.1–T4.2.6 |
| Episodic buffer | T4.3.1–T4.3.5 |
| Immutable fact store | T4.4.1–T4.4.5 |
| **Output** | **Hopfield capacity ≈ 0.138N terverifikasi**, **SDM recall 100 fakta > 95%** |

### Phase 6 — R-STDP + Omniglot (Bulan 11–12)
| Fokus | Deliverable |
|---|---|
| R-STDP + eligibility trace | T2.3.1–T2.3.4 |
| Omniglot Lake split setup | T6.2.1–T6.2.3 |
| Reimplementasi ProtoNet baseline | T6.2.5 |
| Perbandingan protocol identik | T6.2.6 |
| **Output** | **Omniglot 5-way 1-shot 55–60%** di Lake split |

### Phase 7 — Inference + Eval (Bulan 13–14)
| Fokus | Deliverable |
|---|---|
| Event-driven + sparse activation | T5.1, T5.2 |
| Temporal coding | T5.3.1–T5.3.4 |
| Benchmark lengkap | T7.2.1–T7.2.4 |
| Explainability | T7.4.1–T7.4.4 |
| Ablation penuh | T7.5.1–T7.5.5 |
| **Output** | Tabel benchmark mean ± std seeds 1..10 |

### Phase 8 — Energy + Publikasi (Bulan 15–18)
| Fokus | Deliverable |
|---|---|
| SynOps vs FLOPs | T7.3.1–T7.3.2 |
| RAPL joule (labeled estimated) | T7.3.3–T7.3.6 |
| e-prop | T2.4.1–T2.4.4 |
| Blog series | T8.2.1–T8.2.5 |
| Paper draft | T8.3.1–T8.3.2 |

### Phase 9 — Ekstensi (Bulan 19–24, opsional)
| Fokus | Deliverable |
|---|---|
| Predictive coding (online learning) | T2.5.1–T2.5.4 |
| Sequence memory | T6.4 |
| Sparse attention via Hopfield | T3.5 |
| Language mini | T6.5 |
| Consolidation | T4.5 |
| Submit paper | T8.3.3 |

---

## 9. Metrik Keberhasilan (revisi)

| Metrik | v1.0 | v1.1 | Alasan perubahan |
|---|---|---|---|
| MNIST akurasi | 95% | **≥ 85%** (stretch 89%) | Repo resmi achieve 89% from-scratch dgn 400 neuron; 95% butuh 6400 neuron + tuning conductance (C2) |
| Omniglot 5-way 1-shot | 70% | **55–60%** (Lake split) | ProtoNet di Lake split = 49.42%. 70% = +20 poin, tidak realistis |
| Parameter count | < 10M | **< 10M** (tidak berubah) | — |
| Energy | < 10x backprop | **SynOps < 0.5× ANN FLOPs** | Spike count ≠ joules; joule di CPU = estimasi |
| Online learning | Bisa belajar tanpa lupa | **Sama, + metrik forgetting rate** | Perlu metrik kuantitatif |
| Immutable memory | Recall > 95% | **Recall > 95% @ 100 fakta** | Ditambah metrik interference |
| Blog posts | 10 | **10** (tidak berubah) | — |
| Paper | 1 draft | **1 draft** (tidak berubah) | — |

### 9.1 Metrik Tambahan (baru)

| Metrik | Target | Cara ukur |
|---|---|---|
| **Seed variance** | mean ± std over seeds 1..10 | Reported di setiap tabel |
| **SynOps ratio** | < 0.5 | Counter di eval/energy.py |
| **Firing rate uniformity** | std/mean < 0.1 setelah convergence | Homeostasis working |
| **Hopfield capacity** | ≈ 0.138N | Sweep N |
| **SDM interference** | < 5% pada fakta mirip | Test T4.4.5 |

---

## 10. Risiko & Mitigasi (revisi)

| Risiko | Probabilitas | Mitigasi |
|---|---|---|
| Gak bisa MNIST | 30% | Replikasi Diehl-Cook literal dulu. Bandingkan komponen per-komponen kalau meleset |
| STDP gak converge | **40% → 25%** | Homeostasis sudah jadi P0 (E2). Assertion test stabilitas Zenke 2013 |
| Homeostasis belum cukup | 40% | Oja/BCM normalization sudah termasuk. Escalate ke triple STDP |
| **Energy gak bisa dibuktikan** | **30% → 50%** | Sudah acknowledges di NFR-04-Rev. SynOps jadi metrik, joule jadi bonus |
| **Event-driven lebih lambat** | **0% → 90%** | Sudah diekspektasikan. Hasilnya publishable sebagai temuan |
| **Omniglot split salah** | **0% → 40%** | Lake split di-pin di Phase 0 (ADR-0001) |
| **Portable code gagal di Termux** | 30% | NFR-02 jadi requirement. Test minimal di kedua backend tiap Phase |
| Memory gak stabil | 50% | SDM (bukan Hopfield) untuk fact store (E11) |
| Butuh 2 tahun lebih | 60% | Iterasi kecil. 9 phase, masing-masing punya output verifiable |
| Kehilangan motivasi | 50% | Komunitas, blog, milestone kecil. Blog post tiap phase |
| **Scope creep** | **60%** | 5 fitur dipotong ke Phase 9 (Izhikevich tetap di Phase 1 karena core; PC, attention, sequence, language, consolidation ditunda) |

---

## 11. Referensi

### Paper Wajib Dibaca (Phase 1–2)

| Paper | Untuk |
|---|---|
| Diehl & Cook 2015, Front Comput Neurosci | MNIST 95% STDP. **Spec lengkap T6.1** |
| Zenke, Hennequin & Gerstner 2013, PLoS Comput Biol | Aturan stabilitas homeostasis. **Assertion test T1B** |
| Querlioz et al. 2013, Front Neural Circuits | Adaptive threshold + STDP MNIST |
| Izhikevich 2003, IEEE Trans Neural Networks | Model Izhikevich |
| Snell, Swersky & Zemel 2017, NeurIPS | Prototypical Networks (baseline Omniglot) |
| Lillicrap et al. 2020, "e-prop" | Three-factor learning rule |
| Millidge 2023, "Thoughts on the future of Predictive Coding" | Reposisi PC. **WAJIB baca sebelum T2.5** |
| Li & Talwalkar 2020, JMLR 21(55) | Best practices validasi. **Aturan T7.1** |
| Krotov & Hopfield 2016, NeurIPS | Kapasitas dense associative memory |
| Dalla Chiara et al. 2024, PRL 132:077301 | Modern Hopfield exponential capacity |
| Lukosevicius & Jaeger 2019, Comp Sci Rev | Reservoir computing, spectral radius |
| Bassett & Bullmore 2016, Network Neuroscience | Small-world brain networks |
| Chavlis & Poirazi 2025, Nat Commun | Dendritic ANN, parameter efficiency |

### Implementasi Referensi (baca, jangan pakai)

| Repo | Untuk |
|---|---|
| norse/norse | LIF + trace-based STDP (4 varian algoritma) |
| nengo/nengo | **Learning rules lengkap**: PES, RLS, BCM, Oja, Voja |
| jeshraghian/snntorch | API neuron module, surrogate gradient baseline |
| brian-team/brian2 | Arsitektur simulator, scheduling |
| fangwei123456/spikingjelly | Benchmarking SynOps |

### Paper untuk Validasi Angka

| Angka | Sumber |
|---|---|
| MNIST 95% STDP unsupervised | Diehl & Cook 2015 (6400 neuron, 15 epoch) |
| Omniglot Lake 5-way 1-shot 49.42% | Snell et al. 2017 |
| Hopfield capacity 0.138N | Krotov & Hopfield 2016 |
| Modern Hopfield exp(αN) | Dalla Chiara et al. 2024 |
| Loihi 2 energy 10–35× Jetson | Energy benchmark Loihi 2 2025 |
| SNN T=8 ≈ 2× ANN cost | SpikingJelly MNIST benchmark 2025 |
| Encoding accuracy table | SC-NeuroCore encoding guide |
| Local learning scale negative | arxiv 1807.04587 |

---

## 12. Pertanyaan Terbuka

1. **Omniglot split**: Lake original sudah diputuskan (Phase 0, ADR-0001)
2. **Target MNIST**: 90% vs 95% — realistis bergantung phase mana yang bener
3. **Bahasa dokumentasi**: Indonesia atau Inggris? Blog?)
4. **Nama repo**: `rak` atau nama lain?
5. **Venue paper**: NeurIPS/ICLR workshop dulu, atau langsung ECS?
6. **Apakah ada hardware neuromorf yang bisa diakses** (Loihi 2 via Intel Neuromorphic Community)? Kalau iya, klaim energi berubah drastis.

---

*PRD v1.1 — selesai. Next: repo scaffold + Phase 1.*