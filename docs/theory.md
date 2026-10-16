# Catatan Teori RAK

Dokumen ini menjelaskan **kenapa** setiap keputusan teknis diambil.
Bukan tutorial cara pakai — untuk itu lihat `docs/tutorials/`.

---

## 1. Kenapa bentuk implisit untuk LIF

Rumus LIF di literatur:

```
V[t] = V[t-1] + (dt/tau) · (V_rest - V[t-1] + R·I[t])
```

Ini bentuk **Euler eksplisit**. Coefficient `(dt/tau)` harus `< 1`
supaya stabil. Kalau `dt > tau`:

- coefficient `> 1` → gain > 1 → membran bisa **meldak**
- lebih buruk, kalau `V[t]` mendarat tepat di `v_threshold`,
  floating point bisa membuat `V[t] >= v_th` **selalu** benar →
  neuron **freeze** dan berhenti menembakkan spike selamanya

Bug ini nyata dan mahal dicari. Norse dan snnTorch sama-sama memakai
bentuk implisit karena alasan yang sama.

Bentuk yang kita pakai:

```
V[t] = (1 - dt/tau) · V[t-1] + (dt/tau) · (V_rest + R·I[t])
```

`alpha = 1 - dt/tau` selalu `< 1` untuk `tau > 0`, jadi stabil untuk
**semua** timestep. Stability menang atas kesesuaian-notasi-paper.

Test yang menjaga ini: `test_numerical_stability_large_tau_small_dt`.

---

## 2. Kenapa spike trace dibangun di Phase 1, bukan Phase 2

Spike trace:

```
x[t] = γ · x[t-1] + z[t],   γ = exp(-dt/τ)
```

Adalah satu-satunya state yang dibutuhkan STDP trace-based:

```
Δw_ij = η_plus · x_pre_i[t] · z_post_j[t]
      - η_minus · x_post_j[t] · z_pre_i[t]
```

Kalau trace tidak ada di `SpikeLayer` sejak awal, Phase 2 harus
mengubah signature `forward()` dan mengulang semua test Phase 1.

Trace juga **kausal**: `x[t]` hanya bergantung spike sampai `t`. Itu
yang bikin STDP bisa dihitung online tanpa menyimpan histori penuh.

Test yang menjaga kausalitas: `test_trace_is_causal`.

---

## 3. Kenapa homeostasis bukan opsional

Pertanyaan paling sering: "Kenapa tidak pakai STDP aja?"

Jawaban: **STDP tanpa homeostasis diverge.** Bobot tumbuh tanpa
batas, semua neuron fire bersamaan, jaringan mati.

Empat komponen dari Diehl & Cook 2015 yang wajib ada:

| Komponen | Fungsi |
|---|---|
| Lateral inhibition | kompetisi antar neuron |
| Adaptive threshold | semua neuron dapat chances |
| Synaptic scaling | bobot tidak unbounded |
| Trace-based STDP | Coupling timing-based |

Aturan kuantitatif dari **Zenke, Hennequin & Gerstner 2013** (PLoS
Comput Biol) — ini jadi assertion di test, bukan tebakan:

```
τ_detector   : detik–menit      (lebih lambat → stabilitas hilang)
τ_scaling    : ~ τ_plasticity   (lebih besar → osilasi, lebih kecil → drift)
τ_plasticity : detik–menit      (bobot berubah)
homeostasis  : jam–hari         (reaksi lambat)
```

Kenapa homeostasis harus lambat: kalau reaksinya secepat plasticity,
homeostasis jadi **inert** — tidak sempat menstabilkan. Kecepatan
lambat itulah yang membuat homeostasis bisa menyeimbangkan drift
jangka panjang tanpa melawan fluktuasi pendek.

Konsekuensi ke Phase 1: `AdaptiveLIF.tau_adapt_ms` default-nya
5 detik (demo adaptasi). **Phase 2 wajib menaikkan ini** ke detik-
menit secara eksplisit. Ada catatan di docstring-nya supaya tidak
terlupa.

---

## 4. Kenapa trace pakai `exp`, bukan `1 - dt/tau`

Steady-state dari `x[t] = (1-dt/tau)x[t-1]` adalah `tau/dt`.
Steady-state dari `x[t] = exp(-dt/tau)x[t-1]` adalah
`1/(1 - exp(-dt/tau))`.

Untuk `dt << tau` keduanya hampir sama. Untuk `dt ~ tau` berbeda
signifikan, dan yang eksak:

- tidak pernah negatif walau `dt > tau`
- steady-state-nya **exact**, bukan pendekatan

Rumus STDP di paper hampir selalu ditulis dengan eksponensial, jadi
memakai bentuk eksak membuat plotting STDP window cocok dengan
publikasi.

---

## 5. Kenapa Izhikevich menolak `dt != 1`

Koefisien Izhikevich (`0.96`, `0.40`, `0.556`, `0.92`, `0.08`) **tidak
universal** — itu hasil tuning untuk `dt = 1 ms`, `a = 0.02`,
`b = 0.2`, `c = -65`, `d = 8`.

Untuk `dt` lain, koefisiennya harus diskalakan ulang — dan itu
tidak bisa dilakukan tanpa tabel dari paper asli. Daripada diam-diam
menghasilkan hasil yang salah secara biologis, kita **fail keras**
dengan pesan error yang jelas.

Kalau Phase 3 butuh `dt` yang lebih kecil untuk hemat compute, itu
pekerjaan riset tersendiri — bukan sekadar mengubah angka.

---

## 6. Kenapa SpikeLayer dense di Phase 1

Bobot dense `[n_in, n_out]` disengaja.

T1.4.5 meminta benchmark throughput. Kalau sparsity sudah aktif,
benchmark-nya mengukur sparse overhead — bukan baseline dense yang
jujur.

Sparse connectivity (FITUR 3.1, Phase 4) punya backend sendiri dengan
benchmark sendiri. Yang diukur di Phase 1 adalah: "seberapa cepat
dense timestep simulation yang divinektor-sPEGIH?" — itu angka
baseline yang semua sparsity claim dibandingkan.

---

## 7. Kapan pakai LIF vs Adaptive LIF vs Izhikevich

| Model | Pakai kalau | Don't pakai kalau |
|---|---|---|
| **LIF** | Default. Neuron Independent, cepat, trace-nya bersih untuk STDP | Butuh neurona yang tahu riwayat pembakaran dirinya sendiri |
| **Adaptive LIF** | Butuh homeostasis **per-neuron** (T1B.1). Neuron yang terlalu aktif harus dihukum | Butuh neurona yang bisa di-trigger ulang dalam window |
| **Izhikevich** | Perlu reproduksi mode firing biologis spesifik (bursting, chattering). Riset neuroscience | Butuh model yang bisa di-train. Izhikevich punya mode reset yang **tidak bisa di-backprop** (threshold + reset bukan differentiable) |

Rule praktis: mulai dari LIF. Naik ke Adaptive LIF kalau ada
masalah homeostasis. Naik ke Izhikevich kalau Question-nya soal
biologi, bukan soal performa.

---

## 8. Referensi silang

- ADR-0001 — Omniglot split (Lake original)
- ADR-0002 — Metrik energi (SynOps, bukan joule)
- ADR-0003 — Spec MNIST (replikasi Diehl & Cook)

## 9. Paper untuk dibaca sebelum Phase berikutnya

| Phase | Baca dulu |
|---|---|
| 2 | Zenke et al. 2013 (PlOS Comput Biol) |
| 3 | Diehl & Cook 2015 (Front Comput Neurosci) |
| 4 | Lukosevicius & Jaeger 2019 (Comp Sci Rev) |
| 5 | Krotov & Hopfield 2016 (NeurIPS) |
| 6 | Snell et al. 2017 (NeurIPS) |
| 7 | Li & Talwalkar 2020 (JMLR 21(55)) |
| 9 | Millidge 2023 (PC position paper) |