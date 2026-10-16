# ADR-0003: Spec MNIST Phase 3 = replikasi Diehl & Cook 2015

- Status: Accepted
- Date: 2026-10-15
- Affects: Phase 3 (T6.1), target akurasi

## Konteks

PRD v1.0 menulis "Latih dengan STDP, ukur akurasi" tanpa spesifikasi.
Itu tidak cukup — implementasi yang berbeda jauh bisa dapat hasil
berbeda 10%.

Benchmark yang akan kita replikasi:

Diehl, P.U. & Cook, M. "Unsupervised learning of digit recognition using
spike-timing-dependent plasticity", *Frontiers in Computational
Neuroscience*, 2015.

Hasil mereka: **95.0% MNIST** dengan STDP **murni unsupervised**
(tanpa teaching signal, tanpa label).

## Spesifikasi

```
Arsitektur : 6400 excitatory neuron, 784 input neuron
Neuron     : LIF
Input      : Poisson spike train, 350 ms
             max pixel intensity 255 / 4 = 63.75 Hz
             repeat presentasi sampai excitatory neuron fire ≥ 5 spike
Synapse    : CONDUCTANCE-based (bukan current-based)
Learning   : STDP unsupervised (TIDAK ADA teaching signal)
                + adaptive threshold (intrinsic plasticity)
                + lateral inhibition
Training   : 15× repeat of full MNIST train set (60k)
Test       : 10.000 digit, 10× presentasi
Target     : ≥ 90% (stretch 95%)
```

## Kenapa tiap komponen wajib

Jaringan mereka punya **4 komponen**, dan PRD v1.0 hanya punya 1:

| Komponen | Status di v1.0 | Kenapa wajib |
|---|---|---|
| Lateral inhibition | ❌ baru bulan 13 | Kompetisi antar neuron → WTA |
| Adaptive threshold | ⚠️ implicit | Homeostasis → semua neuron dapat chances |
| Conductance-based synapse | ❌ | Sinapse realistis, training stabil |
| 15× repeat | ❌ | 1 epoch tidak cukup untuk konvergensi |

Scaling dari paper:

| Neuron | Epoch | Accuracy |
|---|---|---|
| 100 | 3 | 82.9% |
| 400 | 7 | 87.0% |
| 1600 | 7 | 91.9% |
| 6400 | 15 | 95.0% |

## Keputusan

1. **Replikasi literal dulu.** Ikuti spec di atas persis. Target
   minimum **90%**.
2. **Ablasi wajib** — matikan satu per satu {conductance, inhibition,
   adaptive threshold}, tabulas dampaknya. Ini yang bikin blog post dan
   paper kita bernilai: menunjukkan mana yang benar-benar penting.
3. **Target realistis 90%, bukan 95%.** 95% butuh 6400 neuron +
   15 epoch + tuning conductance yang benar. Itu bisa 4–6 bulan,
   bukan 2.

## Konsekuensi

- NFR-01 (4 GB RAM) diuji dengan 6400×784 = 5M synapse. Kalau OOM,
  turun ke 1600 neuron (91.9% verified) dan tulis honestly.
- Kompleksitas: 5M synapse × T timestep. Dengan T=350, itu 1.75G
  synaptic operation per sample. **Batch 1 saja.** Batch lebih besar
  butuh storage untuk spike history.
- Baseline ANN (backprop) **wajib** di Phase 3 juga — kalau tidak ada,
  angka 90% tidak punya makna karena tidak ada yang.databandingkan.

## Alternatif yang ditolak

- **Train ANN dulu lalu convert** — ditolak sebagai approach utama.
  Conversion achieves 99.1% tapi itu bukan STDP, itu CURik
  informasi dari backprop. Boleh sebagai baseline pembanding (T6.1.12),
  tidak boleh sebagai hasil RAK.
- **Surrogate gradient** — ditolak sebagai learning rule utama.
  Boleh sebagai baseline benchmark (T7.2.2).
- **Target 95% sebagai minimum** — ditolak, tidak realistis dengan
  1 orang + 2 tahun + RAK juga harus/learn yang lain (Phase 4–8).