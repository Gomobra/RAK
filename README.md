# RAK — Rekam, Akses, Komposisi

Pure spiking architecture dengan local learning. Dibangun dari nol:
tidak pakai pre-trained model, tidak pakai API LLM, tidak pakai
backpropagation sebagai learning rule utama.

Status: **Phase 3 — MNIST** (STDP tanpa label; readout linear terbaik 86.80% pada test 10k)

## Prinsip

| Prinsip | Arti |
|---|---|
| Spike-based | komunikasi lewat spike diskrit, bukan angka kontinu |
| Event-driven | komputasi hanya saat ada spike |
| Local learning | update bobot pakai aturan lokal (STDP), bukan backprop global |
| Sparse | sedikit neuron aktif di satu waktu |
| Memory co-located | memori dan komputasi di tempat yang sama |
| Online | belajar terus-menerus, bukan batch |
| Explainable | setiap keputusan bisa dilacak |
| Ringan | jalan di CPU, tanpa GPU |

## Install

```bash
# Codespaces (x86_64)
pip install -e ".[dev,tasks]"

# Termux / Android (aarch64) — tidak ada wheel PyPI untuk torch
apt install python-torch python-torchvision python-numpy python-scipy
pip install -e ".[dev]" --no-deps
```

## Jalan

```bash
rak info                        # versi + backend
rak selfcheck                  # smoke test neuron + layer
rak bench --n-in 784 --n-out 100 --steps 350

# Readout: ubah spike count jadi kelas (label hanya untuk readout)
rak readout --weights runs/a/weights.pt runs/b/weights.pt \
    --train-samples 5000 --test-samples 10000 --trials 5 \
    --save-readout runs/readout.pt

pytest                          # test suite
pytest -m "not slow"            # skip benchmark
```

## Struktur

```
src/rak/
  neuron/       LIF, Adaptive LIF, Izhikevich, SpikeLayer, spike trace
  homeostasis/  adaptive threshold, synaptic scaling, lateral inhibition  (Phase 2)
  learning/     Hebbian, STDP, R-STDP, e-prop, predictive coding          (Phase 2+)
  data/         MNIST loader, Poisson encoder, adaptive intensity
  network/      CompetitiveNetwork, STDP, conductance LIF (Phase 3)
  readout/      Ridge readout + ensemble fitur spike count (Phase 3)
  eval/         seeds, benchmark, energy, explainability, ablation       (Phase 7)
```

## Roadmap

| Phase | Isi | Output terverifikasi |
|---|---|---|
| 0 | Scaffold, seeds, test harness | `pytest` hijau |
| 1 | Neuron models | 6 mode firing Izhikevich tervalidasi |
| 2 | Homeostasis + STDP | STDP converge, tidak diverge |
| 3 | MNIST | ≥ 90% unsupervised STDP |
| 4 | Sparse + reservoir | Reservoir match STDP network |
| 5 | Memory | Hopfield ≈ 0.138N, SDM recall > 95% |
| 6 | R-STDP + Omniglot | 55–60% Lake split |
| 7 | Inference + eval | Tabel mean ± std seeds 1–10 |
| 8 | Energy + publikasi | SynOps < 0.5× ANN FLOPs |

Detail lengkap + rationale: [`PRD.md`](PRD.md)

## Readout (bukan bagian learning)

Jaringan dilatih tanpa label. Setelah training, spike count tiap
gambar dipetakan ke kelas oleh `rak.readout.RidgeReadout`:

```python
from rak.readout import RidgeReadout, ensemble_counts

x_train = ensemble_counts([net_a, net_b], train_images, trials=5)
x_test = ensemble_counts([net_a, net_b], test_images, trials=5)

readout = RidgeReadout().fit(x_train, train_labels)
pred = readout.predict(x_test)
```

- Alpha dipilih di split validasi data **TRAIN**; test tidak dipakai
  untuk memilih apa pun.
- `trials=5` menjumlahkan 5 presentasi Poisson per gambar. Encoder
  stokastik; ini mengurangi varians sampling, bukan mengubah bobot.
- Ensemble = beberapa checkpoint, fitur spike count digabung menjadi
  satu vektor. Bukan rata-rata prediksi.

Hasil lengkap dan perbandingan jujur vs paper: lihat
[`experiments/exp_003_mnist/results.md`](experiments/exp_003_mnist/results.md).

## Referensi kunci

- Diehl & Cook 2015 — MNIST 95% unsupervised STDP (spec Phase 3)
- Zenke, Hennequin & Gerstner 2013 — syarat stabilitas homeostasis (Phase 2)
- Snell, Swersky & Zemel 2017 — Omniglot Lake split baseline (Phase 6)
- Li & Talwalkar 2020, JMLR — disiplin validasi (Phase 7)
- Krotov & Hopfield 2016 — kapasitas associative memory (Phase 5)

## License

MIT