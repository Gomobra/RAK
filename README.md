# RAK — Recurrent Adaptive Knowledge

Pure spiking architecture dengan local learning. Dibangun dari nol:
tidak pakai pre-trained model, tidak pakai API LLM, tidak pakai
backpropagation sebagai learning rule utama.

Status: **Phase 1 — Neuron Models** (in progress)

## Prinsip

| Prinsip | Arti |
|---|---|
| Spike-based | komunikasi lewat spike diskrit |
| Event-driven | komputasi hanya saat ada spike |
| Local learning | update bobot pakai aturan lokal (STDP) |
| Sparse | sedikit neuron aktif |
| Memory co-located | memori dan komputasi di tempat yang sama |
| Online | belajar terus-menerus |
| Explainable | setiap keputusan bisa dilacak |
| Ringan | jalan di CPU, tanpa GPU |

## Install

```bash
pip install -e ".[dev]"
```

## Jalan

```bash
python -m rak selfcheck          # smoke test
pytest                           # test suite (25 test)
python scripts/plot_neurons.py   # visualisasi neuron
```

## Struktur

```
src/rak/
  neuron/       LIF, Adaptive LIF, Izhikevich, SpikeLayer, trace, encode
tests/
  test_lif.py, test_adaptive_lif.py, test_izhikevich.py,
  test_layer.py, test_trace.py, test_encode_neuron.py
experiments/
  exp_001_neuron/   figures
```

## License

MIT