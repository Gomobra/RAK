# ADR 0009 — Baseline Model Fase 1 (Transformer Decoder dan Hybrid Rekuren + Attention Lokal)

Status: Diajukan

## 1. Konteks

Fase 1 (PRD F1.2, T1.2.1–T1.2.5) membutuhkan dua baseline model yang dilatih dengan engine RAK C++17 sendiri (Tensor, autograd, ops L7), untuk menjadi pembanding sebelum membangun komponen RAK baru (Fase 2+). Tanpa baseline yang kuat dan terukur, klaim hipotesis inti RAK (belajar terus tanpa lupa, hemat data) tidak dapat dibuktikan.

Engine yang tersedia (Fase 0 sudah ASLI): GEMM, elementwise (add/mul/silu/gelu), reduksi (sum/mean/max), RMSNorm, softmax/log_softmax, embedding, recurrent_scan, local_attention, sparse_lookup_topk, sparse_update, AdamW optimizer (T0.4.1), checkpoint dengan CRC32 + RNG state (T0.4.3/T0.4.4).

Keterbatasan engine yang diakomodasi dalam desain:
- `local_attention` adalah attention dua arah pada jendela `[i-window, i+window]` tanpa masking causal bawaan. Untuk decoder Transformer yang membutuhkan causal self-attention, masking diterapkan di lapisan model dengan menambahkan matriks mask (0 pada posisi diizinkan, nilai sangat negatif pada posisi masa depan) ke logits sebelum softmax, menggunakan op `add` yang sudah ada. Tidak perlu op baru.
- Tidak ada op masked-softmax atau causal-attention dedicated; semua dibangun dari primitif ops di atas.

## 2. Keputusan

Dibangun dua baseline model dengan engine RAK, konfigurasi sama (data, token, compute, seed) untuk perbandingan adil:

### Model A — Transformer Decoder Murni
Stack decoder standar: token embedding (+ positional embedding learned), N blok decoder (causal MHA → residual → RMSNorm → FFN → residual → RMSNorm), output head linear tied dengan embedding, loss cross-entropy pada token berikutnya.

### Model B — Hybrid Rekuren + Attention Lokal
Setiap blok: cabang paralel `recurrent_scan` (pengkodean konteks panjang dengan state terkompres) dan `local_attention` (detail jendela dekat), digabungkan residual, diikuti FFN + RMSNorm. `sparse_lookup_topk` dipakai sebagai memory slot bank (opsional per blok). Tidak menggunakan komponen RAK lain (MoE, memory layer, adaptive loop, homeostasis, L12 language network), sesuai batas T1.2.2.

Ukuran target: 10M dan 30M parameter, minimal 3 seed per varian, data/token/compute sama antar model dan antar seed.

## 3. Spesifikasi Arsitektur

### 3.1 Transformer Decoder

| Parameter | 10M | 30M |
|---|---|---|
| d_model | 256 | 384 |
| n_layer | 8 | 12 |
| d_ff | 1024 | 1536 (4×d) |
| n_head | 8 | 12 (head_dim = 32) |
| vocab_size | 16384 | 16384 |
| max_seq | 1024 | 2048 |
| Output head | Tied dengan embedding | Tied dengan embedding |
| Aktivasi FFN | SiLU | SiLU |
| Normalisasi | RMSNorm (eps 1e-5) | RMSNorm (eps 1e-5) |
| Bias | Tidak ada (semua proyeksi) | Tidak ada |

Per blok decoder:
1. h₁ = RMSNorm(x)
2. attn = CausalMHA(h₁, h₁, h₁) — QKV via GEMM, skor QKᵀ, tambah mask causal via `add`, softmax, kali V via GEMM, output proyeksi via GEMM
3. h₂ = x + attn (residual)
4. h₃ = RMSNorm(h₂)
5. ff = FFN(h₃) — W₁ (d×d_ff), SiLU, W₂ (d_ff×d)
6. out = h₂ + ff (residual)

Forward loss: logits = h_final @ W_embᵀ (tied). Cross-entropy = -mean(log_softmax(logits)[target]) pada token target berikutnya, posisi padding diabaikan (mask).

### 3.2 Hybrid Rekuren + Attention Lokal

| Parameter | 10M | 30M |
|---|---|---|
| d_model | 256 | 384 |
| n_layer | 8 | 11 |
| d_ff | 1024 | 1536 |
| vocab_size | 16384 | 16384 |
| max_seq | 1024 | 2048 |
| recurrent decay | Per-fitur [d], diinisialisasi dekat 0.9 | Sama |
| local_attention window | 64 | 128 |
| sparse_lookup n_slot | 512 | 768 |
| sparse_lookup top_k | 4 | 4 |
| Memory bank | Bersama di semua lapisan | Bersama di semua lapisan |

Per blok hybrid (cabang paralel):
1. h = RMSNorm(x)
2. rec_branch = W_r_out · recurrent_scan(W_r_in · h, decay) — W_r_in, W_r_out (d² masing-masing), decay [d]
3. attn_branch = W_o · local_attention(W_q·h, W_k·h, W_v·h, window) — 4×d²
4. h = h + rec_branch + attn_branch (residual gabungan)
5. h = RMSNorm(h)
6. h = h + FFN(h) — 2·d·d_ff, SiLU
7. sparse_lookup (opsional): query dari h, ambil top-k slot dari memory bank (n_slot×d), gabungkan ke h via proyeksi ringan + `add`

Output head: tied dengan embedding. Loss cross-entropy sama seperti Transformer.

## 4. Estimasi Parameter

Rumus Transformer per blok: `4·d²` (MHA) + `2·d·d_ff` (FFN) + `2·d` (2×RMSNorm).
Total = n_layer·(4·d² + 2·d·d_ff + 2·d) + V·d (embedding tied) + max_seq·d (posisi).

Rumus Hybrid per blok: `2·d²` (rec in/out) + `d` (decay) + `4·d²` (attn QKVO) + `d` + `d` (2×RMSNorm) + `2·d·d_ff` (FFN) = `6·d² + 2·d·d_ff + 3·d`.
Total = n_layer·(6·d² + 2·d·d_ff + 3·d) + V·d + max_seq·d + n_slot·d (memory bank, satu salinan).

Semua angka di bawah adalah estimasi, akan diverifikasi dengan hitungan bobot aktual saat implementasi.

| Model | Varian | Estimasi Parameter |
|---|---|---|
| Transformer | 10M | ≈ 10.752.000 |
| Transformer | 30M | ≈ 28.320.768 |
| Hybrid | 10M | ≈ 11.933.696 |
| Hybrid | 30M | ≈ 30.093.696 |

Rincian per komponen:

**Transformer 10M (d=256, L=8, d_ff=1024, V=16384, max_seq=1024):**
- Per blok: 786.944 × 8 = 6.295.552
- Embedding tied: 4.194.304
- Posisi: 262.144
- Total: 10.752.000

**Transformer 30M (d=384, L=12, d_ff=1536, V=16384, max_seq=2048):**
- Per blok: 1.770.240 × 12 = 21.242.880
- Embedding tied: 6.291.456
- Posisi: 786.432
- Total: 28.320.768

**Hybrid 10M (d=256, L=8, d_ff=1024, V=16384, max_seq=1024, n_slot=512):**
- Per blok: 918.272 × 8 = 7.346.176
- Embedding tied: 4.194.304
- Posisi: 262.144
- Memory bank: 131.072
- Total: 11.933.696

**Hybrid 30M (d=384, L=11, d_ff=1536, V=16384, max_seq=2048, n_slot=768):**
- Per blok: 2.065.536 × 11 = 22.720.896
- Embedding tied: 6.291.456
- Posisi: 786.432
- Memory bank: 294.912
- Total: 30.093.696

## 5. Implementasi di Engine C++17

Semua model dibangun di atas `rak::core::Tensor` dan autograd reverse-mode (`g_autograd`), memakai ops L7 yang sudah ada:

- `rak::ops::embedding` — token lookup
- `rak::ops::gemm` — proyeksi linear (QKV, output, FFN, output head)
- `rak::ops::rms_norm` — normalisasi
- `rak::ops::softmax` / `rak::ops::log_softmax` — attention dan loss
- `rak::ops::add` / `rak::ops::mul` — residual, mask causal
- `rak::ops::silu` — aktivasi FFN
- `rak::ops::recurrent_scan` — state rekuren hybrid
- `rak::ops::local_attention` — attention jendela hybrid
- `rak::ops::sparse_lookup_topk` — memory slot bank hybrid

Backprop melalui `Autograd::backward(loss_tensor)`; gradien akumulasi ke buffer `grad` tiap leaf parameter. Optimizer AdamW (`rak::train::optimizer`, T0.4.1) dengan gradient clipping (norm global) dan jadwal LR warmup linear + cosine decay (T0.4.1). Checkpoint berisi bobot + state optimizer + state RNG + posisi data, verifikasi CRC32, resume deterministik (T0.4.3/T0.4.4).

Pemakaian: sesuai PRD 6.2, Python untuk orkestrasi/eksperimen dan C++ untuk inti. Model dapat dirangkai di Python lewat binding `_rak` (tensor/op), sehingga perubahan konfigurasi cepat, sedangkan forward/backward inner-loop berjalan di C++. Header model ditempatkan di `cpp/include/rak/models/` dan program latihan di `cpp/apps/`, atau orkestrasi di `src/rak/train/`.

## 6. Pipeline Training

1. Forward: token input → embedding → blok-belok → output head → logits [batch, seq, vocab]
2. Loss: cross-entropy (log_softmax + NLL) pada token target berikutnya, mask posisi padding, rata-rata per token
3. Backward: `Autograd::backward(loss)` → gradien semua parameter
4. Optimizer: AdamW update parameter, gradient clipping, LR scheduler
5. Evaluasi validation per epoch: loss, perplexity, bits-per-byte
6. Checkpoint: bobot + state optimizer + RNG state + posisi data, CRC32, resume deterministik

## 7. Konfigurasi, Seed, dan Pelacakan Kurva Latihan

- Satu berkas YAML per eksperimen di `configs/` (contoh: `configs/baseline_transformer_10m.yaml`, `configs/baseline_hybrid_10m.yaml`). Hash konfigurasi dicatat per run.
- Seed global dicatat per run; minimal 3 seed per varian (contoh: 42, 123, 777). Digunakan untuk inisialisasi bobot, shuffling data, dan dropout (jika ada).
- Pelacakan kurva: training loss, validation loss/perplexity (bits-per-byte), learning rate, norma gradien, waktu langkah per langkah, diringkas ke laporan otomatis mean ± std di `experiments/reports/`.
- Evaluasi: bits-per-byte dan perplexity pada set validasi/uji tetap.

## 8. Konsekuensi

- Kedua baseline harus dapat kalah jujur sebelum klaim RAK unggul (PRD 1.6 risiko utama); hasil negatif tetap dilaporkan.
- Transformer causal masking via `add` mask + softmax menambah satu operasi elemen-wise per head, tidak memerlukan op baru.
- Hybrid memanfaatkan ops RAK yang sudah ada tanpa komponen baru, sesuai T1.2.2.
- Semua target 10–30M berjalan di CPU; CUDA opsional via flag `RAK_CUDA`.

## 9. Langkah Selanjutnya (DoD F1.2, PRD 2.5)

- Implementasi kedua model sesuai spesifikasi ini (T1.2.1, T1.2.2).
- Latih keduanya dengan data, token, compute sama (T1.2.3).
- Skala ke 100M dijadwalkan Fase 6 (T1.2.4).
- Catat konfigurasi, seed, dan kurva latihan (T1.2.5) — tercakup bagian 7.
- Gradient check, determinisme, test lengkap, benchmark, dan data real ≥ 3 seed sebelum naik status ke ASLI.
