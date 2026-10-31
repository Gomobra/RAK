# ADR 0007 — Sumber data teks Indonesia untuk baseline Fase 1

Status: Diterima

## Konteks

Fase 1 (PRD bagian FASE 1, F1.1 Pipeline Data) membutuhkan data teks Indonesia yang bersih, berlisensi jelas, dan tersedia publik untuk melatih dua baseline (Transformer dan hybrid) serta harness evaluasi. PRD 1.7 dan 2.3 menegaskan: hanya data dengan lisensi yang jelas, lisensi dicatat per sumber, dan tidak ada bobot/tokenizer/codec pretrained.

## Keputusan

**Sumber data utama:** Wikipedia Indonesia (dump XML, CC-BY-SA).

**Cadangan (fallback):** OSCAR Indonesia (Common Crawl hasil pembersihan, ODC-BY) dan mC4 Indonesia (ODC-BY), keduanya tersedia di Hugging Face sebagai shard per bahasa.

**Prototipe cepat:** korpus cerita pendek Indonesia dalam jumlah kecil (berkas teks publik dengan lisensi yang jelas, mis. cerita dari domain publik atau karya dengan lisensi CC), hanya untuk smoke test pipeline sebelum data besar siap.

## Alasan pemilihan

| Sumber | Lisensi | Alasan utama |
|---|---|---|
| Wikipedia Indonesia | CC-BY-SA | Lisensi jelas, konten tereditasi, beragam topik, ukuran sedang (cocok untuk baseline 10–30M/100M di Fase 1), tersedia dump resmi dari Wikimedia. |
| OSCAR Indonesia | ODC-BY | Sudah melalui dedup dan filter kualitas dari Common Crawl, ukuran besar, cocok sebagai cadangan bila Wikipedia tidak mencukupi. |
| mC4 Indonesia | ODC-BY | Mirip OSCAR, bagian dari rilis mC4/mT5, tersedia per bahasa di Hugging Face. |
| Korpus cerita pendek Indonesia | CC / publik | Kecil, memungkinkan prototipe cepat dan uji pipeline end-to-end sebelum data besar diproses. |

## Strategi verifikasi lisensi dan kualitas

1. **Lisensi:** sebelum mengunduh, periksa lisensi resmi dari halaman sumber (Wikimedia dumps untuk Wikipedia; lisensi dataset di Hugging Face untuk OSCAR/mC4; lisensi per berkas untuk cerita pendek). Catat lisensi per sumber dalam `docs/STATUS.md` dan konfigurasi pipeline data (T1.1.1, T1.1.2). Jika lisensi tidak dapat diverifikasi, sumber tidak digunakan.
2. **Kualitas:** Wikipedia dianggap berkualitas tinggi (tereditasi). OSCAR/mC4 sudah dibersihkan (dedup, filter) tetapi masih perlu pembersihan tambahan: normalisasi Unicode/UTF-8, filter bahasa Indonesia, hapus duplikat, cek kebocoran train/validasi/uji (T1.1.2, T1.1.3).
3. **Ukuran vs memori:** gunakan shard streaming dan format biner dengan mmap (T1.1.4) agar data muat di RAM terbatas (Termux). Wikipedia Indonesia cukup untuk prototype; OSCAR/mC4 dipakai sebagai cadangan skalanya besar.
4. **Reproduktibilitas:** catat versi dump/tanggal unduh, hash config per run, dan seed (PRD 2.4, 5.2).

## Konsekuensi

- Pipeline data harus mencatat lisensi per sumber dan gagal dengan jelas jika lisensi tidak diverifikasi.
- Data diletakkan di `data/` (di-gitignore), hanya config dan ringkasan laporan yang di-commit.
- Sumber cadangan (OSCAR/mC4) hanya diunduh bila Wikipedia tidak mencukupi untuk baseline target ukuran.
