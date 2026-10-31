# ADR 0008 — Format shard biner dan streaming sampler (F1.1 Pipeline Data)

Status: Diterima

## Konteks

Fase 1 (PRD F1.1 Pipeline Data) membutuhkan alur data dari teks mentah ke shard biner siap latih: pembersihan, deduplikasi, filter kualitas, pembagian train/validasi/uji yang tidak bocor, dan pencatatan lisensi per sumber. Sampler harus membaca shard secara streaming agar muat di RAM terbatas (Termux). ADR ini merancang format shard biner yang mmap-compatible, strategi dedup, cek kebocoran train/uji, format split, dan streaming sampler untuk T1.1.2–T1.1.5. ADR ini melengkapi ADR 0007 (sumber data).

## Keputusan

### 1. Format shard biner (RAKSHARD v1)

Setiap shard satu berkas biner dengan tata berikut, semuanya little-endian, tanpa pointer absolut, siap mmap:

- Header (64 byte, rata kanan):
  - magic: 8 byte `"RAKSHARD\0"`
  - version: uint16 = 1
  - token_dtype: uint8 (1 = uint16, 2 = uint32)
  - reserved: 3 byte
  - doc_count: uint64 (jumlah dokumen dalam shard)
  - token_count: uint64 (total token di seluruh dokumen)
  - index_offset: uint64 (byte offset ke tabel indeks dari awal berkas)
  - index_size: uint64 (ukuran byte tabel indeks)
  - header_checksum: uint32 (CRC32 atas 60 byte pertama header)
- Data token: blok token kontigu (uint16 atau uint32 menurut token_dtype)
- Tabel indeks: array berurutan, satu entri per dokumen (urutan = indeks dokumen), setiap entri 32 byte (ukuran tetap, akses acak O(1)):
  - token_offset: uint64 (offset ke token pertama dokumen, relatif awal berkas)
  - token_length: uint64 (jumlah token dokumen)
  - doc_hash: uint64 (hash 64-bit isi dokumen, untuk dedup & leak check)
  - source_id: uint32 (referensi ke tabel sumber)
  - length_chars: uint32 (panjang teks asli dalam karakter UTF-8)
  - reserved: 8 byte
- Tabel sumber (opsional, diakhir): array `{source_id, source_path, lisensi}` untuk pelacakan lisensi per sumber.

### 2. mmap-compatible

Semua offset adalah absolut dari awal berkas (tidak ada pointer ke alamat memori proses), header dan indeks pada posisi tetap, endianness fixed little-endian, entri indeks ukuran tetap. Buka berkas, mmap seluruhnya, baca header, indeks, dan token langsung lewat pointer memori tanpa parsing penuh. Pada Termux (RAM terbatas), mmap membiarkan OS memuat hanya halaman yang diakses; sampler streaming hanya menyentuh halaman yang dibutuhkan. Tidak perlu memuat seluruh shard ke RAM.

### 3. Streaming sampler

- Input: daftar shard + manifest split (train/valid/test)
- Baca shard secara bertahap (per dokumen) melalui mmap
- Buffer shuffling: isi buffer sejumlah dokumen (contoh 1000 dokumen), acak dengan Fisher-Yates menggunakan PRNG berseed, keluarkan satu per satu, isi ulang. Ukuran buffer konfigurable agar pas RAM.
- Multiple shard: tiap epoch, ulangi daftar shard (round-robin atau bobot per shard); ambil dokumen dari shard yang berbeda secara bergantian agar satu shard besar tidak mendominasi batch.
- Deterministik: seluruh acakan berasal dari PRNG dengan seed yang dicatat; urutan shard dan urutan dokumen dalam buffer dapat direproduksi pada mesin yang sama.
- Split: pembagian dilakukan pada tingkat dokumen (sebelum atau selama pembuatan shard) dengan proporsi konfigurable (train/valid/test, default 90/5/5).

### 4. Dedup strategi

- Exact match: hitung hash 64-bit (FNV-1a atau xxhash) atas isi dokumen; dokumen dengan hash sama dianggap duplikat eksak, sisakan satu.
- Near-duplicate: simhash ringan (64-bit) atas fitur n-gram karakter/token dokumen; bandingkan jarak Hamming, jika di bawah ambang (default 3) anggap near-duplicate dan buang satu. Simhash ringan agar ringan di Termux (tidak perlu library berat).
- Dedup dilakukan sebelum sharding, hasilnya ditulis ke berkas indeks hash agar bisa diverifikasi dan direproduksi.

### 5. Cek kebocoran train/uji

- Setiap dokumen punya doc_hash. Saat pembagian split, bangun himpunan hash per split.
- Verifikasi: irisan antara himpunan hash train dan test harus kosong. Tidak ada dokumen yang muncul di kedua split.
- Pembagian split deterministik berdasarkan doc_hash (mis. hash % 100 < pct_test) sehingga dokumen yang sama selalu ke split yang sama, dan uji ulang menghasilkan split identik.

### 6. Format split

- Setiap split direpresentasikan oleh manifest (berkas JSON/teks) yang mencantumkan daftar shard dan rentang dokumen yang masuk ke split tersebut.
- Proporsi train/valid/test konfigurable (default 90/5/5, dapat diubah per eksperimen).
- Manifest mencatat seed, versi shard, dan daftar doc_hash per split untuk audit kebocoran.

### 7. Set uji tetap dan domain berurutan

- Set uji tetap: manifest uji yang dikunci (frozen), versi-controlled, tidak berubah antar run, digunakan untuk mengukur forgetting dan Bits-per-byte konsisten.
- Set domain berurutan: daftar terurut dari manifest/domain (domain A, lalu B, dst.) untuk uji continual learning; model belajar domain satu per satu, ukur lupa pada domain sebelumnya.

## Alasan pemilihan

| Pertimbangan | Keputusan | Alasan |
|---|---|---|
| Magic + versi | `"RAKSHARD"` v1 | Mudah deteksi berkas, versi untuk kompatibilitas ke depan |
| Token dtype | uint16 atau uint32 di header | uint16 hemat untuk vokab kecil; uint32 jika vokab besar |
| Checksum | CRC32 header | Deteksi korupsi berkas, ringan |
| mmap | Offset absolut, endian fixed | Tanpa parsing penuh, cocok RAM terbatas Termux |
| Dedup near-dup | Simhash 64-bit | Ringan, satu hash per dokumen, tidak perlu library berat |
| Split | Dokument level + hash | Dokument tidak terpecah antar split; verifikasi irisan hash kosong |
| Sampler | Buffer shuffling + PRNG seeded | Deterministik, hemat memori, cocok streaming |

## Strategi verifikasi

1. **Format:** tulis shard, baca kembali via mmap, bandingkan token dan metadata bit-per-bit.
2. **mmap:** ukur RAM — sampler tidak boleh memuat seluruh shard; hanya halaman yang diakses.
3. **Dedup:** masukkan duplikat sengaja, pastikan tersisihkan satu; near-dup dengan variasi teks kecil harus terdeteksi.
4. **Leak check:** buat split dengan dokumen sengaja muncul di kedua sisi, pastikan deteksi gagal (irisan tidak kosong) dan verifikasi normal menghasilkan irisan kosong.
5. **Determinisme:** jalankan dua kali dengan seed sama, bandingkan urutan sampel dan split bit-per-bit.
6. **Data real:** uji pada ≥ 3 seed, laporan mean ± std (PRD 2.4).
7. **Edge case:** dokumen kosong, 1 token, UTF-8 tidak valid, shard 0 dokumen, file rusak (checksum tidak cocok).

## Konsekuensi

- Pipeline data (Python) bertanggung jawab membuat shard, dedup, split, dan manifest; C++ hanya membaca shard via mmap dan menyediakan antarmuka sampler ringan.
- Semua hash dan seed dicatat per run untuk reproduktibilitas (PRD 5.2).
- Data berada di `data/` (di-gitignore); hanya config dan ringkasan laporan yang di-commit.
- ADR ini menjadi acuan implementasi T1.1.4 dan T1.1.5. Spesifikasi struktur data in-memory C++ akan dicatat di `docs/struktur-data.md` atau header `include/rak/train/` saat implementasi dimulai.
