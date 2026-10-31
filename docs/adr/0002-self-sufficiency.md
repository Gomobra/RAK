# ADR 0002 — Mandiri (tanpa pretrained)

Status: Diterima

## Konteks
Klaim mandiri mensyaratkan latih dari nol, kernel sendiri, tokenizer/codec sendiri.

## Keputusan
Tidak menggunakan bobot, tokenizer, atau codec pretrained. Data dengan lisensi jelas.

## Konsekuensi
- Reproduksibilitas dan kontrol lisensi.
- Waktu latih lebih panjang; prototipe kecil dulu.
