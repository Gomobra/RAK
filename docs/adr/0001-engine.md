# ADR 0001 — Engine tulis sendiri

Status: Diterima

## Konteks
RAK membutuhkan tensor, autograd, dan kernel yang dapat ditelusuri dan dikontrol.

## Keputusan
Engine ditulis sendiri di C++ (autograd reverse-mode, tape). PyTorch hanya pembanding di mesin dev.

## Konsekuensi
- Kontrol penuh atas numerik dan memori.
- Beban pemeliharaan lebih besar; gradient check wajib per op.
