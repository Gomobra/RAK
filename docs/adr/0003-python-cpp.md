# ADR 0003 — Python orkestrasi, C++ inti

Status: Diterima

## Konteks
Inti butuh kecepatan; orkestrasi butuh fleksibilitas.

## Keputusan
Inti di C++ (binding pybind11/nanobind), orkestrasi/data/eval di Python. CMake satu perintah.

## Konsekuensi
- Build ekstensi native di Termux perlu diuji.
- Tensor dibagi tanpa salin lewat buffer protocol.
