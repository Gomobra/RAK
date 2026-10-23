"""Wrapper lama: readout ridge kini jadi fitur bawaan paket.

Gunakan `python -m rak readout ...` atau skrip ini (keduanya sama).
Lihat `rak/readout/` untuk implementasinya.
"""

from rak.readout.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
