"""RAK (Recurrent Adaptive Knowledge), Vbeta.

Paket Python untuk orkestrasi, konfigurasi, pipeline data, eksperimen, dan evaluasi.
Inti berat ada di C++ dan dipanggil lewat modul native ``_rak``.
"""

from rak._native import native

__all__ = ["native"]
