"""Readout linear untuk RAK.

Modul ini TIDAK menyentuh bobot SNN. Ia hanya memetakan spike count
menjadi kelas, dilatih di data TRAIN, dipilih di validasi TRAIN.

    from rak.readout import RidgeReadout
    readout = RidgeReadout().fit(x_train, y_train)
    pred = readout.predict(x_test)
"""

from rak.readout.features import collect_counts, ensemble_counts
from rak.readout.linear import RidgeReadout, Standardizer

__all__ = ["RidgeReadout", "Standardizer", "collect_counts", "ensemble_counts"]
