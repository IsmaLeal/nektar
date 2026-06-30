from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover
    h5py = None


def _is_hdf5(path: Path) -> bool:
    return path.suffix.lower() in {".h5", ".hdf5"}


def load_data(path: Path) -> Any:
    if _is_hdf5(path):
        if h5py is None:
            raise RuntimeError("h5py is required to load HDF5 files.")
        return h5py.File(path, "r")
    return np.load(path, allow_pickle=True)


def read_array(data: Any, key: str) -> np.ndarray:
    if key not in data:
        raise KeyError(f"Missing required key '{key}' in input data.")
    return np.asarray(data[key])


def close_data(data: Any) -> None:
    if hasattr(data, "close"):
        data.close()
