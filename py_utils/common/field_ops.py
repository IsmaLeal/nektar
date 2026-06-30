from __future__ import annotations

import numpy as np


def normalize_time_axis(arr: np.ndarray, kind: str = "spatial_time") -> np.ndarray:
    """Return arrays with time on axis 0.

    The py_utils pipeline stores fields time-last:
    - spatial fields: (Ny, Nx, T)
    - mode fields: (S, Ny, Nx, T)
    """
    arr = np.asarray(arr)
    if kind == "spatial_time":
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D array, got shape {arr.shape}")
        return np.transpose(arr, (2, 0, 1))
    if kind == "modes_spatial_time":
        if arr.ndim != 4:
            raise ValueError(f"Expected 4D array, got shape {arr.shape}")
        return np.transpose(arr, (3, 0, 1, 2))
    raise ValueError(f"Unsupported kind: {kind}")


def divergence_2d(u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> np.ndarray:
    du_dx = np.gradient(u, dx, axis=-1, edge_order=2)
    dv_dy = np.gradient(v, dy, axis=-2, edge_order=2)
    return du_dx + dv_dy


def l2_norm(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(a) ** 2)))


def weighted_inner_product(a: np.ndarray, b: np.ndarray, cell_area: float) -> float:
    return float(np.sum(np.asarray(a) * np.asarray(b)) * cell_area)
