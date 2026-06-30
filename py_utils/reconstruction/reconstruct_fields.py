#!/usr/bin/env python3
"""Reconstruct physical realizations from DO mean/modes/coefficient arrays.

Expected arrays in the input container (NPZ/HDF5 from `py_utils/load_chk.py`):
- mean field: `u` and/or `v` with shape (Ny, Nx, T)
- mode field: `mode_u` and/or `mode_v` with shape (S, Ny, Nx, Tdo)
- coefficients: `yi` with shape (Np, S, Tdo)
- optional alignment: `common_idx_chk`, `common_idx_do`

Outputs reconstructed realizations, e.g. `u_real` with shape (Np, Ny, Nx, K):
    u_real = u_mean + sum_s yi[:,s,:] * mode_u[s,...]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover
    h5py = None


def _is_hdf5_path(path: Path) -> bool:
    return path.suffix.lower() in {".h5", ".hdf5"}


def _open_data(path: Path):
    if _is_hdf5_path(path):
        if h5py is None:
            raise RuntimeError(
                "h5py is required to read HDF5 input (.h5/.hdf5). "
                "Install h5py or use NPZ."
            )
        return h5py.File(path, "r")
    return np.load(path, allow_pickle=True)


def _keys(data) -> list[str]:
    if hasattr(data, "files"):
        return list(data.files)
    return list(data.keys())


def _arr(data, key: str) -> np.ndarray:
    return np.asarray(data[key])


def _save_arrays(path: Path, arrays: dict[str, object]) -> None:
    if _is_hdf5_path(path):
        if h5py is None:
            raise RuntimeError(
                "h5py is required to write HDF5 output (.h5/.hdf5). "
                "Install h5py or use NPZ."
            )
        with h5py.File(path, "w") as h5:
            for k, v in arrays.items():
                arr = np.asarray(v)
                if arr.dtype == object:
                    dt = h5py.string_dtype(encoding="utf-8")
                    ds = h5.create_dataset(k, shape=arr.shape, dtype=dt)
                    ds[...] = arr.astype(str)
                elif arr.dtype.kind in {"U", "S"}:
                    dt = h5py.string_dtype(encoding="utf-8")
                    ds = h5.create_dataset(k, shape=arr.shape, dtype=dt)
                    ds[...] = arr.astype(str)
                else:
                    h5.create_dataset(k, data=arr, compression="gzip")
        return
    np.savez_compressed(path, **arrays)


def _aligned_triplet(data, mean_key: str, mode_key: str):
    mean = _arr(data, mean_key)  # (Ny, Nx, T)
    mode = _arr(data, mode_key)  # (S, Ny, Nx, Tdo)
    yi = _arr(data, "yi")       # (Np, S, Tdo)

    if "common_idx_chk" in data and "common_idx_do" in data:
        ic = _arr(data, "common_idx_chk")
        id_ = _arr(data, "common_idx_do")
        if ic.size > 0 and id_.size > 0:
            mean = mean[..., ic]
            mode = mode[..., id_]
            yi = yi[..., id_]
    else:
        # Fallback: truncate to common trailing length.
        k = min(mean.shape[-1], mode.shape[-1], yi.shape[-1])
        mean = mean[..., :k]
        mode = mode[..., :k]
        yi = yi[..., :k]

    return mean, mode, yi


def reconstruct_component(data, mean_key: str, mode_key: str) -> np.ndarray:
    mean, mode, yi = _aligned_triplet(data, mean_key, mode_key)
    # yi: (P,S,K), mode: (S,Ny,Nx,K) -> (P,Ny,Nx,K)
    return mean[None, ...] + np.einsum("psk,sxyk->pxyk", yi, mode)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data", type=Path, help="Input NPZ/HDF5 from py_utils/load_chk.py")
    ap.add_argument("--out", type=Path, default=None, help="Output path (.npz or .h5/.hdf5)")
    ap.add_argument("--components", default="u,v", help="Comma-separated subset: u,v")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    args = ap.parse_args()

    data = _open_data(args.data)
    comps = [c.strip() for c in args.components.split(",") if c.strip()]
    out = {}

    for c in comps:
        mean_key = c
        mode_key = f"mode_{c}"
        if mean_key not in data:
            raise KeyError(f"Missing mean field '{mean_key}' in {args.data}")
        if mode_key not in data:
            raise KeyError(
                f"Missing interpolated mode field '{mode_key}' in {args.data}'. "
                "Run load_chk.py with --interp-modes-from-archive first."
            )
        arr = reconstruct_component(data, mean_key, mode_key).astype(args.dtype, copy=False)
        out[f"{c}_real"] = arr
        print(f"{c}_real shape = {arr.shape}")

    if "t" in data:
        out["t"] = _arr(data, "t")
    if "t_do" in data:
        out["t_do"] = _arr(data, "t_do")
    if "common_idx_chk" in data:
        out["common_idx_chk"] = _arr(data, "common_idx_chk")
    if "common_idx_do" in data:
        out["common_idx_do"] = _arr(data, "common_idx_do")

    out_path = args.out or args.data.with_name(args.data.stem + "_reconstructed.npz")
    _save_arrays(out_path, out)
    if hasattr(data, "close"):
        data.close()
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
