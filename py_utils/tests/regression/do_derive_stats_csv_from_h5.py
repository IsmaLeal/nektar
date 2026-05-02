#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover
    h5py = None


def _load_time(h5) -> np.ndarray:
    if "t_do" in h5:
        return np.asarray(h5["t_do"], dtype=float).reshape(-1)
    if "t" in h5:
        return np.asarray(h5["t"], dtype=float).reshape(-1)
    raise KeyError("HDF5 input has neither 't_do' nor 't'.")


def _load_diag_c(h5, nt: int) -> np.ndarray:
    if "archive_diag_cov" not in h5:
        raise KeyError("HDF5 input is missing 'archive_diag_cov'.")
    diag = np.asarray(h5["archive_diag_cov"], dtype=float)

    n_modes = None
    if "do_dims" in h5:
        dd = np.asarray(h5["do_dims"]).reshape(-1)
        if dd.size >= 3:
            n_modes = int(dd[2])

    if diag.ndim == 1:
        if n_modes is not None and diag.size == n_modes + 1:
            diag_c = diag[1:]
        else:
            diag_c = diag
        if diag_c.size != nt:
            raise ValueError(
                f"archive_diag_cov length {diag_c.size} is incompatible with time length {nt}."
            )
        return diag_c.reshape(1, -1)

    if diag.ndim == 2:
        if diag.shape[1] != nt:
            raise ValueError(
                f"archive_diag_cov shape {diag.shape} is incompatible with time length {nt}."
            )
        if n_modes is not None and diag.shape[0] == n_modes + 1:
            return diag[1:, :]
        return diag

    raise ValueError(f"Unexpected archive_diag_cov shape: {diag.shape}")


def _load_ortho_err(h5, nt: int) -> np.ndarray:
    if "archive_ortho_err" not in h5:
        return np.full((nt,), np.nan, dtype=float)
    arr = np.asarray(h5["archive_ortho_err"], dtype=float).reshape(-1)
    if arr.size == nt:
        return arr
    out = np.full((nt,), np.nan, dtype=float)
    n = min(nt, arr.size)
    out[:n] = arr[:n]
    return out


def _load_y_stats(h5, nt: int) -> tuple[np.ndarray, np.ndarray]:
    if "yi" not in h5:
        nan = np.full((nt,), np.nan, dtype=float)
        return nan, nan

    yi = np.asarray(h5["yi"], dtype=float)
    if yi.ndim != 3:
        nan = np.full((nt,), np.nan, dtype=float)
        return nan, nan

    k = yi.shape[2]
    y_rms = np.full((nt,), np.nan, dtype=float)
    y_max = np.full((nt,), np.nan, dtype=float)
    n = min(nt, k)
    if n > 0:
        sub = yi[:, :, :n]
        y_rms[:n] = np.sqrt(np.mean(sub * sub, axis=(0, 1)))
        y_max[:n] = np.max(np.abs(sub), axis=(0, 1))
    return y_rms, y_max


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Derive DO stats CSV from canonical SEM HDF5 archive/postprocessed output."
        )
    )
    ap.add_argument("--sem-h5", type=Path, required=True, help="Input out.h5")
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path("py_utils/tests/results/do_stats_derived.csv"),
        help="Output CSV path",
    )
    args = ap.parse_args()

    if h5py is None:
        raise RuntimeError("h5py is required to read --sem-h5 input.")

    with h5py.File(args.sem_h5, "r") as h5:
        t = _load_time(h5)
        nt = t.size
        diag_c = _load_diag_c(h5, nt)
        ortho = _load_ortho_err(h5, nt)
        y_rms, y_max = _load_y_stats(h5, nt)

    trace_c = np.sum(diag_c, axis=0)
    min_diag = np.min(diag_c, axis=0)
    max_diag = np.max(diag_c, axis=0)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "step",
                "time",
                "traceC",
                "minDiagC",
                "maxDiagC",
                "orthoErr",
                "covSymErr",
                "modeDirichletBCErr",
                "yRms",
                "yMaxAbs",
            ]
        )
        for i in range(nt):
            w.writerow(
                [
                    i + 1,
                    f"{t[i]:.16g}",
                    f"{trace_c[i]:.16g}",
                    f"{min_diag[i]:.16g}",
                    f"{max_diag[i]:.16g}",
                    f"{ortho[i]:.16g}",
                    "nan",
                    "nan",
                    f"{y_rms[i]:.16g}",
                    f"{y_max[i]:.16g}",
                ]
            )

    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
