#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover
    h5py = None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot SEM DO mode variances vs time from DO stats CSV."
    )
    ap.add_argument(
        "--stats-csv",
        default=None,
    )
    ap.add_argument(
        "--sem-h5",
        default=None,
        help="SEM postprocessed HDF5 with 'archive_diag_cov' and 't_do'",
    )
    ap.add_argument("--out", default="py_utils/tests/results/DO_mode_variance_vs_time.png")
    ap.add_argument("--title", default="DO mode variances vs time (SEM, S=2, rotated)")
    args = ap.parse_args()

    if (args.stats_csv is None) == (args.sem_h5 is None):
        raise ValueError("Pass exactly one of --stats-csv or --sem-h5.")

    root = Path(__file__).resolve().parents[3]
    t = []
    v_max = []
    v_min = []
    tr = []
    if args.stats_csv is not None:
        p = Path(args.stats_csv)
        if not p.is_absolute():
            p = root / p
        if not p.exists():
            raise FileNotFoundError(f"Stats CSV not found: {p}")

        with p.open("r", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                time = float(r["time"])
                min_d = float(r["minDiagC"])
                max_d = float(r["maxDiagC"])
                trace = float(r["traceC"])
                t.append(time)
                v_max.append(max_d)
                v_min.append(min_d)
                tr.append(trace)
    else:
        if h5py is None:
            raise RuntimeError("h5py is required to read --sem-h5 input.")
        p = Path(args.sem_h5)
        if not p.is_absolute():
            p = root / p
        if not p.exists():
            raise FileNotFoundError(f"HDF5 file not found: {p}")
        with h5py.File(p, "r") as h5:
            if "archive_diag_cov" not in h5:
                raise KeyError(
                    f"{p} is missing dataset 'archive_diag_cov' needed for plotting."
                )
            if "t_do" in h5:
                tt = np.asarray(h5["t_do"], dtype=float).reshape(-1)
            elif "t" in h5:
                tt = np.asarray(h5["t"], dtype=float).reshape(-1)
            else:
                raise KeyError(f"{p} has neither 't_do' nor 't' dataset.")
            diag = np.asarray(h5["archive_diag_cov"], dtype=float)
            n_modes = None
            if "do_dims" in h5:
                dd = np.asarray(h5["do_dims"]).reshape(-1)
                if dd.size >= 3:
                    n_modes = int(dd[2])
            if diag.ndim != 2:
                raise ValueError(
                    f"Expected archive_diag_cov to be 2D (S,nt), got {diag.shape}"
                )
            if diag.shape[1] != tt.size:
                raise ValueError(
                    f"archive_diag_cov shape {diag.shape} is incompatible with time size {tt.size}."
                )
            if n_modes is not None and diag.shape[0] == n_modes + 1:
                # archive row0 stores mean kinetic energy, rows 1..S are diag(C)
                diag = diag[1:, :]
            t = tt.tolist()
            v_max = np.max(diag, axis=0).tolist()
            v_min = np.min(diag, axis=0).tolist()
            tr = np.sum(diag, axis=0).tolist()

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    ax.plot(t, v_max, label="max diag(C)", linewidth=2.0)
    ax.plot(t, v_min, label="min diag(C)", linewidth=2.0)
    ax.plot(t, tr, label="trace(C)", linewidth=1.5, linestyle="--", alpha=0.8)
    ax.set_xlabel("time")
    ax.set_ylabel("variance")
    ax.set_title(args.title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
