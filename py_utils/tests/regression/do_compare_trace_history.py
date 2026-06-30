#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Tuple

import numpy as np
try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover
    h5py = None


def _load_sem_stats_csv(path: Path) -> np.ndarray:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append(
                (
                    int(r["step"]),
                    float(r["time"]),
                    float(r["traceC"]),
                    float(r.get("yRms", "nan")),
                    float(r.get("yMaxAbs", "nan")),
                )
            )
    if not rows:
        raise ValueError(f"No rows found in SEM stats CSV: {path}")
    return np.asarray(rows, dtype=float)


def _load_sem_from_h5(path: Path) -> np.ndarray:
    if h5py is None:
        raise RuntimeError("h5py is required to read --sem-h5 input.")
    with h5py.File(path, "r") as h5:
        if "archive_diag_cov" not in h5:
            raise KeyError(
                f"{path} is missing dataset 'archive_diag_cov' needed for DO trace."
            )
        diag = np.asarray(h5["archive_diag_cov"], dtype=float)
        n_modes = None
        if "do_dims" in h5:
            dd = np.asarray(h5["do_dims"]).reshape(-1)
            if dd.size >= 3:
                n_modes = int(dd[2])
        if "t_do" in h5:
            t = np.asarray(h5["t_do"], dtype=float).reshape(-1)
        elif "t" in h5:
            t = np.asarray(h5["t"], dtype=float).reshape(-1)
        else:
            raise KeyError(f"{path} has neither 't_do' nor 't' dataset.")

        if diag.ndim == 1:
            if n_modes is not None and diag.size == n_modes + 1:
                diag_c = diag[1:]
            else:
                diag_c = diag
            trace = diag_c.reshape(-1)
            if trace.size != t.size:
                raise ValueError(
                    f"Incompatible sizes: trace={trace.size}, time={t.size}."
                )
        elif diag.ndim == 2:
            if diag.shape[1] != t.size:
                raise ValueError(
                    f"Incompatible shapes: archive_diag_cov={diag.shape}, time={t.shape}."
                )
            if n_modes is not None and diag.shape[0] == n_modes + 1:
                diag_c = diag[1:, :]
            else:
                diag_c = diag
            trace = np.sum(diag_c, axis=0)
        else:
            raise ValueError(
                f"Unexpected archive_diag_cov shape {diag.shape}; expected 1D/2D."
            )

    rows = np.empty((t.size, 5), dtype=float)
    rows[:, 0] = np.arange(1, t.size + 1, dtype=float)
    rows[:, 1] = t
    rows[:, 2] = trace
    rows[:, 3] = np.nan
    rows[:, 4] = np.nan
    return rows


def _load_ns2d_trace_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    rows_t = []
    rows_tr = []
    with path.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows_t.append(float(r["time"]))
            rows_tr.append(float(r["traceC"]))
    if not rows_t:
        raise ValueError(f"No rows found in ns2d trace CSV: {path}")
    return np.asarray(rows_t, dtype=float), np.asarray(rows_tr, dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Post-processing only: compare SEM vs ns2d_do trace(C) history from "
            "precomputed CSV files."
        )
    )
    ap.add_argument(
        "--sem-stats-csv",
        type=Path,
        default=None,
        help="SEM stats CSV with columns: step,time,traceC[,yRms,yMaxAbs]",
    )
    ap.add_argument(
        "--sem-h5",
        type=Path,
        default=None,
        help="SEM postprocessed HDF5 with 'archive_diag_cov' and 't_do' datasets",
    )
    ap.add_argument(
        "--ns2d-trace-csv",
        type=Path,
        required=True,
        help="ns2d trace CSV with columns: time,traceC",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path("py_utils/tests/results/do_trace_compare.csv"),
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=Path("py_utils/tests/results/do_trace_compare_summary.json"),
    )
    args = ap.parse_args()

    if (args.sem_stats_csv is None) == (args.sem_h5 is None):
        raise ValueError("Pass exactly one of --sem-stats-csv or --sem-h5.")
    sem = (
        _load_sem_stats_csv(args.sem_stats_csv)
        if args.sem_stats_csv is not None
        else _load_sem_from_h5(args.sem_h5)
    )
    t_ns, tr_ns = _load_ns2d_trace_csv(args.ns2d_trace_csv)

    sem_t = sem[:, 1]
    sem_tr = sem[:, 2]
    tr_ns_on_sem = np.interp(sem_t, t_ns, tr_ns)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8") as f:
        f.write("step,time,trace_sem,trace_ns2d,ratio_sem_over_ns2d,yRms,yMaxAbs\n")
        for i in range(sem.shape[0]):
            ratio = sem_tr[i] / max(tr_ns_on_sem[i], 1e-30)
            f.write(
                f"{int(sem[i,0])},{sem_t[i]:.12g},{sem_tr[i]:.12g},"
                f"{tr_ns_on_sem[i]:.12g},{ratio:.12g},{sem[i,3]:.12g},"
                f"{sem[i,4]:.12g}\n"
            )

    summary = {
        "sem_stats_csv": str(args.sem_stats_csv) if args.sem_stats_csv is not None else None,
        "sem_h5": str(args.sem_h5) if args.sem_h5 is not None else None,
        "ns2d_trace_csv": str(args.ns2d_trace_csv),
        "sem_trace_final": float(sem_tr[-1]),
        "ns2d_trace_final_interp": float(tr_ns_on_sem[-1]),
        "ratio_final_sem_over_ns2d": float(sem_tr[-1] / max(tr_ns_on_sem[-1], 1e-30)),
        "sem_trace_max": float(np.max(sem_tr)),
        "ns2d_trace_max_interp": float(np.max(tr_ns_on_sem)),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.out_csv} and {args.out_json}")


if __name__ == "__main__":
    main()
