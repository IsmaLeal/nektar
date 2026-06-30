#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover
    h5py = None


def _extract_metric(log_text: str, pattern: str) -> Optional[float]:
    m = re.search(pattern, log_text)
    if not m:
        return None
    return float(m.group(1))


def _extract_tag_value(text: str, tag: str) -> Optional[float]:
    m = re.search(rf"<{tag}>([^<]+)</{tag}>", text)
    if not m:
        return None
    return float(m.group(1))


def _as_float_or_nan(v: Optional[float]) -> float:
    return np.nan if v is None else float(v)


def _get_nested_value(dct: dict, key: str) -> Optional[float]:
    if key in dct:
        try:
            return float(dct[key])
        except Exception:
            return None
    for v in dct.values():
        if isinstance(v, dict):
            out = _get_nested_value(v, key)
            if out is not None:
                return out
    return None


def _sem_from_fld(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8")
    return {
        "trace_c": _as_float_or_nan(_extract_tag_value(text, "DO_TraceC")),
        "min_diag_c": _as_float_or_nan(_extract_tag_value(text, "DO_MinDiagC")),
        "max_diag_c": _as_float_or_nan(_extract_tag_value(text, "DO_MaxDiagC")),
        "ortho_err": _as_float_or_nan(_extract_tag_value(text, "DO_OrthoErr")),
        "cov_sym_err": _as_float_or_nan(_extract_tag_value(text, "DO_CovSymErr")),
        "cov_cond_proxy": _as_float_or_nan(_extract_tag_value(text, "DO_CovCondProxy")),
    }


def _sem_from_h5(path: Path) -> dict[str, float]:
    if h5py is None:
        raise RuntimeError("h5py is required to read --sem-h5 input.")
    with h5py.File(path, "r") as h5:
        if "archive_diag_cov" not in h5:
            raise KeyError(
                f"{path} is missing dataset 'archive_diag_cov' needed for DO stats."
            )
        diag = np.asarray(h5["archive_diag_cov"], dtype=float)
        n_modes = None
        if "do_dims" in h5:
            dd = np.asarray(h5["do_dims"]).reshape(-1)
            if dd.size >= 3:
                n_modes = int(dd[2])

        if diag.ndim == 1:
            if n_modes is not None and diag.size == n_modes + 1:
                # archive row0 stores mean kinetic energy, rows 1..S are diag(C)
                diag_last = diag[1:]
            else:
                diag_last = diag
        elif diag.ndim == 2:
            diag_cols = diag[:, -1]
            if n_modes is not None and diag.shape[0] == n_modes + 1:
                # archive row0 stores mean kinetic energy, rows 1..S are diag(C)
                diag_last = diag_cols[1:]
            else:
                diag_last = diag_cols
        else:
            raise ValueError(
                f"Unexpected archive_diag_cov shape {diag.shape}; expected 1D/2D."
            )
        ortho = np.nan
        if "archive_ortho_err" in h5:
            oo = np.asarray(h5["archive_ortho_err"])
            if oo.size > 0:
                ortho = float(oo.reshape(-1)[-1])
        return {
            "trace_c": float(np.sum(diag_last)),
            "min_diag_c": float(np.min(diag_last)) if diag_last.size else np.nan,
            "max_diag_c": float(np.max(diag_last)) if diag_last.size else np.nan,
            "ortho_err": ortho,
            "cov_sym_err": np.nan,
            "cov_cond_proxy": np.nan,
        }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Post-processing only: compare SEM DO scalar metrics against precomputed "
            "ns2d_do metrics."
        )
    )
    ap.add_argument(
        "--sem-fld",
        type=Path,
        default=None,
        help="SEM XML .fld file with DO metadata tags",
    )
    ap.add_argument(
        "--sem-h5",
        type=Path,
        default=None,
        help="SEM postprocessed HDF5 (e.g., py_utils/load_chk.py out.h5)",
    )
    ap.add_argument(
        "--sem-log",
        type=Path,
        default=None,
        help="Optional SEM log file for L2 metrics",
    )
    ap.add_argument(
        "--ns2d-json",
        type=Path,
        required=True,
        help="Precomputed ns2d_do metrics JSON (must include trace/min diag values).",
    )
    ap.add_argument(
        "--json-out",
        type=Path,
        default=Path("py_utils/tests/results/do_sem_vs_ns2d_stats.json"),
    )
    args = ap.parse_args()

    if (args.sem_fld is None) == (args.sem_h5 is None):
        raise ValueError("Pass exactly one of --sem-fld or --sem-h5.")

    sem = _sem_from_fld(args.sem_fld) if args.sem_fld is not None else _sem_from_h5(args.sem_h5)
    sem["l2_u"] = np.nan
    sem["l2_v"] = np.nan
    sem["l2_p"] = np.nan

    if args.sem_log is not None:
        log_text = args.sem_log.read_text(encoding="utf-8", errors="replace")
        sem["l2_u"] = _as_float_or_nan(
            _extract_metric(log_text, r"L 2 error \(variable u\)\s*:\s*([0-9eE+\-\.]+)")
        )
        sem["l2_v"] = _as_float_or_nan(
            _extract_metric(log_text, r"L 2 error \(variable v\)\s*:\s*([0-9eE+\-\.]+)")
        )
        sem["l2_p"] = _as_float_or_nan(
            _extract_metric(log_text, r"L 2 error \(variable p\)\s*:\s*([0-9eE+\-\.]+)")
        )

    ns2d_raw = json.loads(args.ns2d_json.read_text(encoding="utf-8"))
    ns2d = {
        "trace_c": _as_float_or_nan(_get_nested_value(ns2d_raw, "trace_c")),
        "min_diag_c": _as_float_or_nan(_get_nested_value(ns2d_raw, "min_diag_c")),
        "max_diag_c": _as_float_or_nan(_get_nested_value(ns2d_raw, "max_diag_c")),
        "energy_mean": _as_float_or_nan(_get_nested_value(ns2d_raw, "energy_mean")),
        "energy_mode_sum": _as_float_or_nan(
            _get_nested_value(ns2d_raw, "energy_mode_sum")
        ),
    }

    out = {
        "sem_fld": str(args.sem_fld) if args.sem_fld is not None else None,
        "sem_h5": str(args.sem_h5) if args.sem_h5 is not None else None,
        "sem_log": str(args.sem_log) if args.sem_log is not None else None,
        "ns2d_json": str(args.ns2d_json),
        "sem": sem,
        "ns2d_do": ns2d,
        "ratios": {
            "trace_c_sem_over_ns2d": sem["trace_c"] / max(ns2d["trace_c"], 1e-30),
            "minddiag_sem_over_ns2d": sem["min_diag_c"]
            / max(ns2d["min_diag_c"], 1e-30),
        },
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
