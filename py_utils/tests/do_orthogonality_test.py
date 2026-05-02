#!/usr/bin/env python3
"""DO test: orthogonality/normalization of DO modes and coefficient stats.

Checks:
- Solver-space orthogonality from DO archive diagnostics when available.
- Optional grid-space Gram matrix diagnostic on interpolated modes.
- Optional yi coefficients are approximately zero-mean over particles.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from py_utils.common import (
        close_data,
        load_data,
        normalize_time_axis,
        read_array,
        weighted_inner_product,
    )
    from py_utils.tests.test_common import summarize_check, write_json_report
else:
    from ..common import close_data, load_data, normalize_time_axis, read_array, weighted_inner_product
    from .test_common import summarize_check, write_json_report


def _gram_error(mode_u_t: np.ndarray, mode_v_t: np.ndarray, cell_area: float) -> tuple[float, float]:
    nt, nmode = mode_u_t.shape[0], mode_u_t.shape[1]
    errs = np.empty((nt,), dtype=float)
    for k in range(nt):
        g = np.zeros((nmode, nmode), dtype=float)
        for i in range(nmode):
            for j in range(nmode):
                g[i, j] = (
                    weighted_inner_product(mode_u_t[k, i], mode_u_t[k, j], cell_area)
                    + weighted_inner_product(mode_v_t[k, i], mode_v_t[k, j], cell_area)
                )
        errs[k] = float(np.linalg.norm(g - np.eye(nmode), ord="fro"))
    return float(np.median(errs)), float(np.max(errs))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--solver-ortho-thresh", type=float, default=1.0e-10)
    ap.add_argument("--gram-thresh", type=float, default=1.0e-2)
    ap.add_argument(
        "--enforce-grid-gram",
        action="store_true",
        help="Fail the test if interpolated-grid Gram error exceeds --gram-thresh.",
    )
    ap.add_argument("--yi-mean-thresh", type=float, default=2.0e-2)
    ap.add_argument("--report-json", type=Path, default=None)
    args = ap.parse_args()

    data = load_data(args.data)
    try:
        checks = []

        # Primary: solver-space orthogonality exported by DO archive writer.
        if "archive_ortho_err" in data:
            ortho = np.asarray(data["archive_ortho_err"], dtype=float)
            ortho_med = float(np.median(ortho)) if ortho.size else 0.0
            ortho_max = float(np.max(ortho)) if ortho.size else 0.0
            checks.append(
                summarize_check(
                    "mode_orthogonality_solver_space",
                    ortho_max <= args.solver_ortho_thresh,
                    {
                        "median_error": ortho_med,
                        "max_error": ortho_max,
                        "threshold": args.solver_ortho_thresh,
                    },
                )
            )

        # Secondary: interpolated-grid Gram matrix (diagnostic by default).
        if all(k in data for k in ["x", "y", "mode_u", "mode_v"]):
            x = read_array(data, "x")
            y = read_array(data, "y")
            dx = float(np.mean(np.diff(x[0, :])))
            dy = float(np.mean(np.diff(y[:, 0])))
            if dx <= 0 or dy <= 0:
                raise ValueError(f"Invalid grid spacing: dx={dx}, dy={dy}")
            cell_area = dx * dy

            mode_u_t = normalize_time_axis(read_array(data, "mode_u"), kind="modes_spatial_time")
            mode_v_t = normalize_time_axis(read_array(data, "mode_v"), kind="modes_spatial_time")
            nt = min(mode_u_t.shape[0], mode_v_t.shape[0])
            mode_u_t = mode_u_t[:nt]
            mode_v_t = mode_v_t[:nt]

            gram_med, gram_max = _gram_error(mode_u_t, mode_v_t, cell_area)
            gram_ok = gram_med <= args.gram_thresh
            checks.append(
                summarize_check(
                    "mode_gram_identity_grid",
                    gram_ok if args.enforce_grid_gram else True,
                    {
                        "median_frobenius_error": gram_med,
                        "max_frobenius_error": gram_max,
                        "threshold": args.gram_thresh,
                        "enforced": bool(args.enforce_grid_gram),
                    },
                )
            )
        elif "archive_ortho_err" not in data:
            raise KeyError(
                "Missing orthogonality inputs. Need either archive_ortho_err "
                "or grid arrays x,y,mode_u,mode_v."
            )

        if "yi" in data:
            yi = np.asarray(data["yi"])
            if yi.ndim != 3:
                raise ValueError(f"Expected yi shape (P,S,T), got {yi.shape}")
            yi_mean = np.mean(yi, axis=0)
            yi_abs = np.abs(yi_mean)
            yi_med = float(np.median(yi_abs))
            yi_max = float(np.max(yi_abs))
            checks.append(
                summarize_check(
                    "yi_zero_mean",
                    yi_med <= args.yi_mean_thresh,
                    {
                        "median_abs_mean": yi_med,
                        "max_abs_mean": yi_max,
                        "threshold": args.yi_mean_thresh,
                    },
                )
            )

        passed = all(c["passed"] for c in checks)
        payload = {
            "test": "do_orthogonality_test",
            "passed": passed,
            "checks": checks,
            "data": str(args.data),
        }
        print(json.dumps(payload, indent=2))
        write_json_report(args.report_json, payload)
        return 0 if passed else 2
    finally:
        close_data(data)


if __name__ == "__main__":
    raise SystemExit(main())
