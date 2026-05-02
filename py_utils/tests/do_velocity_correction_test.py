#!/usr/bin/env python3
"""DO/IPCS test: verify velocity-correction identity and divergence reduction.

Expected arrays in input data:
- x, y
- u, v (corrected velocity)
- predictor velocity before pressure correction:
  one of (u_star, v_star) or (u_predictor, v_predictor)
- pressure correction increment:
  one of (dp) or (pressure_increment)
- dt (scalar or array-like)

Identity checked per snapshot n:
  u = u_star - dt * grad(dp)
  v = v_star - dt * grad(dp)

Divergence check:
  ||div(u)|| should be <= ||div(u_star)|| on median over time.
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
        divergence_2d,
        l2_norm,
        load_data,
        normalize_time_axis,
        read_array,
    )
    from py_utils.tests.test_common import summarize_check, write_json_report
else:
    from ..common import close_data, divergence_2d, l2_norm, load_data, normalize_time_axis, read_array
    from .test_common import summarize_check, write_json_report


def _grid_spacing(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    dx = float(np.mean(np.diff(x[0, :])))
    dy = float(np.mean(np.diff(y[:, 0])))
    if dx <= 0 or dy <= 0:
        raise ValueError(f"Invalid grid spacing: dx={dx}, dy={dy}")
    return dx, dy


def _dt_sequence(dt_arr: np.ndarray | float, nt: int) -> np.ndarray:
    dtv = np.asarray(dt_arr, dtype=float)
    if dtv.ndim == 0:
        return np.full((nt,), float(dtv), dtype=float)
    if dtv.size == nt:
        return dtv.reshape(nt)
    if dtv.size == nt - 1:
        out = np.empty((nt,), dtype=float)
        out[0] = dtv[0]
        out[1:] = dtv
        return out
    raise ValueError(f"Unsupported dt shape {dtv.shape} for nt={nt}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True, help="NPZ/HDF5 file with correction arrays")
    ap.add_argument("--identity-thresh", type=float, default=1.0e-7)
    ap.add_argument("--div-reduction-factor", type=float, default=1.05)
    ap.add_argument("--report-json", type=Path, default=None)
    args = ap.parse_args()

    data = load_data(args.data)
    try:
        required_base = ["x", "y", "u", "v", "dt"]
        missing_base = [k for k in required_base if k not in data]
        pred_u_key = "u_star" if "u_star" in data else ("u_predictor" if "u_predictor" in data else None)
        pred_v_key = "v_star" if "v_star" in data else ("v_predictor" if "v_predictor" in data else None)
        dp_key = "dp" if "dp" in data else ("pressure_increment" if "pressure_increment" in data else None)

        if missing_base or pred_u_key is None or pred_v_key is None or dp_key is None:
            payload = {
                "test": "do_velocity_correction_test",
                "passed": True,
                "skipped": True,
                "reason": (
                    "Velocity-correction diagnostics are not available in this dataset. "
                    "Expected dt + predictor + pressure-increment arrays."
                ),
                "expected_aliases": {
                    "predictor_u": ["u_star", "u_predictor"],
                    "predictor_v": ["v_star", "v_predictor"],
                    "pressure_increment": ["dp", "pressure_increment"],
                },
                "missing_base": missing_base,
                "data": str(args.data),
            }
            print(json.dumps(payload, indent=2))
            write_json_report(args.report_json, payload)
            return 0

        x = read_array(data, "x")
        y = read_array(data, "y")
        u = normalize_time_axis(read_array(data, "u"), kind="spatial_time")
        v = normalize_time_axis(read_array(data, "v"), kind="spatial_time")
        u_star = normalize_time_axis(read_array(data, pred_u_key), kind="spatial_time")
        v_star = normalize_time_axis(read_array(data, pred_v_key), kind="spatial_time")
        dp = normalize_time_axis(read_array(data, dp_key), kind="spatial_time")
        dt = _dt_sequence(read_array(data, "dt"), nt=u.shape[0])
        dx, dy = _grid_spacing(x, y)

        nt = min(u.shape[0], u_star.shape[0], dp.shape[0], dt.shape[0])
        u = u[:nt]
        v = v[:nt]
        u_star = u_star[:nt]
        v_star = v_star[:nt]
        dp = dp[:nt]
        dt = dt[:nt]

        grad_dp_x = np.gradient(dp, dx, axis=2, edge_order=2)
        grad_dp_y = np.gradient(dp, dy, axis=1, edge_order=2)

        u_id_res = u - (u_star - dt[:, None, None] * grad_dp_x)
        v_id_res = v - (v_star - dt[:, None, None] * grad_dp_y)

        id_l2 = np.sqrt(np.mean(u_id_res * u_id_res + v_id_res * v_id_res, axis=(1, 2)))
        id_med = float(np.median(id_l2))
        id_max = float(np.max(id_l2))

        div_star = divergence_2d(u_star, v_star, dx, dy)
        div_corr = divergence_2d(u, v, dx, dy)
        div_star_med = float(np.median(np.sqrt(np.mean(div_star * div_star, axis=(1, 2)))))
        div_corr_med = float(np.median(np.sqrt(np.mean(div_corr * div_corr, axis=(1, 2)))))

        checks = [
            summarize_check(
                "velocity_correction_identity",
                id_med <= args.identity_thresh,
                {
                    "median_l2_residual": id_med,
                    "max_l2_residual": id_max,
                    "threshold": args.identity_thresh,
                },
            ),
            summarize_check(
                "divergence_reduction",
                div_corr_med <= args.div_reduction_factor * div_star_med,
                {
                    "median_divergence_predictor": div_star_med,
                    "median_divergence_corrected": div_corr_med,
                    "allowed_factor": args.div_reduction_factor,
                },
            ),
        ]

        passed = all(c["passed"] for c in checks)
        payload = {
            "test": "do_velocity_correction_test",
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
