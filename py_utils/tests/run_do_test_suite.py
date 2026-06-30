#!/usr/bin/env python3
"""Run the full DO validation suite from py_utils/tests."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from py_utils.common import close_data, load_data, read_array
else:
    from ..common import close_data, load_data, read_array


SCRIPT_ORDER = [
    "do_incompressible_ns_test.py",
    "do_orthogonality_test.py",
    "do_velocity_correction_test.py",
]


def _raw_do_preflight(data_path: Path, allow_missing_raw: bool) -> dict[str, object]:
    data = load_data(data_path)
    try:
        required = ["archive_diag_cov", "archive_ortho_err", "yi", "t_do"]
        missing = [k for k in required if k not in data]
        if missing:
            passed = bool(allow_missing_raw)
            msg = (
                "missing raw DO datasets "
                f"{missing}; expected archive-native diagnostics in canonical out.h5"
            )
            return {
                "name": "raw_do_datasets_present",
                "passed": passed,
                "required": required,
                "missing": missing,
                "message": msg,
                "data_basis": "raw_archive",
            }

        diag = np.asarray(read_array(data, "archive_diag_cov"), dtype=float)
        ortho = np.asarray(read_array(data, "archive_ortho_err"), dtype=float).reshape(-1)
        yi = np.asarray(read_array(data, "yi"), dtype=float)
        t_do = np.asarray(read_array(data, "t_do"), dtype=float).reshape(-1)

        finite_ok = bool(
            np.all(np.isfinite(diag))
            and np.all(np.isfinite(ortho))
            and np.all(np.isfinite(yi))
            and np.all(np.isfinite(t_do))
        )
        if diag.ndim == 2:
            min_diag = float(np.min(diag))
        else:
            min_diag = float(np.min(diag.reshape(-1)))
        nonneg_ok = min_diag >= -1.0e-12
        shape_ok = yi.ndim == 3 and t_do.size > 0
        passed = finite_ok and nonneg_ok and shape_ok

        return {
            "name": "raw_do_datasets_present",
            "passed": bool(passed),
            "required": required,
            "missing": [],
            "checks": {
                "finite_values": finite_ok,
                "diag_cov_min": min_diag,
                "diag_cov_nonnegative_tol": -1.0e-12,
                "yi_shape": list(yi.shape),
                "t_do_size": int(t_do.size),
                "shape_ok": shape_ok,
            },
            "data_basis": "raw_archive",
        }
    finally:
        close_data(data)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True, help="NPZ/HDF5 data file")
    ap.add_argument("--out-dir", type=Path, default=Path("py_utils/tests/results"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument(
        "--allow-missing-raw",
        action="store_true",
        help="Do not fail suite when raw archive DO diagnostics are missing.",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "data": str(args.data),
        "data_policy": {
            "hard_gate": "raw_archive",
            "diagnostic": "interpolated_grid",
        },
        "results": [],
    }
    all_ok = True

    preflight = _raw_do_preflight(args.data, allow_missing_raw=args.allow_missing_raw)
    summary["preflight"] = preflight
    if preflight["passed"]:
        print("[PASS] raw_do_datasets_present")
    else:
        print("[FAIL] raw_do_datasets_present")
        all_ok = False

    for script in SCRIPT_ORDER:
        cmd = [
            args.python,
            str(root / script),
            "--data",
            str(args.data),
            "--report-json",
            str(args.out_dir / f"{Path(script).stem}.json"),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        rc = int(proc.returncode)
        passed = rc == 0
        if rc not in {0, 2}:
            passed = False

        summary["results"].append(
            {
                "script": script,
                "returncode": rc,
                "passed": passed,
                "stdout_tail": "\n".join(proc.stdout.strip().splitlines()[-20:]),
                "stderr_tail": "\n".join(proc.stderr.strip().splitlines()[-20:]),
                "data_basis": (
                    "raw_archive+interpolated_grid"
                    if script == "do_orthogonality_test.py"
                    else "interpolated_grid"
                ),
            }
        )
        all_ok = all_ok and passed

        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {script} (rc={rc})")

    summary["passed"] = all_ok
    summary_path = args.out_dir / "suite_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Suite summary: {summary_path}")
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
