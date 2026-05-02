#!/usr/bin/env python3
"""MPI POD-init smoke test for DOVelocityCorrectionScheme.

Guards against the regression where DOPODInitialiser::RecomputeYiByProjection
left Nektar Array<OneD,T> shared-storage state in a configuration that NaN'd
the first time step's iterative CG on n=2/n=4 MPI ranks. Without
DOVelocityCorrectionScheme::CallRecomputeYiWithStorageGuard() at the call site, this test fails.

What this test does:
  1. Take a working DOVelocityCorrectionScheme POD-init case (default: the cylinder repro under
     cases/cylinder_flow/runs/pod_init_test).
  2. Stage a temporary copy with:
       - FinTime shortened (default 0.05 = 50 timesteps),
       - stochastic forcing disabled (DOForcingNumChannels=0) so the unrelated
         "forcing channel ... has zero mass-norm" race can't masquerade as a
         failure.
  3. Run the solver under mpirun at each requested rank count.
  4. Pass iff: rc==0 AND no "= nan" / "= -nan" / "FATAL" lines in the run log
     AND the per-step CFL printed in the log stays finite.

Default rank counts: 2 and 3. n=4 is intentionally NOT tested by default
because of an independent forcing-init race that flakes ~1/3 of the time
even on a known-good build.

Usage:
  python3 py_utils/tests/mpi_pod_init_smoke_test.py
  python3 py_utils/tests/mpi_pod_init_smoke_test.py --case-dir <path> --np 2 3
  python3 py_utils/tests/mpi_pod_init_smoke_test.py --repeats 5

Exit code 0 iff all (rank-count, repeat) cells pass.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_CASE_DIR = Path(
    "/home/isma/nektar_src_full/cases/cylinder_flow/runs/pod_init_test"
)
DEFAULT_SOLVER = Path(
    "/home/isma/nektar_repro/build/solvers/IncNavierStokesSolver/IncNavierStokesSolver"
)


def _stage_case(src_dir: Path, dst_dir: Path, fin_time: float) -> None:
    """Copy casefile.xml + combined.xml, symlink mesh+snapshots, set
    FinTime and disable forcing channels."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    (dst_dir / "output").mkdir(exist_ok=True)
    for name in ("casefile.xml", "combined.xml"):
        text = (src_dir / name).read_text()
        text = re.sub(r"<P>\s*FinTime\s*=\s*[^<]*</P>",
                      f"<P> FinTime = {fin_time} </P>", text)
        text = re.sub(r"<P>\s*DOForcingNumChannels\s*=\s*[^<]*</P>",
                      "<P> DOForcingNumChannels = 0 </P>", text)
        (dst_dir / name).write_text(text)
    for name in ("mesh", "snapshots"):
        target = src_dir / name
        if target.exists() and not (dst_dir / name).exists():
            (dst_dir / name).symlink_to(target.resolve())


def _is_finite_token(tok: str) -> bool:
    try:
        v = float(tok)
        return v == v and abs(v) < float("inf")
    except ValueError:
        return False


_NAN_RE = re.compile(r"=\s*-?nan|FATAL", re.IGNORECASE)
_CFL_RE = re.compile(r"CFL:\s*([0-9eE.+\-nNaA]+)")


def _check_log(log_path: Path) -> tuple[bool, str]:
    text = log_path.read_text(errors="replace")
    if _NAN_RE.search(text):
        first = next((ln for ln in text.splitlines() if _NAN_RE.search(ln)), "?")
        return False, f"NaN/FATAL marker: {first.strip()[:160]}"
    cfl_vals = _CFL_RE.findall(text)
    if cfl_vals and not all(_is_finite_token(v) for v in cfl_vals):
        bad = [v for v in cfl_vals if not _is_finite_token(v)][:3]
        return False, f"non-finite CFL value(s): {bad}"
    return True, ""


def _run_one(np_count: int, case_dir: Path, solver: Path) -> tuple[bool, str]:
    log = case_dir / "run.log"
    if log.exists():
        log.unlink()
    out = case_dir / "output"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    if np_count == 1:
        cmd = [str(solver), "-i", "Hdf5", "casefile.xml", "combined.xml"]
    else:
        cmd = ["mpirun", "--oversubscribe", "-np", str(np_count),
               str(solver), "-i", "Hdf5", "casefile.xml", "combined.xml"]
    proc = subprocess.run(cmd, cwd=case_dir,
                          stdout=open(log, "wb"), stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        msg = "?"
        try:
            tail = log.read_text(errors="replace").splitlines()[-30:]
            msg = "; ".join(ln.strip() for ln in tail if ln.strip())[:300]
        except Exception:
            pass
        return False, f"rc={proc.returncode}: {msg}"
    return _check_log(log)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR,
                    help="Source DOVelocityCorrectionScheme POD-init case directory.")
    ap.add_argument("--solver", type=Path, default=DEFAULT_SOLVER,
                    help="IncNavierStokesSolver executable.")
    ap.add_argument("--np", type=int, nargs="+", default=[2, 3],
                    help="MPI rank counts to test (default: 2 3).")
    ap.add_argument("--repeats", type=int, default=3,
                    help="Repetitions per rank count (default: 3).")
    ap.add_argument("--fin-time", type=float, default=0.05,
                    help="FinTime override (default: 0.05 → 50 timesteps).")
    args = ap.parse_args()

    if not args.case_dir.is_dir():
        print(f"[skip] case dir not found: {args.case_dir}")
        return 0
    if not args.solver.is_file():
        print(f"[skip] solver not found: {args.solver}")
        return 0

    failures = 0
    with tempfile.TemporaryDirectory(prefix="dovcs_mpi_smoke_") as tmp:
        staged = Path(tmp) / "case"
        _stage_case(args.case_dir, staged, args.fin_time)
        for n in args.np:
            for r in range(1, args.repeats + 1):
                ok, why = _run_one(n, staged, args.solver)
                tag = "PASS" if ok else "FAIL"
                detail = f" — {why}" if why else ""
                print(f"[{tag}] np={n} repeat={r}{detail}")
                if not ok:
                    failures += 1
    summary = "OK" if failures == 0 else f"FAILED ({failures} cells)"
    print(f"\n=== mpi_pod_init_smoke_test: {summary} ===")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
