#!/usr/bin/env python3
"""
MPI scaling study for the DO Velocity Correction Scheme.

By default all rank counts run simultaneously, each pinned to a distinct
CPU set via taskset, so one wall-clock measurement covers the whole sweep.

Usage:
    python3 py_utils/mpi_scaling.py --xml cases/vortex/runs/.../casefile.xml
    python3 py_utils/mpi_scaling.py --xml casefile.xml --steps 50 --ranks 1 2 4 6 8 12
    python3 py_utils/mpi_scaling.py --xml casefile.xml --sequential
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_solver() -> Path:
    env = os.environ.get("NEKTAR_SOLVER")
    if env:
        return Path(env)
    found = shutil.which("IncNavierStokesSolver")
    if found:
        return Path(found)
    return Path(
        "/home/isma/nektar_repro/build/solvers/"
        "IncNavierStokesSolver/IncNavierStokesSolver"
    )


def _available_cpus() -> list[int]:
    try:
        return sorted(os.sched_getaffinity(0))
    except AttributeError:
        return list(range(os.cpu_count() or 1))


def _patch_xml(src: Path, dst: Path, n_steps: int) -> None:
    """Write a patched casefile: override stopping criteria, strip file output.

    Always sets both NumSteps and FinTime so either stopping criterion in
    Nektar limits the run to exactly n_steps steps.
    """
    text = src.read_text(encoding="utf-8")

    # Compute FinTime from TimeStep
    m = re.search(r'<P\b[^>]*>\s*TimeStep\s*=\s*(\S+)\s*</P>',
                  text, re.IGNORECASE)
    if not m:
        sys.exit(f"ERROR: TimeStep not found in {src.name}")
    fin_time = round(n_steps * float(m.group(1)), 10)

    # Patch FinTime; insert after TimeStep line if absent
    text, n_ft = re.subn(
        r'(<P\b[^>]*>\s*FinTime\s*=\s*)\S+(\s*</P>)',
        rf'\g<1>{fin_time}\g<2>',
        text, flags=re.IGNORECASE,
    )
    if n_ft == 0:
        text = re.sub(
            r'(<P\b[^>]*>\s*TimeStep\s*=\s*\S+\s*</P>)',
            rf'\1\n      <P> FinTime = {fin_time} </P>',
            text, count=1, flags=re.IGNORECASE,
        )

    # Patch NumSteps; insert after FinTime line if absent
    text, n_ns = re.subn(
        r'(<P\b[^>]*>\s*NumSteps\s*=\s*)\S+(\s*</P>)',
        rf'\g<1>{n_steps}\g<2>',
        text, flags=re.IGNORECASE,
    )
    if n_ns == 0:
        text = re.sub(
            r'(<P\b[^>]*>\s*FinTime\s*=\s*\S+\s*</P>)',
            rf'\1\n      <P> NumSteps = {n_steps} </P>',
            text, count=1, flags=re.IGNORECASE,
        )

    text = re.sub(
        r'(<FILTERS\b[^>]*>).*?(</FILTERS>)',
        r'\1\2',
        text, flags=re.DOTALL | re.IGNORECASE,
    )
    dst.write_text(text, encoding="utf-8")


def _make_run_dir(case_dir: Path, xml_name: str,
                  root: Path, tag: str, n_steps: int) -> Path:
    """
    Create an isolated run directory:
      - symlink everything in case_dir except the casefile and output dirs
      - write the patched casefile
      - create a fresh empty output/ dir
    casefile_xml/ is excluded so Nektar re-partitions into the temp dir.
    """
    run_dir = root / tag
    run_dir.mkdir()
    skip = {xml_name, "output", "casefile_xml"}
    for item in case_dir.iterdir():
        if item.name not in skip:
            (run_dir / item.name).symlink_to(item.resolve())
    _patch_xml(case_dir / xml_name, run_dir / xml_name, n_steps)
    (run_dir / "output").mkdir()
    return run_dir


def _reset_output(run_dir: Path) -> None:
    out = run_dir / "output"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()


def _build_cmd(solver: Path, run_dir: Path, xml_name: str,
               n: int, mpirun: str,
               cpu_ids: list[int] | None) -> list[str]:
    patched_xml = run_dir / xml_name
    geom = run_dir / "geometry.xml"
    solver_args = (
        [str(geom), str(patched_xml)] if geom.is_file()
        else [str(patched_xml)]
    )
    mpi_part = [mpirun, "-n", str(n), str(solver)] + solver_args
    if cpu_ids and shutil.which("taskset"):
        cpu_str = ",".join(str(c) for c in cpu_ids)
        return ["taskset", "-c", cpu_str] + mpi_part
    return mpi_part


# ---------------------------------------------------------------------------
# Run strategies
# ---------------------------------------------------------------------------

def _run_round_parallel(
    ranks: list[int],
    run_dirs: dict[int, Path],
    xml_name: str,
    solver: Path,
    mpirun: str,
    timeout: int,
    available_cpus: list[int],
) -> dict[int, float]:
    """Launch all ranks simultaneously on non-overlapping CPU slices."""
    total_needed = sum(ranks)
    if total_needed > len(available_cpus):
        print(
            f"  WARNING: need {total_needed} CPUs but only "
            f"{len(available_cpus)} available — runs will share CPUs "
            f"(taskset disabled).",
            file=sys.stderr,
        )
        cpu_assignment: dict[int, list[int] | None] = {n: None for n in ranks}
    else:
        cpu_assignment = {}
        offset = 0
        for n in ranks:
            cpu_assignment[n] = available_cpus[offset:offset + n]
            offset += n

    procs: dict[int, subprocess.Popen] = {}
    t0s: dict[int, float] = {}

    for n in ranks:
        _reset_output(run_dirs[n])
        cmd = _build_cmd(solver, run_dirs[n], xml_name, n, mpirun,
                         cpu_assignment[n])
        t0s[n] = time.perf_counter()
        procs[n] = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    results: dict[int, float] = {}
    for n, proc in procs.items():
        try:
            _, stderr_bytes = proc.communicate(timeout=timeout)
            elapsed = time.perf_counter() - t0s[n]
            if proc.returncode == 0:
                results[n] = elapsed
                print(f"  ranks={n:3d}  {elapsed:.2f}s")
            else:
                print(f"  ranks={n:3d}  FAILED (rc={proc.returncode})")
                if stderr_bytes:
                    tail = stderr_bytes.decode(errors="replace").strip()
                    for line in tail.splitlines()[-10:]:
                        print(f"    {line}")
        except subprocess.TimeoutExpired:
            proc.kill()
            print(f"  ranks={n:3d}  TIMEOUT")
    return results


def _run_round_sequential(
    ranks: list[int],
    run_dirs: dict[int, Path],
    xml_name: str,
    solver: Path,
    mpirun: str,
    timeout: int,
) -> dict[int, float]:
    results: dict[int, float] = {}
    for n in ranks:
        _reset_output(run_dirs[n])
        cmd = _build_cmd(solver, run_dirs[n], xml_name, n, mpirun, None)
        t0 = time.perf_counter()
        try:
            r = subprocess.run(
                cmd, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
            elapsed = time.perf_counter() - t0
            if r.returncode == 0:
                results[n] = elapsed
                print(f"  ranks={n:3d}  {elapsed:.2f}s")
            else:
                print(f"  ranks={n:3d}  FAILED (rc={r.returncode})")
                if r.stderr:
                    tail = r.stderr.decode(errors="replace").strip()
                    for line in tail.splitlines()[-10:]:
                        print(f"    {line}")
        except subprocess.TimeoutExpired:
            print(f"  ranks={n:3d}  TIMEOUT")
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--xml",        type=Path, required=True,
                    help="Case casefile.xml")
    ap.add_argument("--solver",     type=Path, default=None,
                    help="IncNavierStokesSolver binary (default: auto-detect)")
    ap.add_argument("--steps",      type=int, default=100,
                    help="Steps per run (default: 100)")
    ap.add_argument("--ranks",      type=int, nargs="+", default=[1, 2, 4, 6, 8],
                    help="MPI rank counts to sweep (default: 1 2 4 6 8)")
    ap.add_argument("--repeats",    type=int, default=1,
                    help="Rounds to run; best time per rank is kept "
                         "(default: 1)")
    ap.add_argument("--mpirun",     default="mpirun",
                    help="MPI launcher (default: mpirun)")
    ap.add_argument("--timeout",    type=int, default=600,
                    help="Per-run timeout in seconds (default: 600)")
    ap.add_argument("--sequential", action="store_true",
                    help="Run rank counts one at a time instead of in parallel")
    args = ap.parse_args()

    xml = args.xml.resolve()
    if not xml.is_file():
        sys.exit(f"ERROR: casefile not found: {xml}")

    solver = (args.solver or _default_solver()).resolve()
    if not solver.is_file():
        sys.exit(f"ERROR: solver not found: {solver}")

    ranks = sorted(set(args.ranks))
    available_cpus = _available_cpus()
    parallel = not args.sequential

    tmp_root = Path(tempfile.mkdtemp(prefix="do_scaling_"))
    print(f"Temp dir  : {tmp_root}")
    print(f"Solver    : {solver}")
    print(f"Casefile  : {xml}")
    print(f"Steps     : {args.steps}")
    print(f"Ranks     : {ranks}")
    print(f"CPUs avail: {len(available_cpus)}")
    print(f"Mode      : {'parallel' if parallel else 'sequential'}")
    print(f"Repeats   : {args.repeats}")
    print()

    run_dirs = {
        n: _make_run_dir(xml.parent, xml.name, tmp_root, f"run_{n}", args.steps)
        for n in ranks
    }

    best: dict[int, float] = {}

    for rep in range(args.repeats):
        if args.repeats > 1:
            print(f"-- round {rep + 1}/{args.repeats} --")
        if parallel:
            results = _run_round_parallel(
                ranks, run_dirs, xml.name, solver,
                args.mpirun, args.timeout, available_cpus,
            )
        else:
            results = _run_round_sequential(
                ranks, run_dirs, xml.name, solver,
                args.mpirun, args.timeout,
            )
        for n, t in results.items():
            if n not in best or t < best[n]:
                best[n] = t

    shutil.rmtree(tmp_root, ignore_errors=True)

    if not best:
        sys.exit("No successful runs.")

    t1 = best.get(min(ranks), min(best.values()))
    print()
    print(f"{'Ranks':>6}  {'Total (s)':>10}  {'s/step':>8}  "
          f"{'Speedup':>8}  {'Efficiency':>11}")
    print("-" * 52)
    for n in sorted(best):
        t   = best[n]
        sp  = t1 / t
        eff = sp / n * 100.0
        print(f"{n:6d}  {t:10.2f}  {t / args.steps:8.3f}  "
              f"{sp:8.2f}x  {eff:10.1f}%")


if __name__ == "__main__":
    main()
