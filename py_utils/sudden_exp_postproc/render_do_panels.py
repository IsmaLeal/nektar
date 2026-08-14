#!/usr/bin/env python3
"""Render the classical do_panels_u.mp4 for a DO run from scratch.

The grid_cache/gd_STEP.dat -> out_grid.h5 -> do_panels.mp4 pipeline used for
earlier runs (e.g. 2026_07_15_do_clean_dnspod_sig0p0144_t250) relied on a
FieldConvert interppoints loop that was never saved as a script. This
reconstructs it, verified against that run's own grid_cache: same box grid
(901 x 121, x in [-5,40], y in [-1.5,1.5]), same header/row layout in the
.dat files (checked byte-for-byte against gd_000000.dat for line count and
column order for a S=2 test case).

FieldConvert module: interppoints (eCreatePts) -- must be called with NO
positional xml/fld inputs (only fromxml=/fromfld= config keys), else it
aborts with "should not use xml or fld inputs" (utilities/FieldConvert/
FieldConvert.cpp:731-750).

Requires the ASan-instrumented build at
/home/isma/nektar_repro/build/dist/bin/FieldConvert (the only FieldConvert
binary currently built in this tree); LD_PRELOAD/LD_LIBRARY_PATH set up to
match. Exit codes from this binary are unreliable under ASan teardown even
on success, so completion is verified by checking the output .dat file's
row count instead of the process return code.

Incremental like assemble_do_h5.py's own doc comment describes: existing
gd_STEP.dat files are skipped.

Usage:
    python3 render_do_panels.py RUN_DIR [--field u] [--particle N]
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path

FC = "/home/isma/nektar_repro/build/dist/bin/FieldConvert"
NX, NY = 901, 121
XMIN, XMAX, YMIN, YMAX = -5, 40, -1.5, 1.5
EXPECTED_LINES = NX * NY + 3  # Variables line + blank + Zone line + data

HERE = Path(__file__).resolve().parent


def _fc_env():
    lib_dirs = set()
    for root in ("/home/isma/nektar_repro/build/library",
                 "/home/isma/nektar_repro/build/solvers",
                 "/home/isma/nektar_repro/build/ThirdParty/dist"):
        for so in Path(root).rglob("*.so*"):
            lib_dirs.add(str(so.parent))
    env = os.environ.copy()
    env["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libasan.so.8"
    env["LD_LIBRARY_PATH"] = ":".join(sorted(lib_dirs))
    env["ASAN_OPTIONS"] = "detect_leaks=0"
    return env


def casefile_param(run: Path, name: str, default=None):
    txt = (run / "casefile.xml").read_text()
    m = re.search(rf"{name}\s*=\s*([0-9eE.+-/]+)", txt)
    if not m:
        return default
    return eval(m.group(1), {"__builtins__": {}})


def interp_one(run: Path, fld: Path, out_dat: Path, env: dict) -> bool:
    box = f"box={NX},{NY},1,{XMIN},{XMAX},{YMIN},{YMAX},0,0"
    cmd = [FC, "-m",
          f"interppoints:fromxml=geometry.xml:fromfld={fld.relative_to(run)}:{box}",
          str(out_dat)]
    subprocess.run(cmd, cwd=run, env=env, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    if not out_dat.exists():
        return False
    with open(out_dat) as f:
        n = sum(1 for _ in f)
    return n == EXPECTED_LINES


def build_grid_cache(run: Path) -> int:
    cache = run / "output/py_utils_results/grid_cache"
    cache.mkdir(parents=True, exist_ok=True)
    archives = sorted((run / "output/do").glob("casefile.do_*.fld"))
    env = _fc_env()
    done = 0
    for fld in archives:
        step = fld.name.split("do_")[1].split(".")[0]
        out_dat = cache / f"gd_{step}.dat"
        if out_dat.exists():
            with open(out_dat) as f:
                if sum(1 for _ in f) == EXPECTED_LINES:
                    done += 1
                    continue
            out_dat.unlink()
        ok = interp_one(run, fld, out_dat, env)
        if not ok:
            print(f"  FAILED: {fld.name}")
            if out_dat.exists():
                out_dat.unlink()
            continue
        done += 1
        if done % 20 == 0:
            print(f"  {done}/{len(archives)} archives interpolated", flush=True)
    print(f"grid_cache: {done}/{len(archives)} archives OK")
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--field", default="u", choices=["u", "v", "vorticity", "velocity"])
    ap.add_argument("--particle", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    run = args.run.resolve()

    n = build_grid_cache(run)
    if n == 0:
        print("no archives interpolated; aborting")
        return 1

    h5 = run / "output/out_grid.h5"
    rc = subprocess.run(
        ["python3", str(HERE / "assemble_do_h5.py"), str(run), "--out", str(h5)]
    ).returncode
    if rc != 0:
        return rc

    out = args.out or (run / "output/py_utils_results" /
                        (f"hopper_particle{args.particle}_{args.field}.mp4"
                         if args.particle is not None else
                         f"do_panels_{args.field}.mp4"))
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["python3",
          str(HERE.parent / "reconstruction/animation/animate_do_panels.py"),
          "--data", str(h5), "--field", args.field, "--out", str(out)]
    if args.particle is not None:
        cmd += ["--particle", str(args.particle)]
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    raise SystemExit(main())
