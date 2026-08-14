#!/usr/bin/env python3
"""Quick-look readers for DO/DNS runs that need no FieldConvert extraction.

Two data sources are readable directly:

1. Field-format DO archives (output/do/casefile.do_NNNNNN.fld): the archive
   filter stores the gathered (nParticles x nModes) Yi matrix as a hex string
   of little-endian doubles in the metadata tag
   <DOVelocityCorrectionScheme_Yi_hex>, identical in every partition file.
   yi_from_fld/yi_series parse it with one regex per archive.

2. HistoryPoints files (output/asymmetry/axis_v.his): one row per point per
   sample, columns t u v p, one block of nPoints rows per output time. For
   the se axis line, point i sits at x = 0.1*i, y = 0.

Usage:
    from do_quicklook import yi_series, read_his
    t, yi = yi_series("output/do", nmodes=6)     # yi: (T, nParticles, nModes)
    t, pts = read_his("output/asymmetry/axis_v.his")   # pts: (T, nPts, 3)
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np

_YI_TAG = re.compile(r"<DOVelocityCorrectionScheme_Yi_hex>([0-9a-fA-F]+)<")


def yi_from_fld(fld_path: str | Path, nmodes: int) -> np.ndarray:
    """Yi matrix (nParticles, nModes) from one Field-format DO archive."""
    fld = Path(fld_path)
    info = fld / "Info.xml" if fld.is_dir() else fld
    m = _YI_TAG.search(info.read_text())
    if m is None:
        raise ValueError(f"no Yi_hex metadata in {info}")
    flat = np.frombuffer(bytes.fromhex(m.group(1)), dtype="<f8")
    if flat.size % nmodes:
        raise ValueError(f"{info}: {flat.size} doubles not divisible by S={nmodes}")
    return flat.reshape(-1, nmodes)


def yi_series(do_dir: str | Path, nmodes: int,
              pattern: str = "*.do_*.fld",
              dt: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Stack Yi over all archives in a directory.

    Returns (t, yi) with t in time units (step index * dt) and
    yi of shape (T, nParticles, nModes).
    """
    files = sorted(glob.glob(str(Path(do_dir) / pattern)))
    if not files:
        raise FileNotFoundError(f"no archives matching {pattern} in {do_dir}")
    steps = np.array([int(re.search(r"do_(\d+)", f).group(1)) for f in files])
    yi = np.stack([yi_from_fld(f, nmodes) for f in files])
    return steps * dt, yi


def read_his(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """HistoryPoints file -> (t, data) with data shape (T, nPoints, 3) = u,v,p.

    The number of points is taken from the '#' header lines.
    """
    npts = 0
    rows = []
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            npts += 1 if s[1:].strip() and s[1:].split()[0].isdigit() else 0
            continue
        rows.append([float(v) for v in s.split()])
    arr = np.asarray(rows)
    nblocks = arr.shape[0] // npts
    arr = arr[: nblocks * npts].reshape(nblocks, npts, arr.shape[1])
    return arr[:, 0, 0], arr[:, :, 1:]


if __name__ == "__main__":
    import sys
    t, yi = yi_series(sys.argv[1], nmodes=int(sys.argv[2]))
    print(f"{yi.shape[0]} snapshots, {yi.shape[1]} particles, S={yi.shape[2]}")
    print(f"t = {t[0]:g} .. {t[-1]:g}")
    print("Var(Yi) at t end:", np.var(yi[-1], axis=0))


# ---------------------------------------------------------------------------
# Per-particle reconstruction on the symmetry axis (shared by plotting tools)
# ---------------------------------------------------------------------------
import os as _os
import subprocess as _subprocess

FC_DEFAULT = _os.environ.get(
    "NEKTAR_FIELDCONVERT",
    "/home/isma/nektar_repro/build/utilities/FieldConvert/FieldConvert")


def _fc_env():
    env = _os.environ.copy()
    try:
        asan = _subprocess.run(["gcc", "-print-file-name=libasan.so"],
                               capture_output=True, text=True).stdout.strip()
        if "/" in asan:
            env["LD_PRELOAD"] = asan
    except OSError:
        pass
    env["ASAN_OPTIONS"] = "detect_leaks=0"
    return env


def casefile_param(run, name, default=None):
    txt = (Path(run) / "casefile.xml").read_text()
    m = re.search(rf"{name}\s*=\s*([0-9eE.+-/]+)", txt)
    if not m:
        return default
    try:
        return float(eval(m.group(1), {"__builtins__": {}}))
    except Exception:
        return default


def mp_series(run, xsensor=2.0, fieldconvert=FC_DEFAULT):
    """Per-particle asymmetry series for a DO run.

    Line-interpolates every archive onto the symmetry axis (cached in
    run/output/py_utils_results/axis_cache, incremental), reads Yi from the
    archive metadata, and reconstructs
        M_p(t) = sign(v_p(xsensor, 0)) * sqrt(int v_p^2 dx)
    for every particle. Returns (t, yi, Mp, x):
        t  (T,), yi (T, P, S), Mp (T, P), x (nPts,)
    """
    run = Path(run)
    S = int(casefile_param(run, "DOModes"))
    dt = casefile_param(run, "TimeStep", 0.01)
    cache = run / "output/py_utils_results/axis_cache"
    cache.mkdir(parents=True, exist_ok=True)
    geom = run / "geometry.xml"
    env = _fc_env()
    for fld in sorted(glob.glob(str(run / "output/do/*.do_*.fld"))):
        step = re.search(r"do_(\d+)", fld).group(1)
        out = cache / f"ax_{step}.dat"
        if out.exists():
            continue
        _subprocess.run(
            [fieldconvert, "-f", "-m",
             f"interppoints:fromxml={geom}:fromfld={fld}:line=401,0,0,40,0",
             str(out)], env=env,
            stdout=_subprocess.DEVNULL, stderr=_subprocess.DEVNULL)

    ncol = 5 + 2 * S  # line layout: x y | u v p | (mode_i_u, mode_i_v)*S
    va, ma, x = {}, {}, None
    for f in sorted(cache.glob("ax_*.dat")):
        st = int(re.search(r"ax_(\d+)", f.name).group(1))
        rows = []
        for ln in f.read_text().splitlines():
            p = ln.split()
            if len(p) >= ncol:
                try:
                    rows.append([float(v) for v in p[:ncol]])
                except ValueError:
                    pass
        d = np.array(rows)
        if d.size == 0:
            continue
        if x is None:
            x = d[:, 0]
        va[st] = d[:, 3]
        ma[st] = np.stack([d[:, 6 + 2 * i] for i in range(S)])

    t_yi, yi = yi_series(str(run / "output/do"), nmodes=S, dt=dt)
    steps_yi = (np.round(t_yi / dt)).astype(int)
    ks = [k for k, st in enumerate(steps_yi) if st in va]
    isen = int(round(xsensor / (x[1] - x[0])))
    trapz = getattr(np, "trapezoid", None) or np.trapz
    Mp = np.array([
        np.sign((va[steps_yi[k]][None, :] + yi[k] @ ma[steps_yi[k]])[:, isen])
        * np.sqrt(trapz((va[steps_yi[k]][None, :]
                         + yi[k] @ ma[steps_yi[k]])**2, x, axis=1))
        for k in ks])
    return t_yi[ks], yi[ks], Mp, x


def mp_series_chain(segments, xsensor=2.0, fieldconvert=FC_DEFAULT):
    """Per-particle asymmetry series across a restart chain.

    segments: list of (run_dir, t_offset) in chronological order, e.g.
        [(parent, 0.0), (continuation, 250.0)].
    Times are shifted by each segment's offset; at every seam the
    duplicated sample (the continuation's t=0 archive equals the
    parent's final archive) is dropped from the LATER segment. Raw Yi
    are gauge-rotated across a restart, so only gauge-invariant
    quantities (Mp, reconstructed fields) are continuous; the returned
    yi is the per-segment raw coefficient array, concatenated for
    convenience but NOT seam-comparable.
    """
    ts, yis, mps = [], [], []
    x = None
    for run, off in segments:
        t, yi, Mp, x = mp_series(run, xsensor=xsensor,
                                 fieldconvert=fieldconvert)
        t = t + off
        if ts and len(t) and abs(t[0] - ts[-1][-1]) < 1e-9:
            t, yi, Mp = t[1:], yi[1:], Mp[1:]
        ts.append(t); yis.append(yi); mps.append(Mp)
    import numpy as _np
    return (_np.concatenate(ts), _np.concatenate(yis),
            _np.concatenate(mps), x)
