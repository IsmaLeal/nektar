"""Reading a DO run's archives for the do_panels video, on any domain: the
FieldConvert extraction onto points, its .dat output, the Yi coefficients kept
in the archive metadata, and the parameters of the run's casefile.xml. Copied
from sudden_exp_postproc/se.py and cases/se/templates/extract_lib.py, which
belong to the sudden-expansion channel."""
import os
import re
import subprocess
import sys
import time
import glob
from pathlib import Path

import numpy as np

FIELDCONVERT = os.environ.get('NEKTAR_FIELDCONVERT', '/home/isma/nektar_repro/build/utilities/FieldConvert/FieldConvert')


def fc_env():
    """Environment for FIELDCONVERT: on this laptop it exists only in the ASan
    build, so libasan is preloaded and leak detection is off."""
    env = os.environ.copy()
    try:
        asan = subprocess.run(['gcc', '-print-file-name=libasan.so'], capture_output=True, text=True).stdout.strip()
        if '/' in asan:
            env['LD_PRELOAD'] = asan
    except OSError:
        pass
    env['ASAN_OPTIONS'] = 'detect_leaks=0'
    return env


def interp(xml, fld, points, out, fieldconvert=FIELDCONVERT):
    """FieldConvert interppoints of the field file fld (on the mesh xml) at
    points ('box=...', 'plane=...', 'line=...'), written to the .dat out.
    Cached: nothing runs when out exists. Returns out."""
    out = Path(out)
    if not out.exists():
        subprocess.run([fieldconvert, '-f', '-m', f'interppoints:fromxml={xml}:fromfld={fld}:{points}', str(out)],
                       env=fc_env(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out


def load_dat(path, ncol):
    """Rows of a FieldConvert .dat with at least ncol numbers, first ncol columns (header lines skipped)."""
    rows = []
    for ln in Path(path).read_text().splitlines():
        p = ln.split()
        if len(p) >= ncol:
            try:
                rows.append([float(v) for v in p[:ncol]])
            except ValueError:
                pass
    return np.array(rows)


def casefile_param(run, name, default=None):
    """Value of the parameter name = ... in run/casefile.xml (a float; expressions like 1/100 are evaluated)."""
    txt = (Path(run) / 'casefile.xml').read_text()
    m = re.search(rf'{name}\s*=\s*([0-9eE.+-/]+)', txt)
    if not m:
        return default
    try:
        return float(eval(m.group(1), {'__builtins__': {}}))
    except Exception:
        return default


_YI_TAG = re.compile(r"<DOVelocityCorrectionScheme_Yi_hex>([0-9a-fA-F]+)<")


def yi_from_fld(path, S):
    """(Np, S) Yi matrix from the hex metadata of a Field-format archive.

    path: the archive (directory with Info.xml for MPI runs, or the single
    .fld file of a serial run). Identical in every partition file.
    """
    p = Path(path)
    info = p / "Info.xml" if p.is_dir() else p
    m = _YI_TAG.search(info.read_text())
    if m is None:
        raise ValueError(f"no Yi_hex metadata in {info}")
    flat = np.frombuffer(bytes.fromhex(m.group(1)), dtype="<f8")
    if flat.size % S:
        raise ValueError(f"{info}: {flat.size} doubles not divisible by S={S}")
    return flat.reshape(-1, S)


def yi_series(do_dir, nmodes, pattern='*.do_*.fld', dt=0.01):
    """t (T,) = step * dt and yi (T, Np, S) from every DO archive of a directory."""
    files = sorted(glob.glob(str(Path(do_dir) / pattern)))
    if not files:
        raise FileNotFoundError(f'no archives matching {pattern} in {do_dir}')
    steps = np.array([int(re.search(r'do_(\d+)', f).group(1)) for f in files])
    yi = np.stack([yi_from_fld(f, nmodes) for f in files])
    return steps * dt, yi


def progress(done, total, t0, what):
    """one line, rewritten in place: [#####.....] done/total what, elapsed s, estimated remaining s"""
    el = time.time() - t0
    left = el / done * (total - done) if done else 0.0
    n = 30
    bar = "#" * int(n * done / total) + "." * (n - int(n * done / total))
    sys.stdout.write(f"\r[{bar}] {done}/{total} {what}, {el:5.0f} s, {left:4.0f} s left")
    sys.stdout.flush()
    if done == total:
        sys.stdout.write("\n")
