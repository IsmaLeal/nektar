#!/usr/bin/env python3
"""Interpolate a Nektar++ field file onto the 901 x 121 extraction plane of
cases/se/templates/extract_lib.py and store it as an .npz (one key per field
variable, plus 'source'), the format plot_plane.py reads.

FieldConvert runs interppoints with the BOX of extract_lib on the geometry
given (the mesh of the run), with the binary and environment of se.py
(FIELDCONVERT, fc_env: the ASan build, libasan preloaded).

Example (leading adjoint eigenmode of the Re 79.3 Arnoldi run):
  python3 plane_from_fld.py cases/se/runs/2026_07_12_m03_Re79p3_arnoldi_adjoint/geometry_eig_0.fld \
      --geometry cases/se/runs/2026_07_12_m03_Re79p3_arnoldi_adjoint/geometry.xml \
      --out cases/se/templates/qdagger_Re79p3.npz
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from se import BOX, FIELDCONVERT, fc_env, read_dat

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("fld", type=Path)
ap.add_argument("--geometry", type=Path, required=True, help="geometry.xml of the mesh the field lives on")
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--fieldconvert", default=FIELDCONVERT)
args = ap.parse_args()

with tempfile.TemporaryDirectory() as tmp:
    dat = Path(tmp) / "plane.dat"
    cmd = [args.fieldconvert, "-f", "-m",
           f"interppoints:fromxml={args.geometry.resolve()}:fromfld={args.fld.resolve()}:{BOX}", str(dat)]
    r = subprocess.run(cmd, env=fc_env(), capture_output=True, text=True)
    if r.returncode != 0 or not dat.exists():
        sys.exit(f"FieldConvert failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    fields = read_dat(dat)

args.out.parent.mkdir(parents=True, exist_ok=True)
np.savez(args.out, source=str(args.fld), geometry=str(args.geometry), fields=list(fields), **fields)
print(f"{args.out}: {', '.join(fields)} on the plane, from {args.fld}")
