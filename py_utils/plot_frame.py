import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("data", type=Path, help="Input HDF5 file (out.h5 from load_chk.py)")
ap.add_argument("--out", type=Path, default=Path("lastframe.png"))
ap.add_argument("--field", choices=["vorticity", "u", "v", "p"], default="vorticity")
ap.add_argument("--pct", type=float, default=99.0,
                help="Percentile for symmetric color limit (default 99). "
                     "Lower this to bring out post-expansion structure.")
args = ap.parse_args()

with h5py.File(args.data, "r") as f:
    x = np.asarray(f["x"])
    y = np.asarray(f["y"])
    u = np.asarray(f["u"])[..., -1]
    v = np.asarray(f["v"])[..., -1]
    if args.field == "p":
        field_data = np.asarray(f["p"])[..., -1]

if args.field == "vorticity":
    dx = np.mean(np.diff(x[0, :]))
    dy = np.mean(np.diff(y[:, 0]))
    field_data = np.gradient(v, dx, axis=1) - np.gradient(u, dy, axis=0)
elif args.field == "u":
    field_data = u
elif args.field == "v":
    field_data = v

# FieldConvert writes exact zeros on all fields for points outside the mesh.
# Masking on (u==0)&(v==0) recovers the solid/step regions reliably.
outside = (u == 0.0) & (v == 0.0)
field_data = np.ma.masked_where(outside, field_data)

lim = np.nanpercentile(np.abs(field_data.compressed()), args.pct)
cmap = plt.cm.RdBu_r.copy()
cmap.set_bad("lightgray")
fig, ax = plt.subplots(figsize=(13, 3))
ax.set_facecolor("lightgray")
im = ax.imshow(field_data, origin="lower",
               extent=[x.min(), x.max(), y.min(), y.max()],
               cmap=cmap, vmin=-lim, vmax=lim)
fig.colorbar(im, ax=ax, label=args.field)
args.out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out, dpi=150)
print(f"Saved: {args.out}  (field={args.field}, pct={args.pct}, clim=+/-{lim:.4g})")
