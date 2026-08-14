#!/usr/bin/env python3
"""Phase-space video of the DO particle ensemble: Y1 vs Y2 over time.

Each particle is a small scatter point coloured by the sign of its initial
Y1 (red for Y1 < 0, blue for Y1 >= 0) and keeps that colour for the whole
video. Every particle leaves a thin trail behind that fades out
continuously with age, so trajectories are visible without overcrowding
the plot. Axes have equal aspect ratio.

Input is either
- an HDF5/NPZ file produced by py_utils/load_chk.py containing `yi`
  (P, S, T) and `t_do`, or
- a run directory, in which case the output/*.h5 file with the most DO
  frames is used; if none has a `yi` dataset, the Yi metadata of the raw
  output/do/*.do_*.fld snapshots is parsed instead (slower).

Example:
    python3 phase_space_video.py cases/se/runs/2026_07_10_do_m02_Re70_bimodal_pod_t500 \
        --out phase_space.mp4 --fps 25 --trail 5
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 14})


def _load_load_chk():
    mod_path = Path(__file__).resolve().parent.parent / "load_chk.py"
    spec = importlib.util.spec_from_file_location("load_chk", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load load_chk.py from {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _yi_from_h5(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (yi, t_do) from an NPZ/HDF5 loader output, or None if the
    file has no yi dataset."""
    if path.suffix == ".npz":
        data = np.load(path)
        if "yi" not in data:
            return None
        yi = np.asarray(data["yi"])
        t = np.asarray(data["t_do"]) if "t_do" in data else None
    else:
        import h5py
        with h5py.File(path, "r") as f:
            if "yi" not in f:
                return None
            yi = f["yi"][...]
            t = f["t_do"][...] if "t_do" in f else None
    if t is None:
        t = np.arange(yi.shape[-1], dtype=float)
    return yi, np.asarray(t, dtype=float)


def _yi_from_fld_snapshots(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse Yi from the metadata of every output/do/*.do_*.fld snapshot."""
    load_chk = _load_load_chk()
    files = load_chk._field_format_files(run_dir / "output" / "do")
    if not files:
        raise FileNotFoundError(f"{run_dir}: no output/do/*.do_*.fld snapshots")
    yis, ts = [], []
    for k, f in enumerate(files):
        md = load_chk._read_field_format_metadata(f)
        yis.append(md["yi"])
        ts.append(md["time"])
        if k % 200 == 0:
            print(f"  parsed {k+1}/{len(files)} snapshots", flush=True)
    yi = np.stack(yis, axis=-1)          # (P, S, T)
    return yi, np.asarray(ts, dtype=float)


def _load_run(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Resolve a run directory or data file into (yi, t_do)."""
    if path.is_file():
        out = _yi_from_h5(path)
        if out is None:
            raise ValueError(f"{path}: no yi dataset found")
        return out
    # run directory: prefer the postprocessed h5 with the most DO frames
    out_dir = path / "output" if (path / "output").is_dir() else path
    best = None
    for h5 in sorted(out_dir.glob("*.h5")):
        try:
            got = _yi_from_h5(h5)
        except OSError:
            continue
        if got is not None and (best is None or got[0].shape[-1] > best[1][0].shape[-1]):
            best = (h5, got)
    if best is not None:
        print(f"using yi from {best[0]} "
              f"({best[1][0].shape[-1]} frames, {best[1][0].shape[0]} particles)")
        return best[1]
    print("no *.h5 with a yi dataset; parsing output/do/*.fld metadata")
    return _yi_from_fld_snapshots(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", type=Path,
                    help="Run directory, or HDF5/NPZ file with yi and t_do.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output video (.mp4 or .gif). Default: "
                         "phase_space_y1y2.mp4 next to the input.")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--trail", type=float, default=10.0,
                    help="Trail length in time units (fades to zero over "
                         "this window). Default 10.")
    ap.add_argument("--qlim", type=float, default=0.2,
                    help="Percent of extreme Y values clipped per tail when "
                         "setting the axis limits, so brief excursions do "
                         "not inflate the view. 0 = full range. Default 0.2.")
    ap.add_argument("--stride", type=int, default=1,
                    help="Use every n-th DO frame. Default 1.")
    ap.add_argument("--marker-size", type=float, default=2.0,
                    help="Scatter marker area in points^2. Default 2.")
    ap.add_argument("--linewidth", type=float, default=0.5,
                    help="Trail line width. Default 0.5.")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    yi, t = _load_run(args.data)
    if yi.ndim != 3 or yi.shape[1] < 2:
        raise ValueError(f"expected yi of shape (P, S>=2, T), got {yi.shape}")
    yi = yi[:, :2, ::max(1, args.stride)]
    t = t[::max(1, args.stride)]
    P, _, T = yi.shape

    # initial colour by sign of Y1, kept for the whole video
    red = yi[:, 0, 0] < 0.0
    base_rgb = np.zeros((P, 3))
    base_rgb[red]  = matplotlib.colors.to_rgb("red")
    base_rgb[~red] = matplotlib.colors.to_rgb("blue")

    # trail length in frames from the DO sampling interval
    dt = float(np.median(np.diff(t))) if T > 1 else 1.0
    L = max(1, int(round(args.trail / dt)))
    # alpha of a segment of age a (in frames): quadratic fade to zero
    seg_alpha = 0.6 * (1.0 - np.arange(1, L + 1) / (L + 1.0))**2

    # robust axis limits: clip the qlim tails so the initial transient and
    # brief excursions do not inflate the view
    q = min(max(args.qlim, 0.0), 50.0)
    lo = np.percentile(yi, q,       axis=(0, 2))
    hi = np.percentile(yi, 100 - q, axis=(0, 2))
    pad = 0.05 * (hi - lo)
    lo, hi = lo - pad, hi + pad

    # figure sized to the data aspect so equal-aspect axes fill the frame
    xspan, yspan = hi - lo
    w = 10.0
    h = min(max(w * yspan / xspan + 1.2, 3.0), 10.0)  # +1.2in for labels/title
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$Y_1$")
    ax.set_ylabel(r"$Y_2$")

    trails = LineCollection([], linewidths=args.linewidth, zorder=2)
    ax.add_collection(trails)
    scat = ax.scatter(yi[:, 0, 0], yi[:, 1, 0], s=args.marker_size,
                      c=base_rgb, linewidths=0, zorder=3)
    title = ax.set_title(f"t = {t[0]:.2f}")
    fig.tight_layout()

    def _update(k: int):
        scat.set_offsets(yi[:, :, k].reshape(P, 2))
        start = max(0, k - L)
        n = k - start
        if n > 0:
            traj = np.transpose(yi[:, :, start:k + 1], (0, 2, 1))  # (P, n+1, 2)
            segs = np.stack([traj[:, :-1], traj[:, 1:]], axis=2)   # (P, n, 2, 2)
            trails.set_segments(segs.reshape(P * n, 2, 2))
            # segment ending at frame k-a has age a+1: alpha = seg_alpha[a]
            rgba = np.empty((P, n, 4))
            rgba[..., :3] = base_rgb[:, None, :]
            rgba[..., 3]  = seg_alpha[:n][::-1][None, :]
            trails.set_colors(rgba.reshape(P * n, 4))
        else:
            trails.set_segments([])
        title.set_text(f"t = {t[k]:.2f}")
        return scat, trails, title

    out = args.out
    if out is None:
        base = args.data if args.data.is_dir() else args.data.parent
        out = base / "phase_space_y1y2.mp4"

    ani = animation.FuncAnimation(fig, _update, frames=T, blit=False)
    writer = (animation.PillowWriter(fps=args.fps) if out.suffix == ".gif"
              else animation.FFMpegWriter(fps=args.fps))
    print(f"writing {T} frames to {out}")
    ani.save(out, writer=writer, dpi=args.dpi,
             progress_callback=lambda k, n:
                 print(f"  frame {k+1}/{n}", flush=True) if (k + 1) % 50 == 0 else None)
    print(f"done: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
