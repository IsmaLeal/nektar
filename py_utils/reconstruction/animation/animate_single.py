#!/usr/bin/env python3
"""Animate a single (deterministic) realization.

Same visual quality as animate_do_panels.py's realization panel, but for a
case that has only u/v fields (e.g. deterministic NS sampling). Reads the
canonical out.h5 produced by py_utils/load_chk.py.

Required arrays in --data:
    u, v        shape (Ny, Nx, T)   velocity components
    x, y        shape (Ny, Nx)      mesh-grid coords
    t           shape (T,)          times (optional; falls back to frame index)

Default field is vorticity (RdBu_r, symmetric color limits, robust percentile,
chunked-in-time). For --field velocity adds a quiver overlay on |u| (viridis).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, colors
from matplotlib.animation import FFMpegWriter

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from common.data_io import close_data, load_data, read_array  # noqa: E402


def _arr(data, key: str) -> np.ndarray:
    return np.asarray(read_array(data, key))


def _dx_dy_from_grid(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    dx = float(np.mean(np.diff(x[0, :]))) if x.shape[1] > 1 else 1.0
    dy = float(np.mean(np.diff(y[:, 0]))) if y.shape[0] > 1 else 1.0
    return dx, dy


def _vorticity(u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> np.ndarray:
    # u, v: (Ny, Nx, T)
    dv_dx = np.gradient(v, dx, axis=1, edge_order=2)
    du_dy = np.gradient(u, dy, axis=0, edge_order=2)
    return dv_dx - du_dy


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True,
                    help="Input HDF5/NPZ from py_utils/load_chk.py (out.h5)")
    ap.add_argument("--out", type=Path, required=True, help="Output .mp4 path")
    ap.add_argument("--field", choices=["vorticity", "velocity"], default="vorticity")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--duration-sec", type=float, default=None,
                    help="Output video duration in seconds. Default = T/fps.")
    ap.add_argument("--dpi", type=int, default=140)
    ap.add_argument("--figsize", type=str, default="13x6.5",
                    help="Figure size in inches as WxH (default 13x6.5).")
    ap.add_argument("--robust-pct", type=float, default=99.0,
                    help="Percentile for robust color limit (default 99).")
    ap.add_argument("--adapt-every", type=int, default=12,
                    help="Frames between recomputing color limit (default 12).")
    ap.add_argument("--scale-mode", choices=["chunked", "global"], default="chunked")
    ap.add_argument("--norm", choices=["linear", "symlog", "power"], default="linear")
    ap.add_argument("--symlog-linthresh-ratio", type=float, default=0.05)
    ap.add_argument("--power-gamma", type=float, default=0.5)
    ap.add_argument("--xlim", type=str, default=None,
                    help="x-axis range as 'lo,hi' (default = data extent).")
    ap.add_argument("--ylim", type=str, default=None,
                    help="y-axis range as 'lo,hi' (default = data extent).")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Cap number of frames (for previews).")
    args = ap.parse_args()

    fig_w, fig_h = (float(s) for s in args.figsize.split("x"))

    data = load_data(args.data)
    try:
        u = _arr(data, "u")
        v = _arr(data, "v")
        x = _arr(data, "x")
        y = _arr(data, "y")
        try:
            t = _arr(data, "t")
        except Exception:
            t = None
    finally:
        close_data(data)

    # load_chk.py emits (ny, nx, nt); accept (nt, ny, nx) too.
    if u.ndim == 3 and u.shape[0] not in (x.shape[0], y.shape[0]):
        u = np.transpose(u, (1, 2, 0))
        v = np.transpose(v, (1, 2, 0))
    ny, nx, T = u.shape
    if t is None or len(t) != T:
        t = np.arange(T, dtype=float)

    if args.max_frames is not None:
        T = min(T, args.max_frames)
        u = u[..., :T]; v = v[..., :T]; t = t[:T]

    dx, dy = _dx_dy_from_grid(x, y)
    is_velocity = args.field == "velocity"
    if is_velocity:
        s = np.sqrt(u * u + v * v)
    else:
        s = _vorticity(u, v, dx, dy)

    robust_pct = float(np.clip(args.robust_pct, 0.0, 100.0))

    def robust_sym_lim(a: np.ndarray) -> float:
        return max(float(np.nanpercentile(np.abs(a), robust_pct)), 1e-12)

    def robust_pos_lim(a: np.ndarray) -> float:
        return max(float(np.nanpercentile(a, robust_pct)), 1e-12)

    def chunked_lims(a: np.ndarray, nframes: int, chunk: int, symmetric: bool) -> np.ndarray:
        out = np.empty(nframes, dtype=float)
        for start in range(0, nframes, chunk):
            stop = min(start + chunk, nframes)
            lim = robust_sym_lim(a[..., start:stop]) if symmetric else robust_pos_lim(a[..., start:stop])
            out[start:stop] = lim
        return out

    symmetric = not is_velocity
    if args.scale_mode == "chunked":
        lims = chunked_lims(s, T, max(1, args.adapt_every), symmetric=symmetric)
    else:
        lim = robust_sym_lim(s) if symmetric else robust_pos_lim(s)
        lims = np.full(T, lim)

    def make_norm(lim: float):
        if args.norm == "linear":
            return None
        if args.norm == "symlog":
            lt = max(float(args.symlog_linthresh_ratio) * lim, 1e-12)
            if symmetric:
                return colors.SymLogNorm(linthresh=lt, vmin=-lim, vmax=lim, base=10.0)
            return colors.SymLogNorm(linthresh=lt, vmin=0.0, vmax=lim, base=10.0)
        if symmetric:
            return colors.PowerNorm(gamma=float(args.power_gamma), vmin=-lim, vmax=lim)
        return colors.PowerNorm(gamma=float(args.power_gamma), vmin=0.0, vmax=lim)

    extent = [float(x.min()), float(x.max()), float(y.min()), float(y.max())]
    if args.xlim is not None:
        lo, hi = (float(s2) for s2 in args.xlim.split(","))
        extent[0], extent[1] = lo, hi
    if args.ylim is not None:
        lo, hi = (float(s2) for s2 in args.ylim.split(","))
        extent[2], extent[3] = lo, hi

    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect("equal")
    fig.subplots_adjust(left=0.06, right=0.95, top=0.92, bottom=0.10)

    lim0 = float(lims[0])
    norm0 = make_norm(lim0)
    cmap = "viridis" if is_velocity else "RdBu_r"

    if is_velocity:
        if norm0 is None:
            im = ax.imshow(s[:, :, 0], origin="lower", extent=extent, cmap=cmap,
                           vmin=0.0, vmax=lim0, animated=True)
        else:
            im = ax.imshow(s[:, :, 0], origin="lower", extent=extent, cmap=cmap,
                           norm=norm0, animated=True)
        # quiver downsample like do_panels
        qstep = max(1, min(nx, ny) // 20)
        xq = x[::qstep, ::qstep]; yq = y[::qstep, ::qstep]
        dxq = abs(float(np.mean(np.diff(xq[0, :])))) if xq.shape[1] > 1 else (extent[1]-extent[0])/20.0
        dyq = abs(float(np.mean(np.diff(yq[:, 0])))) if yq.shape[0] > 1 else (extent[3]-extent[2])/20.0
        target = 0.85 * max(min(dxq, dyq), 1e-8)
        qscale = 1.0 / target
        u0q = u[::qstep, ::qstep, 0]; v0q = v[::qstep, ::qstep, 0]
        m0q = s[::qstep, ::qstep, 0]
        den = np.where(m0q > 1e-14, m0q, 1.0)
        u0n = np.where(m0q > 1e-14, u0q/den, 0.0)
        v0n = np.where(m0q > 1e-14, v0q/den, 0.0)
        q = ax.quiver(xq, yq, u0n, v0n, pivot="mid", angles="xy",
                      scale_units="xy", scale=qscale, color="k", width=0.0035, animated=True)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"$|u|$")
    else:
        if norm0 is None:
            im = ax.imshow(s[:, :, 0], origin="lower", extent=extent, cmap=cmap,
                           vmin=-lim0, vmax=lim0, animated=True)
        else:
            im = ax.imshow(s[:, :, 0], origin="lower", extent=extent, cmap=cmap,
                           norm=norm0, animated=True)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"$\omega$")
        q = None

    ax.set_xlabel("x"); ax.set_ylabel("y")
    if args.xlim is not None: ax.set_xlim(extent[0], extent[1])
    if args.ylim is not None: ax.set_ylim(extent[2], extent[3])
    title = ax.set_title("")

    def set_frame(frame: int):
        f = s[:, :, frame]
        lim = float(lims[frame])
        n = make_norm(lim)
        if n is None:
            im.set_data(f); im.set_clim(vmin=(0.0 if is_velocity else -lim), vmax=lim)
        else:
            im.set_data(f); im.set_norm(n)
        if q is not None:
            uq = u[::qstep, ::qstep, frame]; vq = v[::qstep, ::qstep, frame]
            mq = s[::qstep, ::qstep, frame]
            den = np.where(mq > 1e-14, mq, 1.0)
            un = np.where(mq > 1e-14, uq/den, 0.0)
            vn = np.where(mq > 1e-14, vq/den, 0.0)
            q.set_UVC(un, vn)
        title.set_text(f"{'speed' if is_velocity else 'vorticity'}   t = {float(t[frame]):.3f}")
        artists = [im, title]
        if q is not None:
            artists.append(q)
        return artists

    set_frame(0)
    if args.duration_sec is not None and args.duration_sec > 0:
        fps = max(1, int(round(T / float(args.duration_sec))))
    else:
        fps = max(1, int(args.fps))

    anim = animation.FuncAnimation(fig, set_frame, frames=T, blit=False, interval=1000.0/fps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, codec="libx264",
                          extra_args=["-pix_fmt", "yuv420p", "-crf", "18"])
    anim.save(str(args.out), writer=writer, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved video: {args.out}  ({T} frames @ {fps} fps, dpi={args.dpi})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
