#!/usr/bin/env python3
"""
Render a vorticity video from velocity arrays saved in NPZ/HDF5.

Expected container fields (from load_chk.py):
    - u: (nt, ny, nx)
    - v: (nt, ny, nx)
    - x: (ny, nx)
    - y: (ny, nx)
    - t: (nt,)

Example:
    python3 /home/isma/nektar++/py_utils/vorticity_video.py \
      --data /home/isma/nektar++/cases/cylinder_flow_v3(higherRe)/output/out.h5 \
      --out /home/isma/nektar++/cases/cylinder_flow_v3(higherRe)/output/vorticity.mp4 \
      --fps 30 \
      --duration-sec 12
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

try:
    from py_utils.common.data_io import close_data, load_data, read_array
except Exception:  # pragma: no cover - direct script execution fallback
    from common.data_io import close_data, load_data, read_array


def _validate_inputs(
    u: np.ndarray, v: np.ndarray, x: np.ndarray, y: np.ndarray, t: np.ndarray
) -> None:
    if u.ndim != 3 or v.ndim != 3:
        raise ValueError(f"u and v must be 3D arrays (nt, ny, nx); got {u.shape}, {v.shape}")
    if u.shape != v.shape:
        raise ValueError(f"u and v shapes must match; got {u.shape} vs {v.shape}")
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"x and y must be 2D arrays (ny, nx); got {x.shape}, {y.shape}")
    if x.shape != y.shape:
        raise ValueError(f"x and y shapes must match; got {x.shape} vs {y.shape}")
    if x.shape != u.shape[1:]:
        raise ValueError(f"x/y shape {x.shape} must equal u/v spatial shape {u.shape[1:]}")
    if t.ndim != 1 or t.shape[0] != u.shape[0]:
        raise ValueError(f"t must be shape (nt,), got {t.shape} with nt={u.shape[0]}")


def _compute_vorticity_frame(
    u_frame: np.ndarray, v_frame: np.ndarray, dx: float, dy: float
) -> np.ndarray:
    du_dy = np.gradient(u_frame, dy, axis=0)
    dv_dx = np.gradient(v_frame, dx, axis=1)
    return dv_dx - du_dy


def _robust_omega_limit(
    u: np.ndarray, v: np.ndarray, dx: float, dy: float, sample_count: int, percentile: float
) -> float:
    nt = u.shape[0]
    idx = np.linspace(0, nt - 1, num=min(sample_count, nt), dtype=int)
    vals = []
    for k in idx:
        w = _compute_vorticity_frame(u[k], v[k], dx=dx, dy=dy)
        vals.append(np.abs(w).ravel())
    all_abs = np.concatenate(vals)
    lim = np.percentile(all_abs, percentile)
    return float(max(lim, 1.0e-12))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Input NPZ/HDF5 produced by load_chk.py. Defaults to ./out.h5 when omitted.",
    )
    ap.add_argument(
        "--npz",
        type=Path,
        default=None,
        help="Deprecated alias for --data (kept for backward compatibility).",
    )
    ap.add_argument("--out", type=Path, required=True, help="Output .mp4 path")
    ap.add_argument("--fps", type=int, required=True, help="Output video FPS")
    ap.add_argument(
        "--duration-sec", type=float, required=True, help="Output video duration in seconds"
    )
    ap.add_argument(
        "--cmap", default="RdBu_r", help="Colormap for vorticity field (default: RdBu_r)"
    )
    ap.add_argument(
        "--omega-max",
        type=float,
        default=None,
        help="Fixed symmetric color limit for vorticity (+/-omega_max). "
        "If omitted, inferred robustly from data.",
    )
    ap.add_argument(
        "--omega-percentile",
        type=float,
        default=99.5,
        help="Percentile for automatic symmetric color limit if --omega-max is not set.",
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=130,
        help="Video DPI (resolution control).",
    )
    args = ap.parse_args()

    if args.fps <= 0:
        raise ValueError("--fps must be > 0")
    if args.duration_sec <= 0:
        raise ValueError("--duration-sec must be > 0")
    if not (50.0 <= args.omega_percentile < 100.0):
        raise ValueError("--omega-percentile must be in [50, 100)")

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install ffmpeg or add it to PATH before running."
        )

    if args.data is not None and args.npz is not None:
        raise ValueError("Pass only one of --data or --npz.")

    in_path = args.data if args.data is not None else args.npz
    if in_path is None:
        in_path = Path("out.h5")
    if not in_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {in_path}. Pass --data <path> or --npz <path>."
        )

    data = load_data(in_path)
    try:
        u = read_array(data, "u")
        v = read_array(data, "v")
        x = read_array(data, "x")
        y = read_array(data, "y")
        t = read_array(data, "t")
    finally:
        close_data(data)

    # load_chk.py emits (ny, nx, nt); this script expects (nt, ny, nx). Transpose
    # if the trailing axis matches t's length.
    if u.ndim == 3 and u.shape[-1] == t.shape[0] and u.shape[0] != t.shape[0]:
        u = np.moveaxis(u, -1, 0)
        v = np.moveaxis(v, -1, 0)

    _validate_inputs(u, v, x, y, t)

    dx = float(np.mean(np.diff(x[0, :])))
    dy = float(np.mean(np.diff(y[:, 0])))
    if dx <= 0 or dy <= 0:
        raise ValueError(f"Non-positive grid spacing: dx={dx}, dy={dy}")

    n_frames = max(1, int(round(args.fps * args.duration_sec)))
    nt = u.shape[0]
    sample_idx = np.linspace(0, nt - 1, num=n_frames, dtype=int)

    if args.omega_max is None:
        omega_lim = _robust_omega_limit(
            u=u, v=v, dx=dx, dy=dy, sample_count=40, percentile=args.omega_percentile
        )
    else:
        omega_lim = float(args.omega_max)
        if omega_lim <= 0:
            raise ValueError("--omega-max must be > 0")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    w0 = _compute_vorticity_frame(u[sample_idx[0]], v[sample_idx[0]], dx=dx, dy=dy)
    im = ax.imshow(
        w0,
        extent=[x.min(), x.max(), y.min(), y.max()],
        origin="lower",
        cmap=args.cmap,
        vmin=-omega_lim,
        vmax=omega_lim,
        interpolation="nearest",
        animated=True,
        aspect="equal",
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("vorticity omega_z")
    title = ax.set_title("")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()

    writer = FFMpegWriter(
        fps=args.fps,
        metadata={"title": "Nektar++ vorticity", "artist": "py_utils/vorticity_video.py"},
        codec="libx264",
        bitrate=-1,
    )

    with writer.saving(fig, str(args.out), args.dpi):
        for i, k in enumerate(sample_idx):
            w = _compute_vorticity_frame(u[k], v[k], dx=dx, dy=dy)
            im.set_data(w)
            title.set_text(
                f"Frame {i + 1}/{n_frames}   Snapshot {k + 1}/{nt}   t={float(t[k]):.4f}"
            )
            writer.grab_frame()
            if (i + 1) % max(1, n_frames // 20) == 0 or i == n_frames - 1:
                print(f"[{i + 1:4d}/{n_frames:4d}] rendered")

    plt.close(fig)
    print(f"\nSaved video: {args.out}")
    print(f"Frames: {n_frames}, fps: {args.fps}, duration: {n_frames / args.fps:.3f} s")
    print(f"Color limit: +/-{omega_lim:.6g}, colormap: {args.cmap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
