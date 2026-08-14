#!/usr/bin/env python3
"""Animate a DO run in a 3x3-style panel layout.

Layout:
- (0,0): mean field
- (0,1),(0,2),(1,1),(1,2): first 4 mode fields
- (1,0): one random reconstructed realization field
- bottom row (spans all columns): Yi variances vs time

Requires an NPZ/HDF5 file produced by `py_utils/load_chk.py` with:
- mean fields `u`, `v`
- interpolated modes `mode_u`, `mode_v` (run loader with --interp-modes-from-archive)
- coefficients `yi`
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import colors
from matplotlib.gridspec import GridSpec


def _load_reconstruct_module():
    here = Path(__file__).resolve()
    mod_path = here.parent.parent / "reconstruct_fields.py"
    spec = importlib.util.spec_from_file_location("reconstruct_fields", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load reconstruct_fields.py from {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _aligned_uv_and_yi(data, recon_mod):
    u_mean, mode_u, yi = recon_mod._aligned_triplet(data, "u", "mode_u")
    v_mean, mode_v, yi2 = recon_mod._aligned_triplet(data, "v", "mode_v")
    if yi.shape != yi2.shape:
        raise ValueError("Aligned yi shapes differ between u and v reconstruction paths")
    return u_mean, v_mean, mode_u, mode_v, yi


def _dx_dy_from_grid(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    # x,y are meshgrid arrays of shape (Ny,Nx)
    dx = float(np.mean(np.diff(x[0, :]))) if x.shape[1] > 1 else 1.0
    dy = float(np.mean(np.diff(y[:, 0]))) if y.shape[0] > 1 else 1.0
    return dx, dy


def _vorticity(u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> np.ndarray:
    # u,v shape (..., Ny, Nx, T) or (Ny,Nx,T). We expect (Ny,Nx,T) here.
    dv_dx = np.gradient(v, dx, axis=1, edge_order=2)
    du_dy = np.gradient(u, dy, axis=0, edge_order=2)
    return dv_dx - du_dy


def _pick_particle(data, yi: np.ndarray, seed: int | None, particle: int | None) -> int:
    if particle is not None:
        if not (0 <= particle < yi.shape[0]):
            raise ValueError(f"particle index {particle} out of range [0,{yi.shape[0]-1}]")
        return particle
    rng = np.random.default_rng(seed)
    return int(rng.integers(0, yi.shape[0]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Input NPZ/HDF5 from py_utils/load_chk.py. Defaults to ./out.h5 when omitted.",
    )
    ap.add_argument(
        "--npz",
        type=Path,
        default=None,
        help="Deprecated alias for --data (kept for backward compatibility).",
    )
    ap.add_argument("--out", type=Path, default=None, help="Output animation file (.mp4 or .gif)")
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--dpi", type=int, default=120)
    ap.add_argument("--particle", type=int, default=None, help="Particle index for realization panel")
    ap.add_argument("--seed", type=int, default=0, help="Seed for random particle selection if --particle not set")
    ap.add_argument("--max-frames", type=int, default=None, help="Limit number of frames for quick previews")
    ap.add_argument(
        "--field",
        choices=["vorticity", "u", "v", "w", "velocity"],
        default="u",
        help="Top 6-panel quantity: vorticity (scalar map) or velocity (colored arrows only).",
    )
    ap.add_argument(
        "--scale-mode",
        choices=["fixed", "chunked"],
        default="fixed",
        help="Colorbar scaling mode: fixed over all frames, or chunked updates.",
    )
    ap.add_argument(
        "--adapt-every",
        type=int,
        default=5,
        help="Chunk size in frames when --scale-mode=chunked.",
    )
    ap.add_argument(
        "--robust-pct",
        type=float,
        default=95.0,
        help=("Percentile used to compute robust fixed/chunked limits. "
              "Default 95 saturates ~5% of the most extreme points (boundary "
              "ringing, isolated element-edge spikes) so the colormap "
              "concentrates on the bulk of the mode amplitude. Bump to 99+ "
              "if you want extreme outliers to define the limit instead."),
    )
    ap.add_argument(
        "--norm",
        choices=["linear", "symlog", "power"],
        default="linear",
        help="Colormap normalization.",
    )
    ap.add_argument(
        "--symlog-linthresh-ratio",
        type=float,
        default=0.05,
        help="linthresh = ratio * panel_limit for --norm=symlog.",
    )
    ap.add_argument(
        "--power-gamma",
        type=float,
        default=0.6,
        help="Gamma for --norm=power (smaller boosts low amplitudes).",
    )
    args = ap.parse_args()

    if args.data is not None and args.npz is not None:
        raise ValueError("Pass only one of --data or --npz.")
    in_path = args.data if args.data is not None else args.npz
    if in_path is None:
        in_path = Path("out.h5")
    if not in_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {in_path}. Pass --data <path> or --npz <path>."
        )

    recon_mod = _load_reconstruct_module()
    data = recon_mod._open_data(in_path)
    required = ["u", "v", "mode_u", "mode_v", "yi", "x", "y"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(
            "Missing arrays in input container: "
            f"{missing}. Run load_chk.py with --interp-modes-from-archive."
        )

    u_mean, v_mean, mode_u, mode_v, yi = _aligned_uv_and_yi(data, recon_mod)
    # Shapes: mean (Ny,Nx,K), modes (S,Ny,Nx,K), yi (P,S,K)
    ny, nx, K = u_mean.shape
    S = mode_u.shape[0]
    K = min(K, mode_u.shape[-1], yi.shape[-1])
    if args.max_frames is not None:
        K = min(K, args.max_frames)

    u_mean = u_mean[..., :K]
    v_mean = v_mean[..., :K]
    mode_u = mode_u[..., :K]
    mode_v = mode_v[..., :K]
    yi = yi[..., :K]

    x = recon_mod._arr(data, "x")
    y = recon_mod._arr(data, "y")
    dx, dy = _dx_dy_from_grid(x, y)

    nmodes_plot = min(4, S)

    pidx = _pick_particle(data, yi, args.seed, args.particle)
    # One realization only, avoid reconstructing all particles
    u_real = u_mean + np.einsum("sk,sxyk->xyk", yi[pidx], mode_u)
    v_real = v_mean + np.einsum("sk,sxyk->xyk", yi[pidx], mode_v)

    is_velocity = args.field in ["u", "v", "w", "velocity"]
    # "velocity" is a magnitude (positive, sequential cmap); u/v/w are signed
    # components (diverging cmap, symmetric limits about zero).
    is_speed = args.field == "velocity"
    if is_velocity:
        if args.field == "u":
            s_mean = u_mean
            s_modes = np.empty((nmodes_plot, ny, nx, K), dtype=np.float64)
            for m in range(nmodes_plot):
                s_modes[m] = mode_u[m]
            s_real = u_real
        elif args.field == "v":
            s_mean = v_mean
            s_modes = np.empty((nmodes_plot, ny, nx, K), dtype=np.float64)
            for m in range(nmodes_plot):
                s_modes[m] = mode_v[m]
            s_real = v_real
        elif args.field == "w":
            pass
        elif args.field == "velocity":
            s_mean = np.sqrt(u_mean * u_mean + v_mean * v_mean)
            s_modes = np.empty((nmodes_plot, ny, nx, K), dtype=np.float64)
            for m in range(nmodes_plot):
                s_modes[m] = np.sqrt(mode_u[m] * mode_u[m] + mode_v[m] * mode_v[m])
            s_real = np.sqrt(u_real * u_real + v_real * v_real)
    else:
        s_mean = _vorticity(u_mean, v_mean, dx, dy)
        s_modes = np.empty((nmodes_plot, ny, nx, K), dtype=np.float64)
        for m in range(nmodes_plot):
            s_modes[m] = _vorticity(mode_u[m], mode_v[m], dx, dy)
        s_real = _vorticity(u_real, v_real, dx, dy)

    # Yi variances vs time (across particles)
    yi_var = np.var(yi, axis=0)  # (S,K)
    t_do = recon_mod._arr(data, "t_do") if "t_do" in data else np.arange(K, dtype=float)
    t_do = np.asarray(t_do)[:K]

    # Color scaling and normalization.
    adapt_every = max(1, int(args.adapt_every))
    robust_pct = float(np.clip(args.robust_pct, 0.0, 100.0))

    def robust_sym_lim(a: np.ndarray) -> float:
        lim = float(np.nanpercentile(np.abs(a), robust_pct))
        return max(lim, 1e-12)

    def robust_pos_lim(a: np.ndarray) -> float:
        lim = float(np.nanpercentile(a, robust_pct))
        return max(lim, 1e-12)

    def make_norm(lim: float, symmetric: bool):
        if args.norm == "linear":
            return None
        if args.norm == "symlog":
            linthresh = max(float(args.symlog_linthresh_ratio) * lim, 1e-12)
            if symmetric:
                return colors.SymLogNorm(linthresh=linthresh, vmin=-lim, vmax=lim, base=10.0)
            return colors.SymLogNorm(linthresh=linthresh, vmin=0.0, vmax=lim, base=10.0)
        if symmetric:
            return colors.PowerNorm(gamma=float(args.power_gamma), vmin=-lim, vmax=lim)
        return colors.PowerNorm(gamma=float(args.power_gamma), vmin=0.0, vmax=lim)

    def chunked_lims(a: np.ndarray, nframes: int, chunk: int, symmetric: bool) -> np.ndarray:
        """Per-frame limits updated once per chunk."""
        out = np.empty(nframes, dtype=float)
        for start in range(0, nframes, chunk):
            stop = min(start + chunk, nframes)
            if symmetric:
                lim = robust_sym_lim(a[..., start:stop])
            else:
                lim = robust_pos_lim(a[..., start:stop])
            out[start:stop] = lim
        return out

    panel_keys = ["mean", "m1", "m2", "real", "m3", "m4"]
    if is_velocity:
        fname = "speed" if is_speed else args.field
    else:
        fname = "vorticity"
    panel_titles = [
        f"Mean {fname}",
        f"Mode 1 {fname}",
        f"Mode 2 {fname}",
        f"Realization {fname} (particle {pidx})",
        f"Mode 3 {fname}",
        f"Mode 4 {fname}",
    ]
    panel_scalars = [s_mean, None, None, s_real, None, None]
    if nmodes_plot > 0:
        panel_scalars[1] = s_modes[0]
    if nmodes_plot > 1:
        panel_scalars[2] = s_modes[1]
    if nmodes_plot > 2:
        panel_scalars[4] = s_modes[2]
    if nmodes_plot > 3:
        panel_scalars[5] = s_modes[3]

    panel_u = [u_mean, None, None, u_real, None, None]
    panel_v = [v_mean, None, None, v_real, None, None]
    if nmodes_plot > 0:
        panel_u[1], panel_v[1] = mode_u[0], mode_v[0]
    if nmodes_plot > 1:
        panel_u[2], panel_v[2] = mode_u[1], mode_v[1]
    if nmodes_plot > 2:
        panel_u[4], panel_v[4] = mode_u[2], mode_v[2]
    if nmodes_plot > 3:
        panel_u[5], panel_v[5] = mode_u[3], mode_v[3]

    panel_lims = []
    for fld in panel_scalars:
        if fld is None:
            panel_lims.append(None)
        else:
            symmetric = not is_speed
            if args.scale_mode == "chunked":
                panel_lims.append(chunked_lims(fld, K, adapt_every, symmetric=symmetric))
            else:
                panel_lims.append(robust_sym_lim(fld) if symmetric else robust_pos_lim(fld))

    # constrained_layout=False: the layout solver re-runs on every draw, and
    # any per-frame tick-text width changes on the colorbars get amplified into
    # apparent panel "zoom" jitter. Fix the layout once at init via subplots_adjust.
    fig = plt.figure(figsize=(13, 9), constrained_layout=False)
    gs = GridSpec(4, 3, figure=fig, height_ratios=[0.42, 0.42, 1.0, 0.62],
                  hspace=0.5, wspace=0.22,
                  left=0.055, right=0.985, top=0.94, bottom=0.06)

    axs = {
        "mean": fig.add_subplot(gs[0, 0]),
        "m1": fig.add_subplot(gs[0, 1]),
        "m2": fig.add_subplot(gs[0, 2]),
        "real": fig.add_subplot(gs[1, 0]),
        "m3": fig.add_subplot(gs[1, 1]),
        "m4": fig.add_subplot(gs[1, 2]),
        "sc12": fig.add_subplot(gs[2, 0]),
        "sc23": fig.add_subplot(gs[2, 1]),
        "sc34": fig.add_subplot(gs[2, 2]),
        "var": fig.add_subplot(gs[3, :]),
    }

    extent = [float(x.min()), float(x.max()), float(y.min()), float(y.max())]
    panel_scalar_mappables = [None] * 6
    panel_quivers = [None] * 6
    panel_cbars = [None] * 6

    def _lim_at(idx: int, frame: int) -> float:
        lim_src = panel_lims[idx]
        if isinstance(lim_src, np.ndarray):
            return float(lim_src[frame])
        return float(lim_src)

    if is_velocity:
        qstep = max(1, min(nx, ny) // 14)
        xq = x[::qstep, ::qstep]
        yq = y[::qstep, ::qstep]
        if xq.shape[1] > 1:
            dxq = abs(float(np.mean(np.diff(xq[0, :]))))
        else:
            dxq = (extent[1] - extent[0]) / 20.0
        if yq.shape[0] > 1:
            dyq = abs(float(np.mean(np.diff(yq[:, 0]))))
        else:
            dyq = (extent[3] - extent[2]) / 20.0
        target_len = 0.85 * max(min(dxq, dyq), 1e-8)
        quiver_scale = 1.0 / target_len

        def _q_components(idx: int, frame: int):
            uq = panel_u[idx][:, :, frame][::qstep, ::qstep]
            vq = panel_v[idx][:, :, frame][::qstep, ::qstep]
            # normalize arrows by the local speed, not the plotted scalar
            # (the plotted scalar is signed for u/v/w panels)
            mq = np.sqrt(uq * uq + vq * vq)
            den = np.where(mq > 1e-14, mq, 1.0)
            un = np.where(mq > 1e-14, uq / den, 0.0)
            vn = np.where(mq > 1e-14, vq / den, 0.0)
            return un, vn, mq

    for idx, key in enumerate(panel_keys):
        ax = axs[key]
        fld = panel_scalars[idx]
        if fld is None:
            ax.set_title("(unused)")
            ax.set_visible(False)
            continue

        ax.set_title(panel_titles[idx])
        lim0 = _lim_at(idx, 0)
        norm0 = make_norm(lim0, symmetric=not is_speed)
        if is_velocity:
            un0, vn0, mq0 = _q_components(idx, 0)
            if is_speed:
                im_kwargs = dict(origin="lower", extent=extent, cmap="viridis", animated=True)
                vmin0, vmax0 = 0.0, lim0
            else:
                im_kwargs = dict(origin="lower", extent=extent, cmap="RdBu_r", animated=True)
                vmin0, vmax0 = -lim0, lim0
            if norm0 is None:
                im = ax.imshow(fld[:, :, 0], vmin=vmin0, vmax=vmax0, **im_kwargs)
            else:
                im = ax.imshow(fld[:, :, 0], norm=norm0, **im_kwargs)
            q = ax.quiver(
                xq,
                yq,
                un0,
                vn0,
                pivot="mid",
                angles="xy",
                scale_units="xy",
                scale=quiver_scale,
                color="k",
                width=0.0035,
                animated=True,
            )
            panel_scalar_mappables[idx] = im
            panel_quivers[idx] = q
        else:
            im_kwargs = dict(origin="lower", extent=extent, cmap="RdBu_r", animated=True)
            if norm0 is None:
                im = ax.imshow(fld[:, :, 0], vmin=-lim0, vmax=lim0, **im_kwargs)
            else:
                im = ax.imshow(fld[:, :, 0], norm=norm0, **im_kwargs)
            panel_scalar_mappables[idx] = im

    for key in ["mean", "m1", "m2", "real", "m3", "m4"]:
        axs[key].set_xlabel("x")
        axs[key].set_ylabel("y")

    # 2D scatters of consecutive coefficient pairs, with the realization of
    # the field panel highlighted as a single colored point in each. Panels
    # with indices beyond S are hidden.
    sc_pct = robust_pct                                                     # reuse the same percentile knob as field panels
    def _sym_lim(arr: np.ndarray) -> float:
        a = float(np.nanpercentile(np.abs(arr), sc_pct))
        return max(a, 1e-12)

    def _init_pair(ax, ia: int, ib: int):
        if S <= max(ia, ib):
            ax.set_visible(False)
            return None, None
        s = ax.scatter(yi[:, ia, 0], yi[:, ib, 0], s=8, alpha=0.5)
        hi = ax.scatter([yi[pidx, ia, 0]], [yi[pidx, ib, 0]], s=40,
                        color="#D55E00", zorder=5)
        ax.set_title(f"$(Y_{ia+1},Y_{ib+1})$")
        ax.set_xlabel(f"$Y_{ia+1}$")
        ax.set_ylabel(f"$Y_{ib+1}$")
        xl = _sym_lim(yi[:, ia, :])
        yl = _sym_lim(yi[:, ib, :])
        ax.set_xlim(-xl, xl)
        ax.set_ylim(-yl, yl)
        ax.autoscale(enable=False)          # belt-and-braces against per-frame relim
        return s, hi

    sc12, hi12 = _init_pair(axs["sc12"], 0, 1)
    sc23, hi23 = _init_pair(axs["sc23"], 1, 2)
    sc34, hi34 = _init_pair(axs["sc34"], 2, 3)

    # Variance panel
    lines = []
    for i in range(min(S, 10)):
        (ln,) = axs["var"].plot(t_do, yi_var[i], lw=1.5, label=f"Var(Y{i+1})")
        lines.append(ln)
    time_marker = axs["var"].axvline(t_do[0] if K else 0.0, color="k", ls="--", lw=1)
    axs["var"].set_title("DO coefficient variances vs time")
    axs["var"].set_xlabel("time")
    axs["var"].set_ylabel("variance")
    axs["var"].set_yscale("log")
    axs["var"].grid(True, which="both", alpha=0.3)
    # pin x and y view to the static data range so per-frame redraw can't relim it
    axs["var"].set_xlim(float(t_do.min()), float(t_do.max()))
    if yi_var.size > 0:
        finite = yi_var[np.isfinite(yi_var) & (yi_var > 0)]
        if finite.size:
            ymin = float(finite.min()) * 0.5
            ymax = float(finite.max()) * 2.0
            axs["var"].set_ylim(ymin, ymax)
    axs["var"].autoscale(enable=False)
    if lines:
        axs["var"].legend(loc="upper right", ncol=2, fontsize=8)

    supt = fig.suptitle("")
    last_chunk = -1

    def _update(frame: int):
        nonlocal last_chunk
        artists = []
        chunk_idx = frame // adapt_every if args.scale_mode == "chunked" else 0
        if args.scale_mode == "chunked" and chunk_idx != last_chunk:
            for idx in range(6):
                if panel_lims[idx] is None:
                    continue
                lim = _lim_at(idx, frame)
                symmetric = not is_speed
                if args.norm == "linear":
                    if is_speed:
                        panel_scalar_mappables[idx].set_clim(0.0, lim)
                    else:
                        panel_scalar_mappables[idx].set_clim(-lim, lim)
                else:
                    panel_scalar_mappables[idx].set_norm(make_norm(lim, symmetric=symmetric))
                if panel_cbars[idx] is not None:
                    panel_cbars[idx].update_normal(panel_scalar_mappables[idx])
                    artists.append(panel_cbars[idx].ax)
            last_chunk = chunk_idx

        for idx in range(6):
            if panel_lims[idx] is None:
                continue
            if is_velocity:
                panel_scalar_mappables[idx].set_array(panel_scalars[idx][:, :, frame])
                un, vn, _ = _q_components(idx, frame)
                panel_quivers[idx].set_UVC(un, vn)
                artists.append(panel_scalar_mappables[idx])
                artists.append(panel_quivers[idx])
            else:
                panel_scalar_mappables[idx].set_array(panel_scalars[idx][:, :, frame])
                artists.append(panel_scalar_mappables[idx])

        for cloud, hi, ia, ib in ((sc12, hi12, 0, 1), (sc23, hi23, 1, 2),
                                  (sc34, hi34, 2, 3)):
            if cloud is None:
                continue
            cloud.set_offsets(np.column_stack([yi[:, ia, frame], yi[:, ib, frame]]))
            hi.set_offsets([[yi[pidx, ia, frame], yi[pidx, ib, frame]]])
            artists.extend([cloud, hi])

        if frame < len(t_do):
            time_marker.set_xdata([t_do[frame], t_do[frame]])
            supt.set_text(f"Frame {frame+1}/{K} | t={t_do[frame]:.6g}")
        else:
            supt.set_text(f"Frame {frame+1}/{K}")
        artists.extend([time_marker, supt])
        return artists

    ani = animation.FuncAnimation(fig, _update, frames=K, interval=1000/max(args.fps,1), blit=False)

    out = args.out or in_path.with_name(in_path.stem + "_do_panels.mp4")
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.suffix.lower() == ".gif":
        ani.save(out, writer=animation.PillowWriter(fps=args.fps))
    else:
        ani.save(out, writer=animation.FFMpegWriter(fps=args.fps))
    if hasattr(data, "close"):
        data.close()
    print(f"Saved animation: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
