#!/usr/bin/env python3
"""compare_do_runs.py -- side-by-side comparison of two (or more) DO runs.

Usage:
    python3 compare_do_runs.py A/out.h5 B/out.h5 [C/out.h5 ...]
        [--time T]          snapshot time (default: last frame)
        [--yi-modes I J]    mode pair for Yi 2-D scatter (default: 0 1)
        [--labels A B ...]  one short label per run (default: parent dir name)
        [--out FILE]        .pdf or .png output (default: do_comparison.pdf)
        [--dpi INT]         raster resolution (default: 150)

Field panels use the first two runs only.
Time-series and Yi panels overlay all supplied runs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.stats import gaussian_kde as _kde
    _HAS_KDE = True
except ImportError:
    _HAS_KDE = False

# ---------------------------------------------------------------------------
# Palette helpers
# ---------------------------------------------------------------------------

_RUN_COLORS = list(plt.cm.tab10.colors)
_MODE_COLORS = list(plt.cm.Set1.colors[:9])
_RUN_STYLES  = ["-", "--", ":", "-."]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _label(path: Path) -> str:
    """Derive a short label from the path: <run_dir>/output/out.h5 -> run_dir."""
    return path.resolve().parent.parent.name


def load_run(path: Path) -> dict:
    d: dict = {}
    with h5py.File(path, "r") as f:
        for key in ("u", "v", "x", "y", "t"):
            d[key] = np.asarray(f[key])
        d["mode_u"] = np.asarray(f["mode_u"]) if "mode_u" in f else None
        d["yi"]     = np.asarray(f["yi"])     if "yi"     in f else None

        if "archive_diag_cov" in f:
            cov = np.asarray(f["archive_diag_cov"])
            if "do_dims" in f:
                dims = np.asarray(f["do_dims"]).reshape(-1)
                n_modes = int(dims[2]) if dims.size >= 3 else None
                if n_modes is not None and cov.shape[0] == n_modes + 1:
                    cov = cov[1:, :]   # strip leading kinetic-energy row
            d["diag_cov"] = cov        # (S, T)
        elif d["yi"] is not None:
            d["diag_cov"] = np.var(d["yi"], axis=0)   # empirical fallback (S, T)
        else:
            d["diag_cov"] = None
    return d

# ---------------------------------------------------------------------------
# Field-panel helpers
# ---------------------------------------------------------------------------

_CMAP: matplotlib.colors.Colormap | None = None

def _cmap():
    global _CMAP
    if _CMAP is None:
        _CMAP = plt.cm.RdBu_r.copy()
        _CMAP.set_bad("lightgray")
    return _CMAP


def _mask(field, u_snap, v_snap):
    return np.ma.masked_where((u_snap == 0.0) & (v_snap == 0.0), field)


def _vmax99(arr_masked):
    c = arr_masked.compressed()
    return float(np.nanpercentile(np.abs(c), 99)) if c.size else 1.0


def _imshow(ax, x, y, field_m, vmax, title, add_cbar=True, fig=None):
    ax.set_facecolor("lightgray")
    im = ax.imshow(
        field_m,
        origin="lower",
        extent=[x.min(), x.max(), y.min(), y.max()],
        cmap=_cmap(),
        vmin=-vmax, vmax=vmax,
        aspect="auto",
        interpolation="bilinear",
        rasterized=True,
    )
    ax.set_title(title, fontsize=7, pad=2)
    ax.tick_params(labelsize=5)
    ax.set_yticks([])
    if add_cbar and fig is not None:
        cb = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.01)
        cb.ax.tick_params(labelsize=5)
        cb.formatter.set_powerlimits((-2, 2))
        cb.update_ticks()
    return im

# ---------------------------------------------------------------------------
# Statistics-panel helpers
# ---------------------------------------------------------------------------

def _yi_scatter(ax, runs, t_idx, yi_modes, labels):
    mi, mj = yi_modes
    for run, label, color in zip(runs, labels, _RUN_COLORS):
        if run["yi"] is None:
            continue
        xi = run["yi"][:, mi, t_idx]
        xj = run["yi"][:, mj, t_idx]
        ax.scatter(xi, xj, s=14, alpha=0.45, color=color,
                   label=label, linewidths=0, zorder=3)
        if _HAS_KDE:
            try:
                pts = np.vstack([xi, xj])
                kde = _kde(pts, bw_method="scott")
                gx = np.linspace(xi.min(), xi.max(), 80)
                gy = np.linspace(xj.min(), xj.max(), 80)
                GX, GY = np.meshgrid(gx, gy)
                Z = kde(np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
                ax.contour(GX, GY, Z, levels=5, colors=[color],
                           alpha=0.55, linewidths=0.9, zorder=4)
            except Exception:
                pass
    ax.axhline(0, color="k", lw=0.4, alpha=0.35)
    ax.axvline(0, color="k", lw=0.4, alpha=0.35)
    ax.set_xlabel(f"$Y_{{{mi}}}$", fontsize=10)
    ax.set_ylabel(f"$Y_{{{mj}}}$", fontsize=10)
    ax.set_title(f"Yi scatter  (modes {mi} vs {mj})", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, markerscale=1.5, framealpha=0.8)
    ax.grid(True, alpha=0.18)


def _trace_c(ax, runs, labels, t_actual):
    for run, label, color, ls in zip(runs, labels, _RUN_COLORS, _RUN_STYLES):
        if run["diag_cov"] is None:
            continue
        trace = run["diag_cov"].sum(axis=0)
        ax.plot(run["t"], trace, color=color, ls=ls, lw=1.8, label=label)
    ax.axvline(t_actual, color="k", lw=0.9, ls="--", alpha=0.55,
               label=f"t = {t_actual:.1f}")
    ax.set_xlabel("time", fontsize=9)
    ax.set_ylabel("trace(C)", fontsize=9)
    ax.set_title("trace(C)  over time", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, framealpha=0.8)
    ax.grid(True, alpha=0.18)


def _mode_var(ax, runs, labels, t_actual):
    """Per-mode C_ii(t): colour = mode, linestyle = run."""
    n_modes = max(
        (r["diag_cov"].shape[0] for r in runs if r["diag_cov"] is not None),
        default=0,
    )
    # Build legend handles separately so they don't explode
    mode_handles = []
    for s in range(n_modes):
        for run, label, ls in zip(runs, labels, _RUN_STYLES):
            if run["diag_cov"] is None or s >= run["diag_cov"].shape[0]:
                continue
            ax.plot(
                run["t"], run["diag_cov"][s, :],
                color=_MODE_COLORS[s % len(_MODE_COLORS)],
                ls=ls, lw=1.2, alpha=0.85,
            )
        import matplotlib.patches as mpatches
        mode_handles.append(
            mpatches.Patch(color=_MODE_COLORS[s % len(_MODE_COLORS)],
                           label=f"mode {s}")
        )
    # Run legend (linestyle proxies)
    import matplotlib.lines as mlines
    run_handles = [
        mlines.Line2D([], [], color="k", ls=_RUN_STYLES[i % len(_RUN_STYLES)],
                      lw=1.5, label=labels[i])
        for i in range(min(len(runs), len(_RUN_STYLES)))
    ]
    leg1 = ax.legend(handles=mode_handles, fontsize=6, loc="upper left",
                     framealpha=0.8, title="mode", title_fontsize=6)
    ax.add_artist(leg1)
    ax.legend(handles=run_handles, fontsize=6, loc="upper right",
              framealpha=0.8, title="run", title_fontsize=6)

    ax.axvline(t_actual, color="k", lw=0.9, ls="--", alpha=0.55)
    ax.set_yscale("log")
    ax.set_xlabel("time", fontsize=9)
    ax.set_ylabel("C_ii", fontsize=9)
    ax.set_title("Per-mode variance  C_ii(t)", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.18, which="both")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Compare two or more DO simulation runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("h5_files", nargs="+", type=Path,
                    help="out.h5 files, one per run (at least 2 required)")
    ap.add_argument("--time", type=float, default=None,
                    help="Simulated time for snapshot panels (default: last frame)")
    ap.add_argument("--yi-modes", nargs=2, type=int, default=[0, 1],
                    metavar=("I", "J"),
                    help="Mode indices for Yi 2-D scatter (default: 0 1)")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="Short label per run (default: parent directory name)")
    ap.add_argument("--out", type=Path, default=Path("do_comparison.pdf"))
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    if len(args.h5_files) < 2:
        ap.error("Provide at least two h5 files.")

    labels = list(args.labels or [])
    while len(labels) < len(args.h5_files):
        labels.append(_label(args.h5_files[len(labels)]))

    print(f"Loading {len(args.h5_files)} run(s)...")
    runs = []
    for p, lbl in zip(args.h5_files, labels):
        print(f"  [{lbl}]  {p}")
        runs.append(load_run(p))

    # Time index
    t_arr  = runs[0]["t"]
    t_sel  = t_arr[-1] if args.time is None else args.time
    t_idx  = int(np.argmin(np.abs(t_arr - t_sel)))
    t_actual = float(t_arr[t_idx])
    print(f"Snapshot: requested t={t_sel:.2f}  ->  actual t={t_actual:.2f}  (index {t_idx})")

    # --- Figure skeleton ---
    n_modes = (runs[0]["mode_u"].shape[0]
               if runs[0]["mode_u"] is not None else 0)
    n_field_rows = 1 + n_modes          # mean + modes

    fig = plt.figure(figsize=(20, 2.0 * n_field_rows + 7))
    fig.patch.set_facecolor("white")

    outer = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[n_field_rows * 2.0, 6.5],
        hspace=0.08,
    )
    gs_top = gridspec.GridSpecFromSubplotSpec(
        n_field_rows, 3, subplot_spec=outer[0],
        hspace=0.12, wspace=0.03,
    )
    gs_bot = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[1],
        wspace=0.30,
    )

    # --- Field rows (first two runs) ---
    ra, rb   = runs[0], runs[1]
    la, lb   = labels[0], labels[1]
    x, y     = ra["x"], ra["y"]
    ua = ra["u"][..., t_idx];  va = ra["v"][..., t_idx]
    ub = rb["u"][..., t_idx];  vb = rb["v"][..., t_idx]

    def draw_row(row, fa, fb, row_label):
        fa_m = _mask(fa, ua, va)
        fb_m = _mask(fb, ub, vb)
        diff  = _mask(fa - fb, ua, va)
        vab = max(_vmax99(fa_m), _vmax99(fb_m), 1e-12)
        vd  = max(_vmax99(diff), 1e-12)

        ax0 = fig.add_subplot(gs_top[row, 0])
        ax1 = fig.add_subplot(gs_top[row, 1])
        ax2 = fig.add_subplot(gs_top[row, 2])

        _imshow(ax0, x, y, fa_m, vab, f"{row_label}   [{la}]",
                add_cbar=False)
        _imshow(ax1, x, y, fb_m, vab, f"{row_label}   [{lb}]",
                add_cbar=True, fig=fig)
        _imshow(ax2, x, y, diff,  vd,  f"{row_label}   {la} − {lb}",
                add_cbar=True, fig=fig)

    draw_row(0, ua, ub, "mean u")
    if ra["mode_u"] is not None and rb["mode_u"] is not None:
        for s in range(n_modes):
            draw_row(s + 1, ra["mode_u"][s, ..., t_idx],
                     rb["mode_u"][s, ..., t_idx], f"mode {s}")

    # --- Statistics panels (all runs) ---
    ax_yi = fig.add_subplot(gs_bot[0, 0])
    ax_tc = fig.add_subplot(gs_bot[0, 1])
    ax_cv = fig.add_subplot(gs_bot[0, 2])

    _yi_scatter(ax_yi, runs, t_idx, args.yi_modes, labels)
    _trace_c(ax_tc, runs, labels, t_actual)
    _mode_var(ax_cv, runs, labels, t_actual)

    # --- Title ---
    fig.suptitle(
        f"DO run comparison  |  t = {t_actual:.2f} s  |  "
        f"Yi modes ({args.yi_modes[0]}, {args.yi_modes[1]})",
        fontsize=13, fontweight="bold", y=1.001,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
