#!/usr/bin/env python3
"""do_subspace_convergence.py -- convergence diagnostics for two DO runs.

Usage:
    python3 do_subspace_convergence.py A/out.h5 B/out.h5
        [--labels A B]   short labels (default: parent dir name)
        [--out FILE]     output pdf or png (default: do_convergence.pdf)
        [--dpi INT]      raster resolution (default: 150)

Three panels (all vs simulated time):
  1. Principal angles between the S-dimensional mode subspaces [degrees].
     Stacks u+v velocity components; uses the cross-Gram SVD.
     All angles -> 0: subspaces converged. Flat/growing: not converging.
  3. Per-mode variance C_ii(t) = var_p(Y_{i,p}), semilogy, one line per
     mode per run (colour = mode, linestyle = run).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np


_MODE_COLORS = list(plt.cm.Set1.colors[:9])
_RUN_STYLES  = ["-", "--", ":", "-."]


def _label(path: Path) -> str:
    return path.resolve().parent.parent.name


def load_run(path: Path) -> dict:
    d: dict = {}
    with h5py.File(path, "r") as f:
        d["t"]      = np.asarray(f["t"])           # (T,)
        d["u"]      = np.asarray(f["u"])            # (Ny, Nx, T)
        d["v"]      = np.asarray(f["v"])
        d["mode_u"] = np.asarray(f["mode_u"])       # (S, Ny, Nx, T)
        d["mode_v"] = np.asarray(f["mode_v"])
        d["yi"]     = np.asarray(f["yi"])           # (Np, S, T)
    # C_ii from particle ensemble
    # yi: (Np, S, T) -> var over particles (axis=0) -> (S, T)
    d["diag_cov"] = np.var(d["yi"], axis=0)
    return d


def principal_angles(run_a: dict, run_b: dict) -> np.ndarray:
    """Return principal angles (degrees) shape (S, T) between mode subspaces.

    Stacks u and v velocity components to form a full velocity-space inner
    product.  For each time index t, the cross-Gram G = U_a @ U_b.T is
    formed (S x S), and SVD singular values give cos(theta_k).
    """
    mu = run_a["mode_u"]       # (S, Ny, Nx, T)
    mv = run_a["mode_v"]
    mu2 = run_b["mode_u"]
    mv2 = run_b["mode_v"]

    S, Ny, Nx, T = mu.shape
    N = Ny * Nx

    # flatten spatial: (S, 2*N, T)
    A = np.concatenate([mu.reshape(S, N, T), mv.reshape(S, N, T)], axis=1)
    B = np.concatenate([mu2.reshape(S, N, T), mv2.reshape(S, N, T)], axis=1)

    angles = np.empty((S, T))
    for t in range(T):
        # row-normalise each mode vector before forming Gram (numerics)
        Ua = A[:, :, t]     # (S, 2N)
        Ub = B[:, :, t]
        na = np.linalg.norm(Ua, axis=1, keepdims=True)
        nb = np.linalg.norm(Ub, axis=1, keepdims=True)
        na[na == 0] = 1.0
        nb[nb == 0] = 1.0
        Ua = Ua / na
        Ub = Ub / nb
        G = Ua @ Ub.T       # (S, S)
        sigma = np.linalg.svd(G, compute_uv=False)
        sigma = np.clip(sigma, -1.0, 1.0)
        angles[:, t] = np.degrees(np.arccos(sigma))

    return angles             # (S, T)


def rms_mean_diff(run_a: dict, run_b: dict) -> np.ndarray:
    """RMS pointwise difference of mean u-velocity over time. Shape: (T,)."""
    diff = run_a["u"] - run_b["u"]     # (Ny, Nx, T)
    Ny, Nx, T = diff.shape
    return np.sqrt(np.mean(diff**2, axis=(0, 1)))   # (T,)


def main():
    ap = argparse.ArgumentParser(
        description="Convergence diagnostics for two DO runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("h5_files", nargs=2, type=Path,
                    help="Exactly two out.h5 files (run A then run B)")
    ap.add_argument("--labels", nargs=2, default=None,
                    metavar=("A", "B"), help="Short label per run")
    ap.add_argument("--out", type=Path, default=Path("do_convergence.pdf"))
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    labels = list(args.labels or [])
    while len(labels) < 2:
        labels.append(_label(args.h5_files[len(labels)]))

    print(f"Loading runs...")
    ra = load_run(args.h5_files[0])
    rb = load_run(args.h5_files[1])
    print(f"  [{labels[0]}]: {args.h5_files[0]}")
    print(f"  [{labels[1]}]: {args.h5_files[1]}")

    t = ra["t"]
    S = ra["mode_u"].shape[0]

    print("Computing principal angles...")
    angles = principal_angles(ra, rb)       # (S, T)

    print("Computing RMS mean diff...")
    rms_diff = rms_mean_diff(ra, rb)        # (T,)

    # --- Figure ---
    fig = plt.figure(figsize=(14, 12))
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.40)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])

    # Panel 0: principal angles
    for s in range(S):
        ax0.plot(t, angles[s, :],
                 color=_MODE_COLORS[s % len(_MODE_COLORS)],
                 lw=1.5, alpha=0.85, label=f"angle {s}")
    ax0.axhline(0,  color="k", lw=0.5, ls="--", alpha=0.4)
    ax0.axhline(90, color="k", lw=0.5, ls="--", alpha=0.4)
    ax0.set_xlabel("time", fontsize=10)
    ax0.set_ylabel("angle [deg]", fontsize=10)
    ax0.set_ylim(-5, 95)
    ax0.set_title(
        f"Principal angles between mode subspaces  [{labels[0]}] vs [{labels[1]}]  "
        f"(0=converged, 90=orthogonal)",
        fontsize=9,
    )
    ax0.legend(fontsize=7, ncol=S, loc="upper right", framealpha=0.8)
    ax0.grid(True, alpha=0.18)
    ax0.tick_params(labelsize=8)

    # Panel 1: per-mode variance semilogy
    runs   = [ra, rb]
    for s in range(S):
        for r, ls in zip(runs, _RUN_STYLES):
            ax1.plot(
                r["t"], r["diag_cov"][s, :],
                color=_MODE_COLORS[s % len(_MODE_COLORS)],
                ls=ls, lw=1.2, alpha=0.85,
            )
    # Legend: modes (colour patches) + runs (linestyle proxies)
    mode_handles = [
        mpatches.Patch(color=_MODE_COLORS[s % len(_MODE_COLORS)],
                       label=f"mode {s}")
        for s in range(S)
    ]
    run_handles = [
        mlines.Line2D([], [], color="k", ls=_RUN_STYLES[i % len(_RUN_STYLES)],
                      lw=1.5, label=labels[i])
        for i in range(2)
    ]
    leg1 = ax1.legend(handles=mode_handles, fontsize=6, loc="upper right",
                      framealpha=0.8, title="mode", title_fontsize=6)
    ax1.add_artist(leg1)
    ax1.legend(handles=run_handles, fontsize=6, loc="lower left",
               framealpha=0.8, title="run", title_fontsize=6)
    ax1.set_yscale("log")
    ax1.set_xlabel("time", fontsize=10)
    ax1.set_ylabel(r"$C_{ii}(t)$", fontsize=10)
    ax1.set_title("Per-mode variance  C_ii(t) = var_p(Y_{i,p})", fontsize=9)
    ax1.grid(True, alpha=0.18, which="both")
    ax1.tick_params(labelsize=8)

    fig.suptitle(
        f"DO subspace convergence  |  [{labels[0]}] vs [{labels[1]}]",
        fontsize=12, fontweight="bold",
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
