#!/usr/bin/env python3
"""
Plot the K vector fields g_k(x,y) declared in a Nektar case XML's
<FUNCTION NAME="ForcingChannels"> block. Each channel becomes one panel:
filled colormap of |g_k|(x,y) + unit-length quiver arrows colored by |g_k|.

Usage:
    python3 py_utils/plot_forcing_channels.py --xml <case.xml> --out <png>
    [--data <out.h5>]                     # infer x/y extent from h5; else use --xlim/--ylim
    [--xlim XMIN XMAX] [--ylim YMIN YMAX] # default 0..2*pi
    [--n 200]                             # grid resolution per axis
    [--quiv-step 12]                      # quiver stride
"""
from __future__ import annotations
import argparse, re, sys, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


_VAR_RE = re.compile(r"^g(\d+)_([a-z])$")
_NS = {"pi": np.pi, "sin": np.sin, "cos": np.cos, "tan": np.tan,
       "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "abs": np.abs}


def _eval(expr: str, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Evaluate a Nektar expression string on (x, y, z) meshgrids."""
    py = expr.replace("^", "**").replace("PI", "pi")
    return eval(py, {"__builtins__": None}, dict(_NS, x=x, y=y, z=z))


def _read_channels(xml_path: Path) -> dict[int, dict[str, str]]:
    """Return {k: {'u': expr, 'v': expr, 'w': expr|absent}} from the XML block."""
    root = ET.parse(xml_path).getroot()
    fn = root.find(".//FUNCTION[@NAME='ForcingChannels']")
    if fn is None:
        sys.exit(f"No <FUNCTION NAME='ForcingChannels'> block found in {xml_path}")
    chans: dict[int, dict[str, str]] = {}
    for e in fn.findall("E"):
        var, val = e.get("VAR", ""), e.get("VALUE", "")
        m = _VAR_RE.match(var.strip())
        if not m:
            continue
        chans.setdefault(int(m.group(1)), {})[m.group(2)] = val
    if not chans:
        sys.exit(f"No g{{k}}_{{u,v,w}} entries found inside ForcingChannels.")
    return dict(sorted(chans.items()))


def _domain(args) -> tuple[float, float, float, float]:
    if args.data is not None:
        import h5py
        with h5py.File(args.data, "r") as f:
            x = np.array(f["x"]); y = np.array(f["y"])
        return float(x.min()), float(x.max()), float(y.min()), float(y.max())
    return args.xlim[0], args.xlim[1], args.ylim[0], args.ylim[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xml", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--xlim", nargs=2, type=float, default=[0.0, 2*np.pi])
    ap.add_argument("--ylim", nargs=2, type=float, default=[0.0, 2*np.pi])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--quiv-step", type=int, default=12)
    ap.add_argument("--cmap", default="viridis")
    args = ap.parse_args()

    chans = _read_channels(args.xml)
    K = len(chans)
    xmin, xmax, ymin, ymax = _domain(args)
    xs = np.linspace(xmin, xmax, args.n)
    ys = np.linspace(ymin, ymax, args.n)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    Z = np.zeros_like(X)

    # layout: 1 row of K panels (or 2 rows if K > 4)
    cols = K if K <= 4 else 4
    rows = (K + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.4*cols, 4.0*rows),
                             constrained_layout=True, squeeze=False)
    qs = max(1, int(args.quiv_step))
    for idx, (k, comp) in enumerate(chans.items()):
        ax = axes[idx // cols][idx % cols]
        u = _eval(comp.get("u", "0"), X, Y, Z)
        v = _eval(comp.get("v", "0"), X, Y, Z)
        mag = np.sqrt(u*u + v*v)
        im = ax.imshow(mag, origin="lower", extent=[xmin, xmax, ymin, ymax],
                       cmap=args.cmap, aspect="equal")
        # unit-length arrows colored by |g_k|, suppressed where the field is
        # negligible (alpha-mask < 1% of max) so the colormap still tells the
        # magnitude story without burying the arrows in background-colored noise
        Xq, Yq = X[::qs, ::qs], Y[::qs, ::qs]
        Uq, Vq, Mq = u[::qs, ::qs], v[::qs, ::qs], mag[::qs, ::qs]
        thresh = 0.01 * float(mag.max() if mag.max() > 0 else 1.0)
        keep = Mq > thresh
        if keep.any():
            safe = np.where(Mq > 1e-14, Mq, 1.0)
            Un, Vn = Uq/safe, Vq/safe
            ax.quiver(Xq[keep], Yq[keep], Un[keep], Vn[keep], Mq[keep],
                      cmap=args.cmap, pivot="mid",
                      scale=28, width=0.005, edgecolor="k", linewidth=0.4)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label=r"$|g_k|$")
        ax.set_title(f"channel {k}")
        ax.set_xlabel("x"); ax.set_ylabel("y")
    # hide any unused axes
    for j in range(K, rows*cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.suptitle(f"ForcingChannels  (K = {K},  domain = "
                 f"[{xmin:.2f}, {xmax:.2f}] × [{ymin:.2f}, {ymax:.2f}])",
                 fontsize=12)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
