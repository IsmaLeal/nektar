""" Draws the grid of histograms of one quantity: rows S, columns Np; in every cell the DNS (the house curve
colour) and one DO curve per eps, as step outlines of equal area (densities). The columns are named Np above,
the rows S at the left, the numbers stand on the bottom row only and the x name once under it (every cell has the
same axes; densities are compared, not read: the y axis carries no numbers), the curves are named once, above
the first column, each after a sample of how it is drawn.

grid_hists.py writes the data needed by this script and calls it: python3 grid_hists.py [--q M ...].

Run with the python of the divulg venv (manim + classic_isma); the .cmd next to the image holds the command line.

By hand, on data written by grid_hists.py:
    ~/todisimo/despojos/divulg/.venv/bin/python grid_hists_still.py /tmp/data \
        --out cases/se/figures/Re100_sig0.0144/vary_S_Np_eps/hist_M.png
"""
import argparse
import os
import shlex
import shutil
import sys
import tempfile
from pathlib import Path
import numpy as np
from manim import *
from classic_isma import *

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("data", help="prefix of the .npz written by grid_hists.py")
ap.add_argument("--out", type=Path, required=True, help="the image (.png); the .cmd goes next to it")
args = ap.parse_args()

d = np.load(args.data + ".npz")
Q, H = str(d["q"]), float(d["h"])
SS, NPS, EPS = [int(s) for s in d["S"]], [int(n) for n in d["Np"]], [float(e) for e in d["eps"]]
DNS = (d["dns_edges"], d["dns_dens"])
HAS_DNS = DNS[1].size > 0                                       # Y_1 is the DO's alone
DO = d["do_edges"], d["do_dens"]                                 # (nS, nNp, nEps, nb + 1) and (nS, nNp, nEps, nb)
XNAME = {"M": "$M$", "A": "$A$", "Y1": "$Y_1$", "dL": "$\\Delta L$",
         "escape_A": f"escape time on $A$ (crossings of $\\pm {H:.3f}$)",
         "escape_M": f"escape time on $M$ (crossings of $\\pm {H:.4f}$)",
         "escape_Y1": f"escape time on $Y_1$ (crossings of $\\pm {H:.3f}$)",
         "escape_dL": f"escape time on $\\Delta L$ (crossings of $\\pm {H:.3f}$)"}[Q]
COLOURS = [CURVE, CORAL, SALVIA, MALVA]                         # the DNS, then one eps after another
DNS_WIDTH, DO_WIDTH = 2.0, 3.2                                  # uniform strokes (nib=False): a step outline has no nib
# the three curves nearly coincide, so each has its own texture: the DNS a hatched area under its outline, the first
# eps a solid line, the next ones dashed (dashes and gaps in scene units)
DASH, GAP = 0.2, 0.1

FRAME_H, FRAME_W = 13.0, 16.0  # a frame a little taller than 4:3: three rows of cells, the names above them
CELL_W, CELL_H = 4.0, 2.5      # scene length of the x range and of the y range of one cell
COL_X = [-7.2, -2.1, 3.0]      # scene x of each column's left end (X_LO)
ROW_Y = [2.2, -1.7, -5.6]      # scene y of each row's x axis
NAMES_Y, LEGEND_Y = 6.0, 5.5   # scene y of the columns' names and of the curves' names
assert len(SS) <= len(ROW_Y) and len(NPS) <= len(COL_X), "the frame holds 3 x 3 cells"
NUM_SIZE, NAME_SIZE = 20, 28


def nice(span, n=4):
    """a round tick step giving about n intervals over span"""
    raw = span / n
    k = 10 ** np.floor(np.log10(raw))
    return k * min((m for m in (1, 2, 2.5, 5, 10) if m * k >= raw), key=lambda m: m)


# the ranges, shared by every cell: x over the edges of every curve, rounded outwards to a fine step, with ticks at
# the round values inside; y from 0 past the tallest bin
X_LO, X_HI = min(DNS[0][0], DO[0].min()) if HAS_DNS else DO[0].min(), max(DNS[0][-1], DO[0].max()) if HAS_DNS else DO[0].max()
FINE, X_STEP = nice(X_HI - X_LO, 20), nice(X_HI - X_LO)
X_LO, X_HI = np.floor(X_LO / FINE) * FINE, np.ceil(X_HI / FINE) * FINE
X_TICKS = np.arange(np.ceil(X_LO / X_STEP) * X_STEP, X_HI + X_STEP / 2, X_STEP)
Y_HI = 1.1 * max(DNS[1].max() if HAS_DNS else 0.0, DO[1].max())


def number(v):
    """a tick's number, as formula text"""
    return f"{v:g}" if abs(v) >= 1e-3 or v == 0 else f"{v:.0e}".replace("e-0", r"\times 10^{-").replace("e-", r"\times 10^{-") + "}"


def cell_axes(i, j):
    """the axes of row i, column j: x from X_LO to X_HI and a little past, y from 0 to Y_HI and a little past; the
    axes cross at (0, 0), as axes do, so the y axis stands inside the cell when X_LO < 0; ticks on x only"""
    xpad, ypad = 0.9 * (X_HI - X_LO) / CELL_W, 0.08 * Y_HI          # past the range: the tip (0.35) and a clear stretch
    ax = MyAxes([X_LO, X_HI + xpad], [0, Y_HI + ypad],
                x_length=CELL_W * (X_HI + xpad - X_LO) / (X_HI - X_LO), y_length=CELL_H * (Y_HI + ypad) / Y_HI)
    ax.shift(np.array([COL_X[j], ROW_Y[i], 0]) - ax.c2p(X_LO, 0))
    return VGroup(ax, *(ax.x_axis.get_tick(v, 0.05) for v in X_TICKS))


def corners(ax, edges, dens):
    """the step outline of one histogram as scene points: up from the x axis at the first edge, across every bin,
    down at the last; a corner repeated (equal neighbouring bins) is dropped"""
    xs = np.repeat(edges, 2)[1:-1]
    ys = np.repeat(dens, 2)
    pts = np.array([ax.c2p(edges[0], 0)] + [ax.c2p(x, y) for x, y in zip(xs, ys)] + [ax.c2p(edges[-1], 0)])
    keep = np.r_[True, np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-9]
    return pts[keep]


def outline(pts, color, width):
    """a step outline drawn solid"""
    return curve(VMobject().set_points_as_corners(pts), color=color, width=width, nib=False, n=40 * len(pts))


def dashed(pts, color, width, dash=DASH, gap=GAP):
    """a step outline drawn dashed: every dash a short solid piece of it, by arc length"""
    seg = np.diff(pts, axis=0)
    L = np.linalg.norm(seg, axis=1)
    s = np.r_[0.0, np.cumsum(L)]

    def at(u):
        k = min(int(np.searchsorted(s, u, side="right")) - 1, len(L) - 1)
        return pts[k] + (u - s[k]) / L[k] * seg[k]

    g = VGroup()
    a = 0.0
    while a < s[-1]:
        b = min(a + dash, s[-1])
        piece = [at(a)] + [pts[k] for k in range(len(pts)) if a < s[k] < b] + [at(b)]
        g.add(curve(VMobject().set_points_as_corners(piece), color=color, width=width, nib=False, n=8 * len(piece)))
        a += dash + gap
    return g


def numbers(ax, i):
    """the x ticks' numbers, on the bottom row (densities are compared, not read: the y axis carries none)"""
    g = VGroup()
    if i == len(SS) - 1:
        for v in X_TICKS:
            g.add(formula(number(v), size=NUM_SIZE).next_to(ax.c2p(v, 0), DOWN, buff=0.12))
    return g


def swatch(k):
    """a sample of curve k as it is drawn: a hatched box for the DNS, a solid or a dashed piece of line for an eps"""
    if k == 0:
        box = np.array([[0, 0, 0], [0, 0.26, 0], [0.5, 0.26, 0], [0.5, 0, 0], [0, 0, 0]])
        return VGroup(hatch(box, color=COLOURS[0]), outline(box, COLOURS[0], DNS_WIDTH))
    line = np.array([[0, 0, 0], [0.6, 0, 0]])
    return outline(line, COLOURS[k], DO_WIDTH) if k == 1 else dashed(line, COLOURS[k], DO_WIDTH)


def legend():
    """the curves' names in their colours, each after a sample of its curve, one row above the first column"""
    names = [(0, label("DNS", size=NUM_SIZE, color=COLOURS[0]))] if HAS_DNS else []
    for k, (eps, c) in enumerate(zip(EPS, COLOURS[1:]), 1):
        names.append((k, label(f"DO, $\\varepsilon = 10^{{{int(round(np.log10(eps)))}}}$", size=NUM_SIZE, color=c)))
    g = VGroup(*(VGroup(swatch(k), n).arrange(RIGHT, buff=0.15) for k, n in names)).arrange(RIGHT, buff=0.5)
    return g.move_to([COL_X[0], LEGEND_Y, 0], aligned_edge=LEFT)


class GridHists(Scene):
    def construct(self):
        self.add(grid())
        for i, S in enumerate(SS):
            for j, Np in enumerate(NPS):
                axes = cell_axes(i, j)
                ax = axes[0]
                self.add(axes, numbers(ax, i))
                if HAS_DNS:
                    pts = corners(ax, *DNS)
                    self.add(hatch(pts, color=COLOURS[0]), outline(pts, COLOURS[0], DNS_WIDTH))
                for k, c in enumerate(COLOURS[1:len(EPS) + 1]):
                    pts = corners(ax, DO[0][i, j, k], DO[1][i, j, k])
                    self.add(outline(pts, c, DO_WIDTH) if k == 0 else dashed(pts, c, DO_WIDTH))
                xmid = (ax.c2p(X_LO, 0)[0] + ax.c2p(X_HI, 0)[0]) / 2
                if i == 0:
                    self.add(formula(f"N_p = {Np}", size=NAME_SIZE).move_to([xmid, NAMES_Y, 0]))
                if j == 0:
                    self.add(formula(f"S = {S}", size=NAME_SIZE).rotate(PI / 2)
                             .move_to([COL_X[0] - 0.6, (ax.c2p(X_LO, 0)[1] + ax.c2p(X_LO, Y_HI)[1]) / 2, 0]))
        self.add(legend(), label(XNAME, size=NUM_SIZE + 2).next_to(ax.c2p(X_LO, 0), DOWN, buff=0.45)
                 .set_x(COL_X[len(NPS) // 2] + CELL_W / 2))             # the x name once, under the middle column


args.out.parent.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory() as media, tempconfig({"quality": "high_quality", "media_dir": media, "save_last_frame": True,
                                                         "pixel_height": 1560, "pixel_width": 1920,
                                                         "frame_height": FRAME_H, "frame_width": FRAME_W}):
    scene = GridHists()
    scene.render()
    shutil.move(scene.renderer.file_writer.image_file_path, args.out)
args.out.with_suffix(".cmd").write_text(f"cd {os.getcwd()}\n{sys.executable} {shlex.join(sys.argv)}\n")
