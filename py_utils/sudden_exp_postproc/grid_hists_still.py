""" Draws the grid of histograms of one quantity: rows S, columns Np; in every cell the DNS (the house curve
colour) and one DO curve per eps, as step outlines of equal area (densities), every one solid and of the same
width, the DO curves drawn over the DNS. The columns are named Np above,
the rows S at the left, the numbers stand on the bottom row only and the x name once under it (every cell has the
same axes; densities are compared, not read: the y axis carries no numbers), the curves are named once, above
the first column, each after a sample of how it is drawn.

grid_hists.py writes the data needed by this script and calls it: python3 grid_hists.py [--q M ...].

Run with the python of the divulg venv (manim + classic_isma); the .cmd next to the image holds the command line.

By hand, on data written by grid_hists.py:
    python3 grid_hists_still.py /tmp/data \
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
ap.add_argument("--logy", action="store_true", help="y-axis in log10")
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
# all solid, all the same thin width: the DNS a hatched area under its outline, then one eps after another on top,
# each showing where it leaves the others. Uniform strokes (nib=False): a step outline has no nib
DNS_WIDTH, DO_WIDTHS = 2.0, (2.0, 2.0)

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
Y_LO = 10 ** np.floor(np.log10(min(DNS[1][DNS[1] > 0].min() if HAS_DNS else np.inf, DO[1][DO[1] > 0].min())))
L_LO, L_HI = np.log10(Y_LO), np.log10(Y_HI)
Y0 = Y_LO if args.logy else 0.0


class LogAxes(MyAxes):
    def get_origin(self):
        return self.c2p(self.x_axis.x_min, self.y_axis.x_min)


def number(v):
    """a tick's number, as formula text"""
    return f"{v:g}" if abs(v) >= 1e-3 or v == 0 else f"{v:.0e}".replace("e-0", r"\times 10^{-").replace("e-", r"\times 10^{-") + "}"


def cell_axes(i, j):
    """the axes of row i, column j: x from X_LO to X_HI and a little past, y from 0 to Y_HI and a little past; the
    axes cross at (0, 0), as axes do, so the y axis stands inside the cell when X_LO < 0; ticks on x only"""
    xpad = 0.9 * (X_HI - X_LO) / CELL_W          # past the range: the tip (0.35) and a clear stretch
    xl = CELL_W * (X_HI + xpad - X_LO) / (X_HI - X_LO)
    if args.logy:
        ypad = 0.08 * (L_HI - L_LO)
        ax = LogAxes([X_LO, X_HI + xpad], [L_LO, L_HI + ypad], y_axis_config={"scaling": LogBase(10)},
                     x_length=xl, y_length=CELL_H * (L_HI + ypad - L_LO) / (L_HI - L_LO))
    else:
        ypad = 0.08 * Y_HI
        ax = MyAxes([X_LO, X_HI + xpad], [0, Y_HI + ypad],
                    x_length=xl, y_length=CELL_H * (Y_HI + ypad) / Y_HI)
    ax.shift(np.array([COL_X[j], ROW_Y[i], 0]) - ax.c2p(X_LO, Y0))
    return VGroup(ax, *(ax.x_axis.get_tick(v, 0.05) for v in X_TICKS))


def corners(ax, edges, dens):
    """the step outline of one histogram as scene points: up from the x axis at the first edge, across every bin,
    down at the last; a corner repeated (equal neighbouring bins) is dropped"""
    xs = np.repeat(edges, 2)[1:-1]
    ys = np.repeat(np.maximum(dens, Y0), 2)
    pts = np.array([ax.c2p(edges[0], Y0)] + [ax.c2p(x, y) for x, y in zip(xs, ys)] + [ax.c2p(edges[-1], Y0)])
    keep = np.r_[True, np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-9]
    return pts[keep]


def outline(pts, color, width):
    """a step outline drawn solid"""
    return curve(VMobject().set_points_as_corners(pts), colour=color, width=width, nib=False, n=40 * len(pts))


def numbers(ax, i):
    """the x ticks' numbers, on the bottom row (densities are compared, not read: the y axis carries none)"""
    g = VGroup()
    if i == len(SS) - 1:
        for v in X_TICKS:
            g.add(formula(number(v), size=NUM_SIZE).next_to(ax.c2p(v, Y0), DOWN, buff=0.12))
    return g


def swatch(k):
    """a sample of curve k as it is drawn: a hatched box for the DNS, a piece of line of its width for an eps"""
    if k == 0:
        box = np.array([[0, 0, 0], [0, 0.26, 0], [0.5, 0.26, 0], [0.5, 0, 0], [0, 0, 0]])
        return VGroup(hatch(box, colour=COLOURS[0]), outline(box, COLOURS[0], DNS_WIDTH))
    line = np.array([[0, 0, 0], [0.6, 0, 0]])
    return outline(line, COLOURS[k], DO_WIDTHS[k - 1])


def legend():
    """the curves' names in their colours, each after a sample of its curve, one row above the first column"""
    names = [(0, label("DNS", size=NUM_SIZE, colour=COLOURS[0]))] if HAS_DNS else []
    for k, (eps, c) in enumerate(zip(EPS, COLOURS[1:]), 1):
        # one eps alone: plain "DO" (2026-10-22, the MeGa deck); several: each named by its eps
        text = "DO" if len(EPS) == 1 else f"DO, $\\varepsilon = 10^{{{int(round(np.log10(eps)))}}}$"
        names.append((k, label(text, size=NUM_SIZE, colour=c)))
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
                    self.add(hatch(pts, colour=COLOURS[0]), outline(pts, COLOURS[0], DNS_WIDTH))
                for k, c in enumerate(COLOURS[1:len(EPS) + 1]):
                    pts = corners(ax, DO[0][i, j, k], DO[1][i, j, k])
                    self.add(outline(pts, c, DO_WIDTHS[k]))
                xmid = (ax.c2p(X_LO, Y0)[0] + ax.c2p(X_HI, Y0)[0]) / 2
                if i == 0:
                    self.add(formula(f"N_p = {Np}", size=NAME_SIZE).move_to([xmid, NAMES_Y, 0]))
                if j == 0:
                    self.add(formula(f"S = {S}", size=NAME_SIZE).rotate(PI / 2)
                             .move_to([COL_X[0] - 0.6, (ax.c2p(X_LO, Y0)[1] + ax.c2p(X_LO, Y_HI)[1]) / 2, 0]))
        self.add(legend(), label(XNAME, size=NUM_SIZE + 2).next_to(ax.c2p(X_LO, Y0), DOWN, buff=0.45)
                 .set_x(COL_X[len(NPS) // 2] + CELL_W / 2))             # the x name once, under the middle column


args.out.parent.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory() as media, tempconfig({"quality": "high_quality", "media_dir": media, "save_last_frame": True,
                                                         "pixel_height": 1560, "pixel_width": 1920,
                                                         "frame_height": FRAME_H, "frame_width": FRAME_W}):
    scene = GridHists()
    scene.render()
    shutil.move(scene.renderer.file_writer.image_file_path, args.out)
args.out.with_suffix(".cmd").write_text(f"cd {os.getcwd()}\n{sys.executable} {shlex.join(sys.argv)}\n")
