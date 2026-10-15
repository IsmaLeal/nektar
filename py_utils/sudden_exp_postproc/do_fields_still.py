""" Draws phi (steady) and mode K of a DO run on the extraction plane, x from -5 to 15, and how close the two
stay along the run, one still:

    phi_u, phi_v    the two components of phi
    u_K, v_K        the two components of mode K at one instant
    below           <u_K, phi>^2 and <u_K - phi, u_K - phi>^2 against t, every instant of measures.h5, on a log scale

one colour scale for the four channels (both fields have unit norm). < , > is the inner product on the plane of
extract_lib.py; <u_K - phi, u_K - phi> = <u_K, u_K> - 2 <u_K, phi> + <phi, phi>.

do_fields.py writes the data needed by this script and calls it: python3 do_fields.py [RUN ...].

Run with the python of the divulg venv (manim + classic_isma); the .cmd next to the image holds the command line.

By hand, on data written by do_fields.data():
    ~/todisimo/despojos/divulg/.venv/bin/python do_fields_still.py /tmp/data \
        --out cases/se/figures/Re100_sig0.0144/do_Re100_sig0.0144_S6_Np1000_seed45_eps0.0001/phi_u1.png
"""
import argparse
import os
import re
import shlex
import shutil
import sys
import tempfile
from pathlib import Path
import numpy as np
from manim import *
from classic_isma import *

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("data", help="prefix of the .npz written by do_fields.data()")
ap.add_argument("--out", type=Path, required=True, help="the image (.png); the .cmd goes next to it")
args = ap.parse_args()

d = np.load(args.data + ".npz")
XMIN, XMAX = -5.0, 15.0                                         # the window of every field figure (plot_plane.py)
j = (d["x"] >= XMIN) & (d["x"] <= XMAX)
INSIDE = d["inside"][:, j]
K, T_K = int(d["k"]), float(d["t"])
FIELDS = [(r"\phi_u", d["phi_u"]), (r"\phi_v", d["phi_v"]), (rf"u_{K}", d["u"]), (rf"v_{K}", d["v"])]
FIELDS = [(s, np.where(INSIDE, f[:, j], 0.0)) for s, f in FIELDS]
TS = d["ts"]
NORMS = [(rf"\langle \mathbf{{u}}_{K}, \boldsymbol{{\phi}} \rangle^2", d["a"] ** 2, CURVE),
         (rf"\langle \mathbf{{u}}_{K} - \boldsymbol{{\phi}},\, \mathbf{{u}}_{K} - \boldsymbol{{\phi}} \rangle^2",
          (d["g"] - 2 * d["a"] + d["pp"]) ** 2, CORAL)]

FRAME_H = 12.0        # the still is taller than a 16:9 frame: the channels, and the axes below them
S = 0.45              # scene units per unit of x and of y (true aspect)
X_LEFT = -4.9         # scene x of x = XMIN
X0 = X_LEFT - XMIN * S  # scene x of x = 0: the expansion
ROWS = [4.4, 2.8, 1.0, -0.6]   # scene y of each channel's axis: phi_u, phi_v, then u_K, v_K
F_MAX = 0.4           # the colour runs from the background to its full colour at +-F_MAX (the four fields' extremes are 0.39)
POS, NEG = SALVIA, MALVA    # f > 0, f < 0
# the axes below: log scale from 10^LO to 10^HI, the same 10^-6..1 in every still (every run of 2026-10-13 fits:
# 4.9e-6..1), widened by whole decades only for a run outside it
LO = min(-6, int(np.floor(np.log10(min(f.min() for _, f, _ in NORMS)))))
HI = max(0, int(np.ceil(np.log10(max(f.max() for _, f, _ in NORMS)))))
AX_Y = -5.3           # scene y of their t axis (10^LO)
AX_H = 3.0            # scene height of the HI - LO decades


def field_point(x, y, row):
    """scene point of a point (x, y) of the channel drawn at scene height row"""
    return np.array([X0 + x * S, row + y * S, 0.0])


_BG, _POS, _NEG = (np.array(color_to_rgb(c)) for c in (BG, POS, NEG))


def colours(f):
    """RGBA bytes of a field: the background at 0, towards POS (f > 0) or NEG (f < 0); row 0 is the top"""
    a = np.clip(f / F_MAX, 0, 1)[..., None]
    b = np.clip(-f / F_MAX, 0, 1)[..., None]
    rgb = _BG + a * (_POS - _BG) + b * (_NEG - _BG)
    out = np.empty(f.shape + (4,), dtype=np.uint8)
    out[..., :3] = np.round(255 * rgb)
    out[..., 3] = 255
    return out[::-1]


def walls(row):
    """channel's outline: inlet from x = XMIN, step at x = 0, walls at y = +-1.5 up to x = XMAX"""
    top = [field_point(x, y, row) for x, y in [(XMIN, 0.5), (0, 0.5), (0, 1.5), (XMAX, 1.5)]]
    bot = [field_point(x, -y, row) for x, y in [(XMIN, 0.5), (0, 0.5), (0, 1.5), (XMAX, 1.5)]]
    return VGroup(*(curve(VMobject().set_points_as_corners(p), color=INK, width=2, nib=False) for p in (top, bot)))


def steps(row):
    """the solid beside the inlet, above and below it (with stripes)"""
    return VGroup(*(hatch([field_point(XMIN, s * 0.5, row), field_point(0, s * 0.5, row), field_point(0, s * 1.5, row),
                           field_point(XMIN, s * 1.5, row)]) for s in (1, -1)))


def channel(f, row):
    """the field f drawn in the channel at scene height row"""
    m = ImageMobject(colours(f)).stretch_to_fit_width((XMAX - XMIN) * S).stretch_to_fit_height(3 * S)
    return m.move_to(field_point((XMIN + XMAX) / 2, 0, row))


def colour_bar():
    """the colour key at the channels' right, half way down them: from -F_MAX to F_MAX"""
    h = 3.4
    ff = np.linspace(-F_MAX, F_MAX, 256)[:, None] * np.ones((1, 8))
    bar = ImageMobject(colours(ff)).stretch_to_fit_height(h).stretch_to_fit_width(0.14)
    bar.move_to([field_point(XMAX, 0, 0)[0] + 0.5, (ROWS[0] + ROWS[-1]) / 2, 0])
    y_of = lambda f: bar.get_bottom()[1] + h * (f + F_MAX) / (2 * F_MAX)
    ticks = VGroup(*(formula(s, size=24).next_to([bar.get_right()[0], y_of(v), 0], RIGHT, buff=0.1)
                     for s, v in ((r"0.4", F_MAX), (r"0", 0.0), (r"-0.4", -F_MAX))))
    return Group(bar, ticks)


def run_label():
    """Re, sigma, S, N_p, eps of the run and the instant of the mode, in one line"""
    m = re.match(r"do_Re(\d+)_sig([0-9.]+)_S(\d+)_Np(\d+)_seed\d+(?:_eps([0-9.]+))?", str(d["run"]))
    Re, sig, S_, Np, eps = m.groups()
    e = int(round(np.log10(float(eps) if eps else 0.01)))
    return label(rf"$Re = {Re}$, $\sigma = {sig}$, $S = {S_}$, $N_p = {Np}$, $\varepsilon = 10^{{{e}}}$, "
                 rf"mode at $t = {T_K:g}$", size=24)


def norms():
    """the axes below the channels: t as wide as a channel, log10 of each quantity - LO up; its curves (a thin
    uniform line, every instant: the pen's nib smears a series this dense) and their labels"""
    t_end = float(TS[-1])
    ax = MyAxes([0, 1.05 * t_end, t_end / 4], [0, HI - LO + 0.9, 1], origin=[X_LEFT, AX_Y, 0],
                x_length=(XMAX - XMIN) * S, y_length=AX_H * (HI - LO + 0.9) / (HI - LO),
                axis_config={"include_ticks": True, "tick_size": 0.06})
    o, ux, uy = ax.c2p(0, 0), ax.c2p(1, 0) - ax.c2p(0, 0), ax.c2p(0, 1) - ax.c2p(0, 0)
    g = VGroup(ax)
    for k in range(HI - LO + 1):
        e = k + LO
        g.add(formula({0: "1", 1: "10"}.get(e, rf"10^{{{e}}}"), size=22).next_to(ax.c2p(0, k), LEFT, buff=0.15))
    for t in np.arange(5) * t_end / 4:
        g.add(formula(f"{t:g}", size=22).next_to(ax.c2p(t, 0), DOWN, buff=0.15))
    g.add(formula("t").next_to(ax.x_axis.get_end(), DOWN, buff=0.2))
    for tex, f, c in NORMS:
        p = o + TS[:, None] * ux + (np.log10(f) - LO)[:, None] * uy
        g.add(curve(p, color=c, nib=False, width=TRACE_WIDTH, n=4 * len(TS)),
              formula(tex, size=24, color=c).next_to(p[-1], RIGHT, buff=0.15))
    return g


class DoFields(Scene):
    def construct(self):
        self.add(grid())
        for (sym, f), row in zip(FIELDS, ROWS):
            self.add(channel(f, row), walls(row), steps(row),
                     formula(sym, size=30).next_to(field_point(XMIN, 0, row), LEFT, buff=0.3))
        self.add(colour_bar(), run_label().next_to(field_point(XMAX, 1.5, ROWS[0]), UP, buff=0.25, aligned_edge=RIGHT))
        self.add(norms())


args.out.parent.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory() as media, tempconfig({"quality": "high_quality", "media_dir": media, "save_last_frame": True,
                                                         "pixel_height": 1620, "frame_height": FRAME_H,
                                                         "frame_width": FRAME_H * 1920 / 1620}):
    scene = DoFields()
    scene.render()
    shutil.move(scene.renderer.file_writer.image_file_path, args.out)
args.out.with_suffix(".cmd").write_text(f"cd {os.getcwd()}\n{sys.executable} {shlex.join(sys.argv)}\n")
