""" Animates the wall gradients against x, evolving in time, with the u field between them.

    top     -du/dy at y = 1.475, the plane's first row below the top wall (y = 1.5), against x > 0
    middle  the u field on the extraction plane, x from -1 to 40, aligned in x with the two plots
    bottom  du/dy at y = -1.475, the plane's first row above the bottom wall (y = -1.5), against x > 0

dudy_wall_data.py writes the data needed by this script.

Run with the python of the divulg venv (manim + classic_isma); the .cmd next to the movie holds the command line.

Example:
    ~/todisimo/despojos/divulg/.venv/bin/python dudy_wall_movie.py /tmp/dudy_Re100_sig0.0144_dns_seed45 \
        --out cases/se/figures/Re100_sig0.0144/dns_Re100_sig0.0144_seed45/dudy_wall.mp4
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
ap.add_argument("data", help="prefix written by dudy_wall_data.py (--out there)")
ap.add_argument("--out", type=Path, required=True, help="the movie (.mp4); the .cmd goes next to it")
ap.add_argument("--seconds", type=float, default=60, help="video seconds for the whole run")
args = ap.parse_args()

d = np.load(args.data + ".npz")
X, Y, T, XS, G_TOP, G_BOT = d["x"], d["y"], d["t"], d["xs"], d["up"], d["low"]
U = np.load(args.data + "_u.npy", mmap_mode="r")             # (T, ny, nx), x from X[0]

S = 0.28              # scene units per unit of x
STRETCH = 2.0         # the field's y is drawn STRETCH times its x scale (true aspect: 3 high, 41 long)
X_LEFT = -5.9         # scene x of x = X[0]
X0 = X_LEFT - X[0] * S  # scene x of x = 0: the expansion, where the plots' y axes stand
FIELD_Y = 0.0         # scene y of the channel's axis
PLOT_Y = 2.45         # scene y of the plots' zero lines, above and below
G_MAX = 2.3           # the plots' y range: +-G_MAX (the run's extremes are -2.28 and 2.26)
PLOT_H = 2.0          # scene height of +-G_MAX
U_POS, U_NEG = 1.0, 0.25    # the colour runs from the background to its full colour at u = 1 and at u = -0.25
POS, NEG = SALVIA, MALVA    # u > 0, u < 0; the curves keep the house colour (CURVE)
CURVE_N = 1600        # pen samples along a gradient curve


def field_point(x, y):
    """scene point of a point (x, y) of the channel"""
    return np.array([X0 + x * S, FIELD_Y + y * S * STRETCH, 0.0])


def at(a, t):
    """a[k] at time t, linear between snapshots"""
    k = np.clip((t - T[0]) / (T[1] - T[0]), 0, len(T) - 1)
    i = min(int(k), len(T) - 2)
    w = k - i
    return (1 - w) * np.asarray(a[i], dtype=np.float32) + w * np.asarray(a[i + 1], dtype=np.float32)


_BG, _POS, _NEG = (np.array(color_to_rgb(c)) for c in (BG, POS, NEG))


def colours(u):
    """RGBA bytes of a u array: the background at u = 0, towards POS (u > 0) or NEG (u < 0); row 0 is the top"""
    a = np.clip(u / U_POS, 0, 1)[..., None]
    b = np.clip(-u / U_NEG, 0, 1)[..., None]
    rgb = _BG + a * (_POS - _BG) + b * (_NEG - _BG)
    out = np.empty(u.shape + (4,), dtype=np.uint8)
    out[..., :3] = np.round(255 * rgb)
    out[..., 3] = 255
    return out[::-1]


def plot_axes(y0):
    """one gradient plot: x as the field's, zero line at scene height y0, its y axis at the expansion"""
    ax = MyAxes([X[0], X[-1] + 1.2], [-G_MAX, G_MAX + 0.25], origin=[X0, y0, 0],
                x_length=(X[-1] + 1.2 - X[0]) * S, y_length=(2 * G_MAX + 0.25) * PLOT_H / (2 * G_MAX),
                x_axis_config={'tip_height': 0.35}, y_axis_config={'tip_height': 0.26})
    return ax


def gradient_points(ax, g):
    o, ux, uy = ax.c2p(0, 0), ax.c2p(1, 0) - ax.c2p(0, 0), ax.c2p(0, 1) - ax.c2p(0, 0)
    return o + XS[:, None] * ux + np.asarray(g)[:, None] * uy


def zero_drop(ax, g, wall_y):
    """pencil from first 0 of g on the plot x to the u panel's wall at y=wall_y"""
    i = np.flatnonzero((np.diff(np.sign(g)) != 0) & (XS[:-1] > 0.5))[0]                  # 1st sign change, between XS[i] and XS[i+1]
    x0 = XS[i] - g[i] * (XS[i + 1] - XS[i]) / (g[i + 1] - g[i]) # interpolate where the 0 is
    return pencil(Line(ax.c2p(x0, 0), field_point(x0, wall_y)))


def walls():
    """
    channel's outline: inlet from x = X[0], step at x = 0, walls at y = +-1.5
    top and bot hold the vertices.
    """
    top = [field_point(*p) for p in [(X[0], 0.5), (0, 0.5), (0, 1.5), (X[-1], 1.5)]]
    bot = [field_point(x, -y) for x, y in [(X[0], 0.5), (0, 0.5), (0, 1.5), (X[-1], 1.5)]]
    return VGroup(*(curve(VMobject().set_points_as_corners(p), color=INK, width=2, nib=False) for p in (top, bot)))


def steps():
    """the solid beside the inlet, above and below it (with stripes)"""
    return VGroup(*(hatch([field_point(X[0], s * 0.5), field_point(0, s * 0.5), field_point(0, s * 1.5),
                           field_point(X[0], s * 1.5)]) for s in (1, -1)))


def colour_bar():
    """the colour key at the field's right: u from -U_NEG to U_POS"""
    h = 3 * S * STRETCH
    uu = np.linspace(U_POS, -U_NEG, 256)[::-1][:, None] * np.ones((1, 8))
    bar = ImageMobject(colours(uu)).stretch_to_fit_height(h).stretch_to_fit_width(0.14)
    bar.next_to(field_point(X[-1], 0), RIGHT, buff=0.3)
    y_of = lambda u: bar.get_bottom()[1] + h * (u + U_NEG) / (U_POS + U_NEG)
    ticks = VGroup(*(formula(s, size=24).next_to([bar.get_right()[0], y_of(v), 0], RIGHT, buff=0.1)
                     for s, v in ((r"1", 1.0), (r"0", 0.0), (r"-0.25", -0.25))))
    name = formula("u", size=30).next_to(bar, UP, buff=0.15)
    return Group(bar, ticks, name)


class DudyWall(MovingCameraScene):
    def construct(self):
        # build grid, axes, channel's outline
        self.wait(0.4)
        g = endless(self) # grid
        top, bot = plot_axes(PLOT_Y), plot_axes(-PLOT_Y) # axis centres at height \pm PLOT_Y
        names = VGroup(formula(r"\left.-\dfrac{\partial u}{\partial y}\right|_{\text{top}}", size=19).next_to(top.y_axis.get_end(), LEFT, buff=0.2),
                       formula(r"\left.\dfrac{\partial u}{\partial y}\right|_{\text{bottom}}", size=19).next_to(bot.y_axis.get_end(), LEFT, buff=0.2),
                       formula("x").next_to(top.x_axis.get_end(), DOWN, buff=0.2),
                       formula("x").next_to(bot.x_axis.get_end(), DOWN, buff=0.2))
        wall, solid = walls(), steps()
        self.play(grid_in(g), AxesIn(top), AxesIn(bot), TextIn(names), CurveIn(wall, head=None), Create(solid))

        # create time, field
        t = ValueTracker(float(T[0]))
        field = ImageMobject(colours(at(U, T[0])))
        field.stretch_to_fit_width((X[-1] - X[0]) * S).stretch_to_fit_height(3 * S * STRETCH)
        field.move_to(field_point((X[0] + X[-1]) / 2, 0))
        key = colour_bar()
        clock_name = formula("t =")
        clock = DecimalNumber(T[0], num_decimal_places=0, group_with_commas=False, font_size=clock_name.font_size, color=INK)
        clock_name.to_corner(UR, buff=0.4).shift(LEFT * 1.0)
        clock.next_to(clock_name, RIGHT, buff=0.15)
        self.add(field)
        
        # initialise curves
        up0 = curve(gradient_points(top, at(G_TOP, T[0])), n=CURVE_N)
        low0 = curve(gradient_points(bot, at(G_BOT, T[0])), n=CURVE_N)
        
        self.bring_to_front(wall, solid)    # otherwise the field is over both
        self.play(FadeIn(field), FadeIn(key), TextIn(VGroup(clock_name, clock)),
                  CurveIn(up0, run_time=1.2), CurveIn(low0, run_time=1.2))
        self.wait(0.5)

        up = always_redraw(lambda: curve(gradient_points(top, at(G_TOP, t.get_value())), n=CURVE_N))
        low = always_redraw(lambda: curve(gradient_points(bot, at(G_BOT, t.get_value())), n=CURVE_N))
        field.add_updater(lambda m: setattr(m, "pixel_array", colours(at(U, t.get_value()))))
        clock.add_updater(lambda m: m.set_value(t.get_value()).next_to(clock_name, RIGHT, buff=0.15))
        self.remove(up0, low0)
        self.add(up, low)
        drops = always_redraw(lambda: VGroup(zero_drop(top, at(G_TOP, t.get_value()), 1.5),
                                             zero_drop(bot, at(G_BOT, t.get_value()), -1.5)))
        self.add(drops)
        self.play(t.animate.set_value(float(T[-1])), run_time=args.seconds, rate_func=linear)
        self.wait(0.5)

        for m in (up, low, field, clock, drops):
            m.clear_updaters()                  # frozen as they are, so the exits can take them away
        self.play(CurveOut(up, run_time=1.2), CurveOut(low, run_time=1.2), AxesOut(top), AxesOut(bot),
                  TextOut(names), TextOut(VGroup(clock_name, clock)), CurveOut(wall, head=None), Uncreate(solid),
                  FadeOut(drops), FadeOut(field), FadeOut(key), grid_out(g, run_time=1.2))
        self.wait(0.5)


args.out.parent.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory() as media, tempconfig({"quality": "high_quality", "media_dir": media}):
    scene = DudyWall()
    scene.render()
    shutil.move(scene.renderer.file_writer.movie_file_path, args.out)
args.out.with_suffix(".cmd").write_text(f"cd {os.getcwd()}\n{sys.executable} {shlex.join(sys.argv)}\n")
