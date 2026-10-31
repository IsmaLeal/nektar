""" The mean, four modes and one realisation of a DO run, evolving in time, on any domain: a 3 by 3 grid that
fills the frame,

    row 1   the mean of u, mode 1, mode 2
    row 2   the realisation of particle 0 (the mean plus every mode times its coefficient), modes 3, 4
    row 3   the two colour keys with the clock; the standard deviation of every coefficient against t, two panels wide

Every field is drawn on the box of render_do_panels.py (--box there), the body inside the box in ink. The mean and
the realisation share one colour scale (u > 0 SALVIA, u < 0 MALVA, the background at 0); the modes share another,
symmetric.

render_do_panels.py writes the data needed by this script and calls it.

Run with the python of the divulg venv (manim + classic_isma); the .cmd next to the movie holds the command line.

Example (the preview; --quality high for the final render):
    python3 do_panels_movie.py RUN/output/py_utils_results/do_panels_data \
        --out RUN/output/py_utils_results/do_panels_u.mp4
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
ap.add_argument("data", help="prefix written by render_do_panels.py (its data step)")
ap.add_argument("--out", type=Path, required=True, help="the movie (.mp4); the .cmd goes next to it")
ap.add_argument("--seconds", type=float, default=None, help="video seconds for the whole run (default: one per time unit)")
ap.add_argument("--quality", default="low", choices=["low", "high"],
                help="low: the preview, 854 by 480 at 15 frames per second (default); high: 1920 by 1080 at 60")
args = ap.parse_args()

d = np.load(args.data + ".npz")
X, Y, T, SIG, Y0, BODY = d["x"], d["y"], d["t"], d["sigma"], d["y0"], d["body"]   # axes, times, (S, T) deviations,
                                                                                   # (S, T) coefficients of particle 0, body mask
U_POS, U_NEG, M_SCALE = float(d["u_pos"]), float(d["u_neg"]), float(d["m_scale"])
U = np.load(args.data + "_u.npy", mmap_mode="r")                        # (T, ny, nx) the mean of u
MODES = np.load(args.data + "_modes.npy", mmap_mode="r")                # (S, T, ny, nx) the modes of u
S = MODES.shape[0]
SHOWN = min(4, S)               # modes drawn: 1 to SHOWN; the realisation and the deviations use all S
SECONDS = args.seconds if args.seconds is not None else float(T[-1] - T[0])

COLS, ROWS = 3, 3
GAP = 0.18                      # scene units between panels
PANEL_W = (config.frame_width - 2 * 0.25 - (COLS - 1) * GAP) / COLS
PANEL_H = (config.frame_height - 2 * 0.25 - (ROWS - 1) * GAP) / ROWS
ASPECT = (X[-1] - X[0]) / (Y[-1] - Y[0])
FIELD_W = min(PANEL_W, PANEL_H * ASPECT)        # the field fills its panel, true aspect
FIELD_H = FIELD_W / ASPECT
CURVE_N = 800
PALETTE = [CURVE, SALVIA, MALVA, CORAL, MOSTAZA, PIZARRA]

_BG, _POS, _NEG, _INK = (np.array(color_to_rgb(c)) for c in (BG, SALVIA, MALVA, INK))


def colours(f, pos, neg):
    """RGBA bytes of a field: the background at 0, the full SALVIA at f = pos and the full MALVA at f = -neg;
    row 0 is the top"""
    a = np.clip(f / pos, 0, 1)[..., None]
    b = np.clip(-f / neg, 0, 1)[..., None]
    rgb = _BG + a * (_POS - _BG) + b * (_NEG - _BG)
    out = np.empty(f.shape + (4,), dtype=np.uint8)
    out[..., :3] = np.round(255 * rgb)
    out[..., 3] = 255
    return out[::-1]


def body_image():
    """the body (points outside the fluid) in ink, transparent elsewhere"""
    out = np.zeros(BODY.shape + (4,), dtype=np.uint8)
    out[..., :3] = np.round(255 * _INK)
    out[..., 3] = 255 * BODY
    return out[::-1]


def panel_centre(k):
    """scene centre of panel k of the 3 by 3 grid, row by row from the top left"""
    r, c = divmod(k, COLS)
    x = -config.frame_width / 2 + 0.25 + PANEL_W / 2 + c * (PANEL_W + GAP)
    y = config.frame_height / 2 - 0.25 - PANEL_H / 2 - r * (PANEL_H + GAP)
    return np.array([x, y, 0.0])


def field_image(f, centre, pos, neg):
    m = ImageMobject(colours(f, pos, neg)).stretch_to_fit_width(FIELD_W).stretch_to_fit_height(FIELD_H)
    return m.move_to(centre)


def colour_bar(pos, neg, h, centre, ticks, name=None):
    """the colour key: a bar h high from -neg (bottom) to pos (top) centred at the scene point centre, its ticks
    (text, value) at the right and, if given, its name above"""
    ff = np.linspace(-neg, pos, 256)[:, None] * np.ones((1, 8))
    bar = ImageMobject(colours(ff, pos, neg)).stretch_to_fit_height(h).stretch_to_fit_width(0.14)
    bar.move_to(centre)
    y_of = lambda v: bar.get_bottom()[1] + h * (v + neg) / (pos + neg)
    marks = VGroup(*(formula(s, size=24).next_to([bar.get_right()[0], y_of(v), 0], RIGHT, buff=0.1) for s, v in ticks))
    parts = [bar, marks] + ([formula(name, size=30).next_to(bar, UP, buff=0.15)] if name else [])
    return Group(*parts)


def at(a, t):
    """a[k] at time t, linear between snapshots"""
    k = np.clip((t - T[0]) / (T[1] - T[0]), 0, len(T) - 1)
    i = min(int(k), len(T) - 2)
    w = k - i
    return (1 - w) * np.asarray(a[i], dtype=np.float32) + w * np.asarray(a[i + 1], dtype=np.float32)


def realisation(t):
    """u of particle 0 at time t: the mean plus every mode times its coefficient"""
    k = np.clip((t - T[0]) / (T[1] - T[0]), 0, len(T) - 1)
    i = min(int(k), len(T) - 2)
    w = k - i
    y = (1 - w) * Y0[:, i] + w * Y0[:, i + 1]
    f = at(U, t)
    for j in range(S):
        f = f + y[j] * at(MODES[j], t)
    return f


def sigma_axes(centre, width):
    """the deviations plot: t along x, sigma up to its maximum, filling a panel width wide"""
    s_max = float(SIG.max()) * 1.15
    ax = MyAxes([T[0], T[-1] + 0.06 * (T[-1] - T[0])], [0, s_max], origin=centre + [-width / 2 + 0.55, -PANEL_H / 2 + 0.45, 0],
                x_length=width - 0.8, y_length=PANEL_H - 0.75,
                x_axis_config={'tip_height': 0.25}, y_axis_config={'tip_height': 0.2})
    return ax


def sigma_points(ax, i, t):
    """the curve of sigma_i up to time t"""
    k = int(np.clip((t - T[0]) / (T[1] - T[0]), 0, len(T) - 1)) + 1
    o, ux, uy = ax.c2p(0, 0), ax.c2p(1, 0) - ax.c2p(0, 0), ax.c2p(0, 1) - ax.c2p(0, 0)
    tt, ss = T[:max(k, 2)], SIG[i, :max(k, 2)]
    return o + tt[:, None] * ux + ss[:, None] * uy


class DOPanels(MovingCameraScene):
    def construct(self):
        self.wait(0.4)
        g = endless(self)
        # panels 0..5: the mean, modes 1 and 2, the realisation, modes 3 and 4
        SLOT = [("mean", None), ("mode", 0), ("mode", 1), ("real", None)] + [("mode", j) for j in range(2, SHOWN)]
        def frame_of(kind, j, t):
            return at(U, t) if kind == "mean" else realisation(t) if kind == "real" else at(MODES[j], t)
        def scale_of(kind):
            return (U_POS, U_NEG) if kind in ("mean", "real") else (M_SCALE, M_SCALE)
        fields = [field_image(frame_of(kind, j, T[0]), panel_centre(k), *scale_of(kind)) for k, (kind, j) in enumerate(SLOT)]
        bodies = Group(*(ImageMobject(body_image()).stretch_to_fit_width(FIELD_W).stretch_to_fit_height(FIELD_H)
                         .move_to(panel_centre(k)) for k in range(len(SLOT))))
        label = {"mean": r"\bar{u}", "real": r"\bar{u} + \textstyle\sum_k Y_k u_k"}
        names = VGroup(*(formula(label.get(kind, rf"u_{j + 1 if j is not None else 0}"), size=24)
                         .move_to(panel_centre(k) + [-FIELD_W / 2 + 0.4, FIELD_H / 2 - 0.3, 0], aligned_edge=LEFT)
                         for k, (kind, j) in enumerate(SLOT)))
        key_c = panel_centre(6)
        keys = Group(colour_bar(U_POS, U_NEG, PANEL_H * 0.5, key_c + [-1.2, -0.45, 0],
                                ((f"{U_POS:.2g}", U_POS), ("0", 0.0), (f"-{U_NEG:.2g}", -U_NEG)), name=r"\bar{u}"),
                     colour_bar(M_SCALE, M_SCALE, PANEL_H * 0.5, key_c + [0.4, -0.45, 0],
                                ((f"{M_SCALE:.2g}", M_SCALE), ("0", 0.0), (f"-{M_SCALE:.2g}", -M_SCALE)), name=r"u_k"))
        clock_name = formula("t =", size=30)
        clock = DecimalNumber(T[0], num_decimal_places=1, group_with_commas=False, font_size=clock_name.font_size, color=INK)
        clock_name.move_to(key_c + [-0.5, PANEL_H / 2 - 0.2, 0])
        clock.next_to(clock_name, RIGHT, buff=0.15)
        ax = sigma_axes((panel_centre(7) + panel_centre(8)) / 2, 2 * PANEL_W + GAP)
        ax_names = VGroup(formula(r"\sigma_k", size=22).next_to(ax.y_axis.get_end(), LEFT, buff=0.15),
                          formula("t", size=22).next_to(ax.x_axis.get_end(), DOWN, buff=0.15))
        o, ux, uy = ax.c2p(0, 0), ax.c2p(1, 0) - ax.c2p(0, 0), ax.c2p(0, 1) - ax.c2p(0, 0)
        tags = VGroup(*(formula(rf"\sigma_{i + 1}", size=18, colour=PALETTE[i % len(PALETTE)])
                        .next_to(o + T[-1] * ux + SIG[i, -1] * uy, RIGHT, buff=0.08) for i in range(S)))

        self.play(grid_in(g), AxesIn(ax), TextIn(names), TextIn(ax_names))
        t = ValueTracker(float(T[0]))
        self.add(*fields)
        self.add(bodies)
        self.bring_to_front(names)              # the labels sit on the fields, not under them
        curves0 = [curve(sigma_points(ax, i, T[0]), n=CURVE_N, colour=PALETTE[i % len(PALETTE)]) for i in range(S)]
        self.play(*(FadeIn(f) for f in fields), FadeIn(bodies), FadeIn(keys), TextIn(VGroup(clock_name, clock)),
                  *(CurveIn(c, run_time=1.2) for c in curves0))
        self.wait(0.5)

        for k, (kind, j) in enumerate(SLOT):
            fields[k].add_updater(lambda m, kind=kind, j=j: setattr(
                m, "pixel_array", colours(frame_of(kind, j, t.get_value()), *scale_of(kind))))
        clock.add_updater(lambda m: m.set_value(t.get_value()).next_to(clock_name, RIGHT, buff=0.15))
        curves = [always_redraw(lambda i=i: curve(sigma_points(ax, i, t.get_value()), n=CURVE_N, colour=PALETTE[i % len(PALETTE)]))
                  for i in range(S)]
        self.remove(*curves0)
        self.add(*curves)
        self.play(t.animate.set_value(float(T[-1])), run_time=SECONDS, rate_func=linear)
        self.play(TextIn(tags))
        self.wait(1.0)

        for m in fields + curves + [clock]:
            m.clear_updaters()
        self.play(*(CurveOut(c, run_time=1.2) for c in curves), AxesOut(ax), TextOut(names), TextOut(ax_names), TextOut(tags),
                  TextOut(VGroup(clock_name, clock)), *(FadeOut(f) for f in fields), FadeOut(bodies), FadeOut(keys),
                  grid_out(g, run_time=1.2))
        self.wait(0.5)


args.out.parent.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory() as media, tempconfig({"quality": args.quality + "_quality", "media_dir": media}):
    scene = DOPanels()
    scene.render()
    shutil.move(scene.renderer.file_writer.movie_file_path, args.out)
args.out.with_suffix(".cmd").write_text(f"cd {os.getcwd()}\n{sys.executable} {shlex.join(sys.argv)}\n")
