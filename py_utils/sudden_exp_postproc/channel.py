"""The sudden-expansion channel drawn in manim, shared by do_fields_still.py and dudy_wall_movie.py: a field as
colours, the channel's outline, the solid beside the inlet, the field image and the colour bar. Each function takes
point(x, y), the scene point of a point (x, y) of the channel, so the caller decides where and how big the channel is.
Runs in the divulg venv (manim + classic_isma); no h5py here."""
import numpy as np
from manim import *
from classic_isma import *

_BG, _POS, _NEG = (np.array(color_to_rgb(c)) for c in (BG, SALVIA, MALVA))     # background, f > 0, f < 0


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


def walls(point, x0, x1):
    """channel's outline: inlet from x0, step at x = 0, walls at y = +-1.5 up to x1"""
    top = [point(x, y) for x, y in [(x0, 0.5), (0, 0.5), (0, 1.5), (x1, 1.5)]]
    bot = [point(x, -y) for x, y in [(x0, 0.5), (0, 0.5), (0, 1.5), (x1, 1.5)]]
    return VGroup(*(curve(VMobject().set_points_as_corners(p), color=INK, width=2, nib=False) for p in (top, bot)))


def steps(point, x0):
    """the solid beside the inlet, above and below it (with stripes)"""
    return VGroup(*(hatch([point(x0, s * 0.5), point(0, s * 0.5), point(0, s * 1.5), point(x0, s * 1.5)])
                    for s in (1, -1)))


def field_image(f, point, x0, x1, pos, neg):
    """the field f (rows y from -1.5 to 1.5, columns x from x0 to x1) drawn in the channel"""
    w = point(x1, 0)[0] - point(x0, 0)[0]
    h = point(0, 1.5)[1] - point(0, -1.5)[1]
    m = ImageMobject(colours(f, pos, neg)).stretch_to_fit_width(w).stretch_to_fit_height(h)
    return m.move_to(point((x0 + x1) / 2, 0))


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
