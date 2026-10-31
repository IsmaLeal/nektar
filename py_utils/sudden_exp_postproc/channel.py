"""
The sudden-expansion channel drawn in manim.

Used in do_fields_still.py and dudy_wall_movie.py: fields as colours, the channel's outline, the solid beside the inlet, 
and the colour bar. Each function takes a map `point(x, y)` from channel- to screen-coordinates (defined by the caller).

Needs manim + classic_isma
"""
import numpy as np
from manim import *
from classic_isma import *

_BG, _POS, _NEG = (np.array(color_to_rgb(c)) for c in (BG, SALVIA, MALVA)) # colors (background & colourmaps)


def colours(f, pos, neg):
    """
    For a scalar field `f`, if `f == pos` it saturates to _POS, whereas
                            if `f == neg` then it saturates to _NEG.
    Otherwise, linearly interpolate between them.
    
    Row 0 is the top
    """
    a = np.clip(f / pos, 0, 1)[..., None]  # normalise by `pos` and clip in [0,1] only positive values of `f`
    b = np.clip(-f / neg, 0, 1)[..., None] # same with `neg` and a negative sign: only negative values of `f`
    rgb = _BG + a * (_POS - _BG) + b * (_NEG - _BG) # np.ndarray with rgb vector for each point
    out = np.empty(f.shape + (4,), dtype=np.uint8)
    out[..., :3] = np.round(255 * rgb)     # scale the rgb from [0,1] to [0,255]
    out[..., 3] = 255                      # opacity (alpha) full opaque
    return out[::-1]                       # reverse so that row 0 is the top


def walls(point, x0, x1):
    """
    Creates and returns the channel's outline VGroup (two curves, top and bottom walls).

    `point(x,y)` : callable
    `x0, x1`     : what to show in x (x0>=-5, x1>=15)
    """
    top = [point(x, y) for x, y in [(x0, 0.5), (0, 0.5), (0, 1.5), (x1, 1.5)]]
    bot = [point(x, -y) for x, y in [(x0, 0.5), (0, 0.5), (0, 1.5), (x1, 1.5)]]
    return VGroup(*[curve(VMobject().set_points_as_corners(side), colour=INK, width=2, nib=False) for side in [top, bot]])


def steps(point, x0):
    """
    Creates and returns the solid around the inlet, with stripes, as a VGroup.
    """
    return VGroup(*[hatch([point(x0, s * 0.5), point(0, s * 0.5), point(0, s * 1.5), point(x0, s * 1.5)])
                    for s in [1, -1]])


def field_image(f, point, x0, x1, pos, neg):
    """
    the field f (rows y from -1.5 to 1.5, columns x from x0 to x1) drawn in the channel
    """
    w = point(x1, 0)[0] - point(x0, 0)[0]
    h = point(0, 1.5)[1] - point(0, -1.5)[1]
    # the line below turns RGBA byte array from colours() into Mobject (attribute ImageMobject.pixel_array stores the RGBA)
    m = ImageMobject(colours(f, pos, neg)).stretch_to_fit_width(w).stretch_to_fit_height(h)
    return m.move_to(point((x0 + x1) / 2, 0))


def colour_bar(pos, neg, h, centre, ticks, name=None):
    """
    the colour key: a bar h high from -neg (bottom) to pos (top) centred at the scene point centre, its ticks
    (text, value) at the right and, if given, its name above
    """
    ff = np.linspace(-neg, pos, 256)[:, None] * np.ones((1, 8)) # `ff` here a np.ndarray with 256 rows and 8 cols (the colourbar itself)
    bar = ImageMobject(colours(ff, pos, neg)).stretch_to_fit_height(h).stretch_to_fit_width(0.14)
    bar.move_to(centre)
    y_of = lambda v: bar.get_bottom()[1] + h * (v + neg) / (pos + neg)
    marks = VGroup(*(formula(s, size=24).next_to([bar.get_right()[0], y_of(v), 0], RIGHT, buff=0.1) for s, v in ticks))
    parts = [bar, marks] + ([formula(name, size=30).next_to(bar, UP, buff=0.15)] if name else [])
    return Group(*parts)
