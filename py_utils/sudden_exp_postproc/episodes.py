"""
Every sign episode of A (a stretch of one series between two changes of sign)
as one point: its duration against the farthest |A| it reaches, in units of
A_fp. The DNS of the pair pooled, in blue, and every particle of one DO run,
in red, on the same axes. The first and the last episode of each series are
cut by the ends of the series and are left out.

Usage:
    python3 episodes.py S16_Np1000_seed45_eps1e-4 --out cases/se/figures/Re100_sig0.0144/do_Re100_sig0.0144_S16_Np1000_seed45_eps0.0001/episodes.png
    [--pair Re100_sig0.0144] [--dns 'seed??']      runs named as in panel.py; --dns: the DNS pooled, default every DNS of the pair
"""
import shutil
import tempfile
import argparse
import numpy as np
import matplotlib.pyplot as plt

from figsave import save
import se

from manim import *
from classic_isma import *

RC = {"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9}
SIZE, DPI = (6.3, 4), 170                   # size of the figure, dpi of the png
BLUE, RED = "#0072B2", "#D55E00"


def episodes(runs):
    """
    Duration and max |A| / A_fp of every sign episode of A in the runs given,
    every particle, as two arrays.
    """
    afp = se.a_fp(se.params(runs[0])["Re"])
    dur, dep = [], []
    for run in runs:
        t, A = se.series(run, "A")
        dt = t[1] - t[0]
        for a in A.T:
            s = np.sign(a); s[s == 0] = 1
            cut = np.flatnonzero(s[1:] != s[:-1]) + 1                   # first index of each new episode
            edges = np.concatenate(([0], cut, [len(a)]))
            if len(edges) < 4:                                           # fewer than two whole episodes
                continue
            dur.append(np.diff(edges)[1:-1] * dt)
            dep.append(np.maximum.reduceat(np.abs(a), edges[:-1])[1:-1])
    return np.concatenate(dur), np.concatenate(dep) / afp


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("do")
ap.add_argument("--pair", default="Re100_sig0.0144")
ap.add_argument("--dns")
ap.add_argument("--out", required=True)
args = ap.parse_args()
pair = se.pair_dir(args.pair)
(do,), _ = se.find(pair, args.do)
dns = se.find(pair, args.dns)[0] if args.dns else se.runs("dns", [pair])

d_dns, p_dns = episodes(dns)
d_do, p_do = episodes([do])


class ClassicIsmaScene(Scene):
    def construct(self):
        self.add(grid())
        ax = MyAxes([0, 3.5], [0, 2], origin=[-4.5, -2.8, 0], x_length=9, y_length=5, log='x')
        self.add(ax, decades(ax, 'x', size=24, minor=True))
        for v in (0.5, 1, 1.5, 2):
            self.add(ax.y_axis.get_tick(v), formula(f'{v:g}', size=24).next_to(ax.c2p(1, v), LEFT, buff=0.12))
        self.add(DashedLine(ax.c2p(1, 1), ax.c2p(10 ** 3.5, 1), color=PIZARRA, stroke_width=1.5))
        self.add(*[dot(ax.c2p(d, p), radius=0.02, colour=CORAL, fill_opacity=0.2) for d, p in zip(d_do, p_do)])
        self.add(*[dot(ax.c2p(d, p), radius=0.02, colour=SALVIA) for d, p in zip(d_dns, p_dns)])
        self.add(label('residence time', size=28).next_to(ax.x_axis, DOWN, buff=0.7),
                 formula(r'\max |A| / A_{fp}', size=28).rotate(PI / 2).next_to(ax.y_axis, LEFT, buff=0.6),
                 label(f'{se.label(dns[0])}, {len(dns)} runs, {d_dns.size} residences', size=24, colour=SALVIA).move_to(ax.c2p(1.3, 2.22), aligned_edge=UL),
                 label(f'{se.label(do)}, {d_do.size} residences', size=24, colour=CORAL).move_to(ax.c2p(1.3, 2.4), aligned_edge=UL))

with tempfile.TemporaryDirectory() as media, tempconfig({'quality': 'high_quality',
                                                         'media_dir': media,
                                                         'save_last_frame': True,
                                                         'pixel_width': 1800, 'pixel_height': 1200,
                                                         'frame_width': 12, 'frame_height': 8}):
    scene = ClassicIsmaScene()
    scene.render()
    shutil.move(scene.renderer.file_writer.image_file_path, args.out)




# plt.rcParams.update(RC)
# fig, ax = plt.subplots(figsize=SIZE, constrained_layout=True)
# ax.scatter(d_do, p_do, s=3, alpha=0.12, color=RED, edgecolors="none")
# ax.scatter(d_dns, p_dns, s=5, alpha=0.5, color=BLUE, edgecolors="none")
# for y in (0.5, 1.0):
#     ax.axhline(y, color="0.45", lw=0.6, ls="--")
# ax.set_xscale("log")
# ax.set_xlabel("episode duration")
# ax.set_ylabel(r"$\max |A| / A_{fp}$ in the episode")
# ax.text(0.02, 0.96, f"DNS, {len(dns)} runs, {d_dns.size} episodes", color=BLUE, transform=ax.transAxes, va="top")
# ax.text(0.02, 0.90, f"{se.label(do)}, {d_do.size} episodes", color=RED, transform=ax.transAxes, va="top")
# out = save(fig, args.out, dpi=DPI, also_png=True)
# print(f"Saved: {out}")
