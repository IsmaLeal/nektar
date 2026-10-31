"""
POD of a DNS (Gram matrix by blocks of rows of the plane) next to
the DO modes of a run at one instant.

Usage:
    python3 svd.py seed45 S6_Np1000_seed45 --out cases/se/figures/Re100_sig0.0144/do_Re100_sig0.0144_S6_Np1000_seed45/pod_vs_do.pdf
    [--pair Re100_sig0.0144] [--modes N] [--comp u|v] [--t T]      runs named as in panel.py; --t: the stored instant nearest to T
"""
import h5py
import shutil
import tempfile
import argparse
import numpy as np
import matplotlib.pyplot as plt

from channel import colour_bar, field_image, steps, walls
from draw_modes import RC, SIZE, DPI, field
from figsave import save
import se

from manim import *
from classic_isma import *


def pod(path, S, row_blocks=100):
    """
    Returns the first S left-singular vectors of all the velocity snapshots
    of a DNS run.

    The x-component of the velocity field at a single time instant can be
    flattened as a column vector with I*J elements:
        (u[0,0], u[0,1], ..., u[0,J-1], u[1,0], u[1,1], ...,  u[I-1,J-1]),
    where the sliced index the nodes in x and y, respectively.
    Doing the same with the y-component, one time instant is a vector containing
    2*I*J elements. Hence, the matrix X of all the N snapshots across the run has
    2*I*J rows and N columns. We first de-mean X.

    Then, we find the right-singular vectors V and singular values S by eigen-
    decomposing the gram matrix (X^T @ X). Then, we build
        U = X  V S^{-1}.

    X is always loaded by blocks, slicing the j index, thus selecting all rows and
    snapshots of a slice in y for each block.
    """
    with h5py.File(path, "r") as f:
        N, J, I = f["u"].shape  # N timesteps, J nodes in y, I nodes in x
    gram = np.zeros((N, N))     # X^T @ X
    b0, b1 = 0, row_blocks      # indices of the first and last per-block rows
    with h5py.File(path, "r") as f:
        while b0 < 2*J:         # each row contains all the I points, so there's 2*J rows
            b1 = min(b1, J if b0 < J else 2 * J) # last x- and y- components if J / row_blocks is not an integer
            if b0 < J:
                X_block = f["u"][:, b0:b1, :].reshape(N, -1).T
            else:
                X_block = f["v"][:, (b0-J):(b1-J), :].reshape(N, -1).T
            X_block -= X_block.mean(axis=1, keepdims=True) # de-mean for POD
            gram += X_block.T @ X_block
            b0 = b1
            b1 += row_blocks
            print(b0)
    # real, symmetric eigen-decomposition (ordered descending)
    s2, V = np.linalg.eigh(gram)
    s2 = np.flip(s2); V = np.flip(V, axis=1)
    result = np.zeros((2*I*J, S))
    b0, b1 = 0, row_blocks
    with h5py.File(path, "r") as f:
        while b0 < 2*J:
            b1 = min(b1, J if b0 < J else 2*J)
            if b0 < J:
                X_block = f["u"][:, b0:b1, :].reshape(N, -1).T
            else:
                X_block = f["v"][:, (b0-J):(b1-J), :].reshape(N, -1).T
            X_block -= X_block.mean(axis=1, keepdims=True)
            result[b0*I:b1*I, :] += X_block @ V[:, :S] / np.sqrt(s2[:S] * se.DX * se.DY)
            b0 = b1
            b1 += row_blocks
            print(b0)
    modes_u = result[:I*J].reshape((J, I, S))
    modes_v = result[I*J:].reshape((J, I, S))
    return modes_u, modes_v


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("dns")
ap.add_argument("do")
ap.add_argument("--pair", default="Re100_sig0.0144")
ap.add_argument("--modes", type=int, help="the first N modes; default every mode of the DO run")
ap.add_argument("--comp", default="u", choices=("u", "v"))
ap.add_argument("--t", type=float)
ap.add_argument("--out", required=True)
args = ap.parse_args()
pair = se.pair_dir(args.pair)
(dns,), _ = se.find(pair, args.dns)
(do,), _ = se.find(pair, args.do)

with h5py.File(se.planes(do), "r") as h:
    tg = h["grid/t"][:]
    k = len(tg) - 1 if args.t is None else int(np.argmin(np.abs(tg - args.t)))
    do_modes = h[f"grid/mode_{args.comp}"][k].astype(float)        # (S, NY, NX)
S = min(do_modes.shape[0], args.modes) if args.modes else do_modes.shape[0]
pod_u, pod_v = pod(se.planes(dns), S)
pod_modes = pod_u if args.comp == "u" else pod_v

XMIN, XMAX = -5, 15
j = (se.x >= XMIN) & (se.x <= XMAX)
inside = se.INSIDE[:, j]
# fill with 0 points not inside; fields otherwise
pods = [np.where(inside, pod_modes[:, j, i], 0) for i in range(S)]
dos = [np.where(inside, do_modes[i][:, j], 0) for i in range(S)]

s = 0.3                     # scene units per physical units
ROW_HEIGHT = 1.1            # height of one row in scene units (3*s is the channel's height)
TOP = 8.5                   # y-coord of first row's channel axis (its y=0)
X0_DNS, X0_DO = -5, 1.8    # x-coord of physical x=0
F_MAX = 0.4


def point_map(row, x0):
    return lambda x, y: np.array([x0 + x*s, row+y*s, 0])


class ClassicIsmaPlot(Scene):
    def construct(self):
        g = grid()
        self.add(g)

        for i in range(S):
            row = TOP - i * ROW_HEIGHT
            for f, x0, sym in ((pods[i], X0_DNS, rf'\mathrm{{POD}}\ u_{{{i + 1}}}'),
                               (dos[i], X0_DO, rf'\mathrm{{DO}}\ u_{{{i + 1}}}')):
                p = point_map(row, x0)
                self.add(field_image(f, p, XMIN, XMAX, F_MAX, F_MAX), walls(p, XMIN, XMAX), steps(p, XMIN),
                         formula(sym, size=20).next_to(p(XMIN, 0), LEFT, buff=0.2))

        centre = [X0_DO + XMAX*s + 0.5, TOP - (S - 1)*ROW_HEIGHT / 2, 0]
        self.add(colour_bar(F_MAX, F_MAX, 3.4, centre, ((r'0.4', F_MAX), (r'0', 0.0), (r'-0.4', -F_MAX))))


with tempfile.TemporaryDirectory() as media, tempconfig({'quality': 'high_quality',
                                                         'media_dir': media, 'save_last_frame': True,
                                                         'pixel_width': 1600, 'pixel_height': 1900,
                                                         'frame_width': 16.0, 'frame_height': 19.0}):
    scene = ClassicIsmaPlot()
    scene.render()
    shutil.move(scene.renderer.file_writer.image_file_path, args.out)


# plt.rcParams.update(RC)
# fig, axs = plt.subplots(S, 2, figsize=(2 * SIZE[0], S * SIZE[1]), sharex=True, constrained_layout=True, squeeze=False)
# for i in range(S):
#     field(axs[i, 0], pod_modes[:, :, i], f"POD ${args.comp}_{{{i + 1}}}$")
#     field(axs[i, 1], do_modes[i], f"DO ${args.comp}_{{{i + 1}}}$")
#     axs[i, 1].set_ylabel("")
# axs[0, 0].set_title(se.label(dns), loc="left", fontsize=9)
# axs[0, 1].set_title(se.label(do) + f", $t = {tg[k]:g}$", loc="left", fontsize=9)
# for ax in axs[-1]:
#     ax.set_xlabel("$x$", fontsize=13)
# out = save(fig, args.out, dpi=DPI, also_png=True)
# print(f"Saved: {out}")
