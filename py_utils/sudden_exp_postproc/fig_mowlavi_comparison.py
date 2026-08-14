"""Side-by-side comparison of the DOVCS vortex run against the published DO
baseline of Mowlavi & Sapsis (2018), SIAM J. Sci. Comput. 40(3), section 4.6.

Top rows are adapted from their Figures 10 (DO mean/modes/realization at
t=2.5, vorticity colour + velocity arrows) and 11 (DO coefficient marginals
at t=2.5); bottom rows are the DOVCS solution of the same configuration
(cases/vortex/runs/2026_07_08_validation_2s_repro) at its final time t=2,
rendered in the same arrangement and colormap. Mode signs and ordering are
gauge quantities; the comparison is structural.

The paper panels are cropped from the PDF at 300 dpi on the fly (pdftoppm).
"""
import subprocess
import tempfile
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

RUN_H5 = Path("/home/isma/nektar_src_full/cases/vortex/runs/"
              "2026_07_29_2s_repro_to2p5/output/out.h5")
SDO_PDF = Path("/home/isma/todisimo/studies/7.0_epfl/year1/academic/"
               "readings/2_october/SDOs/SDOs.pdf")
OUT_DIR = Path("/home/isma/todisimo/studies/7.0_epfl/year1/academic/"
               "candidacy_exam/tex/figures")
PARTICLE = 850  # matches the do_panels renders of this run

plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                     "mathtext.fontset": "cm", "font.size": 9})


def autocrop(im, pad=6, thr=245):
    arr = np.asarray(im.convert("L"))
    mask = arr < thr
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r0, r1 = np.argmax(rows), len(rows) - np.argmax(rows[::-1])
    c0, c1 = np.argmax(cols), len(cols) - np.argmax(cols[::-1])
    return im.crop((max(c0 - pad, 0), max(r0 - pad, 0),
                    min(c1 + pad, im.width), min(r1 + pad, im.height)))


def paper_crops():
    tmp = Path(tempfile.mkdtemp())
    subprocess.run(["pdftoppm", "-f", "21", "-l", "22", "-r", "300", "-png",
                    str(SDO_PDF), str(tmp / "p")], check=True)
    p21 = Image.open(tmp / "p-21.png")
    p22 = Image.open(tmp / "p-22.png")
    w, h = p21.size
    fig10 = autocrop(p21.crop((int(.10 * w), int(.28 * h),
                               int(.80 * w), int(.635 * h))))
    fig10 = autocrop(fig10.crop((0, 178, fig10.width, fig10.height - 60)))
    fig11 = autocrop(p22.crop((int(.10 * w), int(.11 * h),
                               int(.80 * w), int(.37 * h))))
    fig11 = autocrop(fig11.crop((fig11.width // 2, fig11.height // 2,
                                 fig11.width, fig11.height)))
    return fig10, fig11


def load_run():
    with h5py.File(RUN_H5, "r") as f:
        x = np.asarray(f["x"])
        y = np.asarray(f["y"])
        u = np.asarray(f["u"])[..., -1]
        v = np.asarray(f["v"])[..., -1]
        mu = np.asarray(f["mode_u"])[..., -1]
        mv = np.asarray(f["mode_v"])[..., -1]
        yi = np.asarray(f["yi"])[..., -1]
    return x, y, u, v, mu, mv, yi


def vort(u, v, dx, dy):
    return np.gradient(v, dx, axis=1) - np.gradient(u, dy, axis=0)


def frame_boxes(img, thr=140):
    """Pixel spans (x of each panel, shared y) of the two axes frames in the
    cropped paper figure, so the DOVCS pair can be placed on top of them."""
    a = np.asarray(img.convert("L")) < thr
    h, w = a.shape

    def runs(idx):
        out, start = [], idx[0]
        for p, q in zip(idx, idx[1:]):
            if q - p > 1:
                out.append((start, p))
                start = q
        return out + [(start, idx[-1])]

    cols = runs(np.where(a.sum(axis=0) > 0.5 * h)[0])
    rows = runs(np.where(a.sum(axis=1) > 0.25 * w)[0])
    return ((cols[0][0], cols[1][1]), (cols[2][0], cols[3][1]),
            (rows[0][0], rows[-1][1]))


def field_panel(ax, x, y, u, v, dx, dy, title, clim, ticks):
    w = vort(u, v, dx, dy)
    pc = ax.pcolormesh(x, y, w, cmap="jet", shading="gouraud",
                       vmin=clim[0], vmax=clim[1])
    s = 7
    ax.quiver(x[::s, ::s], y[::s, ::s], u[::s, ::s], v[::s, ::s],
              color="white", width=0.004)
    ax.set_title(title, fontsize=7)
    ax.set_aspect("equal")
    ax.set_xticks([0, 2, 4, 6])
    ax.set_yticks([0, 2, 4, 6])
    ax.tick_params(labelsize=5, pad=1.5)
    cb = plt.colorbar(pc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_ticks(ticks)
    cb.ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    cb.ax.tick_params(labelsize=4.5, pad=1.5)


def main():
    fig10_img, fig11_img = paper_crops()
    x, y, u, v, mu, mv, yi = load_run()
    dx = np.mean(np.diff(x[0, :]))
    dy = np.mean(np.diff(y[:, 0]))
    ur = u + np.tensordot(yi[PARTICLE], mu, axes=1)
    vr = v + np.tensordot(yi[PARTICLE], mv, axes=1)

    # ---- fields composite ----
    fig = plt.figure(figsize=(6.5, 7.6))
    gs = fig.add_gridspec(3, 1,
                          height_ratios=[fig10_img.height / fig10_img.width,
                                         0.035, 0.59],
                          hspace=0.08)
    ax_top = fig.add_subplot(gs[0])
    ax_top.imshow(fig10_img)
    ax_top.axis("off")
    ax_top.set_title(r"Mowlavi \& Sapsis (2018), Fig. 10: DO, t = 2.5",
                     fontsize=9)
    ax_lbl = fig.add_subplot(gs[1])
    ax_lbl.axis("off")
    ax_lbl.text(0.5, 0.0, "DOVCS, t = 2.5", ha="center", va="bottom",
                fontsize=9, transform=ax_lbl.transAxes)
    sub = gs[2].subgridspec(2, 5, hspace=0.28, wspace=0.62,
                            width_ratios=[0.05, 1, 1, 1, 0.05])
    # per-panel colour limits and colorbar ticks matching Mowlavi & Sapsis
    # (2018) Figure 10 panel by panel (limits extend half a tick step past
    # the end ticks, as in their bars)
    panels = [
        (u, v, "Mean, t = 2.5", (-0.75, 2.75),
         [-0.5, 0, 0.5, 1, 1.5, 2, 2.5]),
        (mu[0], mv[0], "Mode 1", (-0.7, 0.7),
         [-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6]),
        (mu[1], mv[1], "Mode 2", (-1.25, 1.75),
         [-1, -0.5, 0, 0.5, 1, 1.5]),
        (ur, vr, "Realization", (-1, 9), [0, 2, 4, 6, 8]),
        (mu[2], mv[2], "Mode 3", (-2.5, 2.5), [-2, -1, 0, 1, 2]),
        (mu[3], mv[3], "Mode 4", (-2.5, 2.5), [-2, -1, 0, 1, 2]),
    ]
    for k, (uu, vv, ttl, clim, ticks) in enumerate(panels):
        ax = fig.add_subplot(sub[k // 3, 1 + k % 3])
        field_panel(ax, x, y, uu, vv, dx, dy, ttl, clim, ticks)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"vortex_mowlavi_fields.{ext}", dpi=200,
                    bbox_inches="tight")
    plt.close(fig)

    # ---- coefficients composite (paper on top, DOVCS below) ----
    # same stacking as the fields composite; the DOVCS pair is placed on the
    # paper panels' own geometry: same frame size, same left and right edges,
    # same gap, each block under its own header
    w_pap, x_pap, gap, y_bot = 3.2, 0.15, 0.30, 0.25
    ipx = w_pap / fig11_img.width               # inches per paper pixel
    h_pap = fig11_img.height * ipx
    (ax0, ax1), (bx0, bx1), (ry0, ry1) = frame_boxes(fig11_img)
    h_frame = (ry1 - ry0) * ipx
    head = 0.20                                 # header strip above each block
    fw = x_pap + w_pap + 0.15
    fh = y_bot + h_frame + head + gap + h_pap + head

    fig = plt.figure(figsize=(fw, fh))

    def box(x, y, w, h):
        return [x / fw, y / fh, w / fw, h / fh]

    y_pap = y_bot + h_frame + head + gap
    ax_pap = fig.add_axes(box(x_pap, y_pap, w_pap, h_pap))
    ax_pap.imshow(fig11_img)
    ax_pap.axis("off")

    for i, j, px0, px1 in [(0, 1, ax0, ax1), (2, 3, bx0, bx1)]:
        ax = fig.add_axes(box(x_pap + px0 * ipx, y_bot,
                              (px1 - px0) * ipx, h_frame))
        ax.scatter(yi[:, i], yi[:, j], s=5, alpha=0.3, color="#0072B2",
                   linewidths=0)
        ax.set_xlabel(f"$Y_{i + 1}$", fontsize=7)
        ax.set_ylabel(f"$Y_{j + 1}$", fontsize=7)
        ax.tick_params(labelsize=5.5, pad=1.5)

    for y0, txt in ((y_pap + h_pap, r"Mowlavi \& Sapsis (2018), Fig. 11: "
                                    r"DO, t = 2.5"),
                    (y_bot + h_frame + 0.02, "DOVCS, t = 2.5")):
        fig.text((x_pap + 0.5 * (ax0 + bx1) * ipx) / fw, (y0 + 0.04) / fh,
                 txt, ha="center", va="bottom", fontsize=8)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"vortex_mowlavi_coeffs.{ext}", dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    print("saved to", OUT_DIR)


if __name__ == "__main__":
    main()
