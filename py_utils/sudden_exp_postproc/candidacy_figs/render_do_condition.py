#!/usr/bin/env python3
"""Render the DO-condition clip for the candidacy deck (frame 'The DO equations').

Sketch for S = 2 in R^3: two orthonormal modes u_1, u_2 span the plane V_S.
  Phase A (0-5 s)   in-span rotation of the basis: the plane is fixed, the modes
                    spin inside it (there and back); their velocities (grey,
                    struck through) lie in the plane. This is the motion the DO
                    condition forbids.
  Phase B (5-12 s)  out-of-span motion: the basis is frozen inside the plane and
                    the plane tilts twice about in-plane axes; the mode
                    velocities (accent) are normal to the plane, marked with
                    right-angle symbols.
Palette and background follow theme/beamerthemelfmi.sty so the clip blends into
the slide. Output: MP4 (H.264, yuv420p) plus a poster PNG (phase B, mid-tilt).
The tilt axes are chosen by a small search so that the plane normal stays
between ~40 and ~70 degrees from the viewing direction (arrows along the normal
never foreshorten away, the plane is never seen edge-on).

usage: render_do_condition.py OUTDIR [--stills]
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch
from matplotlib.animation import FFMpegWriter

# ---------------------------------------------------------------- palette
PAPER  = "#FBFBF8"
ACCENT = "#1F77B4"
GREY   = "#777777"
BLACK  = "#222222"
HAIR   = "#D2D2CE"   # lfmigrey!30!lfmipaper

plt.rcParams.update({"mathtext.fontset": "cm", "font.size": 26})

# ---------------------------------------------------------------- timing
FPS = 30
T_A0, T_A1 = 0.6, 4.4         # in-span spin there-and-back (phase A)
T_A_END = 5.0
T_B0, T_B1 = 5.6, 8.6         # first tilt (phase B)
T_C0, T_C1 = 9.0, 11.6        # second tilt about another in-plane axis
T_END = 12.6                  # hold on the final frame until here
THETA_MAX = np.deg2rad(40.0)  # in-span spin amplitude
PHI_1 = np.deg2rad(34.0)      # first tilt amplitude (sign chosen by the search)
PHI_2 = np.deg2rad(30.0)      # second tilt amplitude

# ---------------------------------------------------------------- camera
AZ, EL = np.deg2rad(-38.0), np.deg2rad(36.0)
_d     = np.array([np.cos(EL) * np.cos(AZ), np.cos(EL) * np.sin(AZ), np.sin(EL)])
_right = np.array([-np.sin(AZ), np.cos(AZ), 0.0])
_up    = np.cross(_d, _right)
_up   /= np.linalg.norm(_up)

def proj(p):
    p = np.asarray(p, float)
    return np.array([p @ _right, p @ _up])

# ---------------------------------------------------------------- kinematics
def smooth(s):
    """C^1 ease in/out on [0,1] (smoothstep) and its derivative."""
    s = np.clip(s, 0.0, 1.0)
    return 3 * s**2 - 2 * s**3, 6 * s * (1 - s)

def rot(axis, ang):
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K

# base orientation of the plane
R0 = rot(np.array([1.0, 0.3, 0.0]), np.deg2rad(-8.0)) @ rot(np.array([0, 0, 1.0]), np.deg2rad(12.0))
E1, E2 = R0 @ np.array([1.0, 0, 0]), R0 @ np.array([0, 1.0, 0])

def state(t, tilts):
    """Return u1, u2, n, v1, v2, phase, activity in [0,1], plane frame (p1, p2)."""
    if t < T_A_END:
        s = (t - T_A0) / (T_A1 - T_A0)
        f, df = smooth(s)
        theta = THETA_MAX * np.sin(np.pi * f)
        dtheta = THETA_MAX * np.pi * np.cos(np.pi * f) * df / (T_A1 - T_A0)
        n = np.cross(E1, E2)
        Rin = rot(n, theta)
        u1, u2 = Rin @ E1, Rin @ E2
        omega = dtheta * n
        act = min(1.0, abs(dtheta) / (THETA_MAX * np.pi / (T_A1 - T_A0)))
        return u1, u2, n, np.cross(omega, u1), np.cross(omega, u2), "A", act, (E1, E2)
    u1, u2 = E1.copy(), E2.copy()
    omega = np.zeros(3)
    for (t0, t1, phi, beta) in tilts:
        s = (t - t0) / (t1 - t0)
        f, df = smooth(s)
        a = np.cos(beta) * u1 + np.sin(beta) * u2        # axis fixed in the moving frame
        u1, u2 = rot(a, phi * f) @ u1, rot(a, phi * f) @ u2
        if 0.0 <= s <= 1.0:
            omega = phi * df / (t1 - t0) * a
    n = np.cross(u1, u2)
    act = min(1.0, np.linalg.norm(omega) / (abs(PHI_1) * 1.5 / (T_B1 - T_B0)))
    return u1, u2, n, np.cross(omega, u1), np.cross(omega, u2), "B", act, (u1, u2)

def choose_tilts():
    """Pick (beta_1, sign_1, beta_2, sign_2) keeping cos(angle(n, d)) in [0.34, 0.78]
    over phase B and maximising the smallest projected length of the normal while
    arrows are active."""
    best, best_score = None, -1.0
    ts = np.arange(T_A_END, T_END + 1e-9, 1.0 / 15)
    for b1 in np.deg2rad(np.arange(0, 360, 15)):
        for s1 in (+1, -1):
            for b2 in np.deg2rad(np.arange(0, 360, 15)):
                for s2 in (+1, -1):
                    tilts = ((T_B0, T_B1, s1 * PHI_1, b1), (T_C0, T_C1, s2 * PHI_2, b2))
                    cosang, score = [], 1.0
                    ok = True
                    for t in ts:
                        u1, u2, n, v1, v2, _, act, _ = state(t, tilts)
                        c = abs(n @ _d)
                        if not (0.34 <= c <= 0.78):
                            ok = False
                            break
                        if act > 0.3:
                            score = min(score, np.sqrt(1 - c * c))
                    if ok and score > best_score:
                        best, best_score = tilts, score
    return best, best_score

# ---------------------------------------------------------------- drawing
L = 1.30          # plane half-extent (in mode units)
VLEN = 0.55       # velocity arrow length at peak angular speed

def arrow(ax, p0, p1, color, lw, alpha=1.0, ms=28, z=5):
    ax.add_patch(FancyArrowPatch(proj(p0), proj(p1), arrowstyle="-|>", mutation_scale=ms,
                                 color=color, lw=lw, alpha=alpha, zorder=z,
                                 shrinkA=0, shrinkB=0, joinstyle="miter", capstyle="butt"))

def right_angle(ax, tip, e_in, e_n, size, color, lw, alpha):
    p = [tip - size * e_in, tip - size * e_in + size * e_n, tip + size * e_n]
    q = np.array([proj(x) for x in p])
    ax.plot(q[:, 0], q[:, 1], color=color, lw=lw, alpha=alpha, zorder=8, solid_joinstyle="miter")

def draw(ax, t, tilts, lims):
    ax.cla()
    ax.set_facecolor(PAPER)
    ax.set_aspect("equal")
    ax.set_xlim(*lims[0])
    ax.set_ylim(*lims[1])
    ax.axis("off")
    u1, u2, n, v1, v2, phase, act, (p1, p2) = state(t, tilts)

    # plane V_S (outline fixed to the plane frame) with a light grid riding on the basis
    corners = [L * (a * p1 + b * p2) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    fill = Polygon([proj(c) for c in corners], closed=True, facecolor=ACCENT,
                   edgecolor="none", alpha=0.10, zorder=1)
    ax.add_patch(fill)
    ax.add_patch(Polygon([proj(c) for c in corners], closed=True, fill=False,
                         edgecolor=HAIR, lw=1.4, zorder=2))
    for k in (-0.5, 0.0, 0.5):
        for (ea, eb) in ((u1, u2), (u2, u1)):
            q = np.array([proj(1.6 * L * (k * ea - eb)), proj(1.6 * L * (k * ea + eb))])
            ln, = ax.plot(q[:, 0], q[:, 1], color=HAIR, lw=0.9, zorder=2)
            ln.set_clip_path(fill)
    # the normal, for reference
    q = np.array([proj(np.zeros(3)), proj(0.95 * n)])
    ax.plot(q[:, 0], q[:, 1], color=GREY, lw=1.2, ls=(0, (4, 3)), zorder=3, alpha=0.8)

    # modes and labels (labels pushed outward in screen space, away from the arrows)
    arrow(ax, np.zeros(3), u1, ACCENT, 5.0, z=6)
    arrow(ax, np.zeros(3), u2, ACCENT, 5.0, z=6)
    for u, lab in ((u1, r"$u_1$"), (u2, r"$u_2$")):
        q = proj(u)
        e = q / (np.linalg.norm(q) + 1e-12)
        q = q + 0.20 * e + 0.05 * np.array([-e[1], e[0]]) * np.sign(e[0])
        ax.text(q[0], q[1], lab, color=ACCENT, ha="center", va="center", zorder=9)
    q = proj(L * (0.80 * p1 - 1.0 * p2)) + np.array([0.0, -0.12])
    ax.text(q[0], q[1], r"$V_S$", color=GREY, ha="center", va="center", zorder=9)

    # mode velocities
    if act > 0.02:
        if phase == "A":
            for u, v in ((u1, v1), (u2, v2)):
                e = v / (np.linalg.norm(v) + 1e-12)
                tip = u + VLEN * act * e
                arrow(ax, u, tip, GREY, 3.6, alpha=act, z=7)
                # struck through: forbidden
                c = proj(tip) + 0.10 * (proj(tip) - proj(u)) / (np.linalg.norm(proj(tip) - proj(u)) + 1e-12)
                s = 0.055
                for dx, dy in ((s, s), (s, -s)):
                    ax.plot([c[0] - dx, c[0] + dx], [c[1] - dy, c[1] + dy], color=BLACK, lw=3.0,
                            alpha=act, zorder=9, solid_capstyle="round")
        else:
            for u, v in ((u1, v1), (u2, v2)):
                if np.linalg.norm(v) < 1e-9:
                    continue
                e_n = np.sign(v @ n) * n
                arrow(ax, u, u + VLEN * act * e_n, ACCENT, 3.6, alpha=act, z=7)
                right_angle(ax, u, u / np.linalg.norm(u), e_n, 0.11, ACCENT, 2.0, act)

def frame_limits(tilts, aspect=1.25, margin=0.10):
    pts = []
    for t in np.arange(0.0, T_END + 1e-9, 1.0 / 10):
        u1, u2, n, v1, v2, phase, act, (p1, p2) = state(t, tilts)
        for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            pts.append(proj(L * (a * p1 + b * p2)))
        for u, e_n in ((u1, n), (u2, n)):
            pts.append(proj(u + VLEN * e_n))
            pts.append(proj(u - VLEN * e_n))
            pts.append(proj(1.32 * u))
    pts = np.array(pts)
    lo, hi = pts.min(0) - margin, pts.max(0) + margin
    c, ext = (lo + hi) / 2, hi - lo
    if ext[0] / ext[1] < aspect:
        ext[0] = aspect * ext[1]
    else:
        ext[1] = ext[0] / aspect
    return ((c[0] - ext[0] / 2, c[0] + ext[0] / 2), (c[1] - ext[1] / 2, c[1] + ext[1] / 2))

def poster_time(tilts):
    best, bt = -1, T_B0
    for t in np.arange(T_B0, T_B1, 1.0 / 30):
        u1, u2, n, v1, v2, phase, act, _ = state(t, tilts)
        score = act * np.sqrt(1 - (n @ _d) ** 2)
        if score > best:
            best, bt = score, t
    return bt

def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    tilts, score = choose_tilts()
    print("tilts:", [(round(np.rad2deg(b), 1), round(np.rad2deg(phi), 1)) for (_, _, phi, b) in tilts],
          "min projected normal while active: %.2f" % score)
    lims = frame_limits(tilts)
    fig = plt.figure(figsize=(6.4, 5.12), dpi=250, facecolor=PAPER)
    ax = fig.add_axes([0, 0, 1, 1])
    if "--stills" in sys.argv:
        for t in (0.0, 1.5, 3.2, poster_time(tilts), 7.8, 10.3):
            draw(ax, t, tilts, lims)
            fig.savefig(os.path.join(out, f"still_{t:04.1f}.png"), dpi=250, facecolor=PAPER)
        return
    tp = poster_time(tilts)
    draw(ax, tp, tilts, lims)
    fig.savefig(os.path.join(out, "do_condition_poster.png"), dpi=250, facecolor=PAPER)
    writer = FFMpegWriter(fps=FPS, codec="libx264",
                          extra_args=["-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart"])
    nfr = int(round(T_END * FPS))
    with writer.saving(fig, os.path.join(out, "do_condition.mp4"), dpi=250):
        for i in range(nfr + 1):
            draw(ax, i / FPS, tilts, lims)
            writer.grab_frame(facecolor=PAPER)
    print("poster at t = %.2f s" % tp)

if __name__ == "__main__":
    main()
