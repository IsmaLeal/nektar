#!/usr/bin/env python3
"""Candidacy-talk figure for the transitions frame: panel (a) of the
research-plan figure "a transition, reconstructed" (fig_transition_reconstructed.py)
on its own, slide-sized. The reconstructed realisation is drawn in blue with
the three instants (b)-(d) marked on its curve; the other transitioning
realisations are drawn in grey behind it. The exemplar and the instants are
selected exactly as in fig_transition_reconstructed.py (same defaults), so the
marks match the snapshots cropped from transition_reconstructed.pdf.

Traces come from se.mp_series (axis-cache-aware). Example:

python3 py_utils/sudden_exp_postproc/candidacy_figs/fig_transition_trace.py \
  --run cases/se/runs/2026_07_17_do_total_sig0p0144_t1000 \
  --out ~/todisimo/studies/7.0_epfl/year1/academic/candidacy_exam/slides/figures/transition_trace.pdf
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # shared readers live one level up

import se

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9,
})

ap = argparse.ArgumentParser()
ap.add_argument("--run", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--mbar", type=float, default=0.1935)
ap.add_argument("--c", type=float, default=0.8)
ap.add_argument("--tmin", type=float, default=30.0)
ap.add_argument("--settle", type=float, default=60.0,
                help="Window (t.u.) the realisation must spend attached "
                     "on each side of the chosen crossing.")
ap.add_argument("--nshow", type=int, default=4,
                help="Curves drawn, the reconstructed one included.")
ap.add_argument("--width", type=float, default=4.6, help="Figure width (in).")
ap.add_argument("--height", type=float, default=2.5, help="Figure height (in).")
args = ap.parse_args()

# ---------------- traces and exemplar (as in fig_transition_reconstructed.py) ----------------
t, yi, Mp, _ = se.mp_series(args.run)
win = t >= args.tmin
trans = se.crossings(t[win], Mp[win], args.c * args.mbar)

best = None
for p, tr in enumerate(trans):
    for ttr in tr:
        if not (args.tmin + 2 * args.settle < ttr < t[-1] - 2 * args.settle):
            continue
        wb = (t > ttr - 1.5 * args.settle) & (t < ttr - 0.5 * args.settle)
        wa = (t > ttr + 0.5 * args.settle) & (t < ttr + 1.5 * args.settle)
        mb, ma = Mp[wb, p].mean(), Mp[wa, p].mean()
        if mb * ma >= 0:
            continue
        score = min(abs(mb), abs(ma))
        if best is None or score > best[0]:
            best = (score, p, ttr, mb, ma)
assert best is not None, "no settled crossing found"
_, p, ttr, mb, ma = best

near = np.abs(t - ttr) < args.settle
k2 = np.where(near)[0][np.argmin(np.abs(Mp[near, p]))]
k1 = np.argmin(np.abs(t - (t[k2] - args.settle)))
k3 = np.argmin(np.abs(t - (t[k2] + args.settle)))
kk = [k1, k2, k3]

order = [int(q) for q in np.argsort([-len(tr) for tr in trans]) if q != p]
show = [p] + order[: args.nshow - 1]

# ---------------- figure ----------------
BLUE, GREY = "#0072B2", "0.45"
fig, ax = plt.subplots(figsize=(args.width, args.height), constrained_layout=True)
for q in show[1:]:
    ax.plot(t, Mp[:, q], lw=0.55, color=GREY, alpha=0.7, zorder=2)
ax.plot(t, Mp[:, p], lw=0.9, color=BLUE, zorder=3)
for k, lab in zip(kk, ["(b)", "(c)", "(d)"]):
    ax.plot(t[k], Mp[k, p], "o", ms=3.5, color=BLUE, mec="white", mew=0.5, zorder=4)
    if lab == "(c)":
        off, ha, va = (6, 0), "left", "center"
    else:
        off, ha, va = (0, 7), "center", "baseline"
    ax.annotate(lab, (t[k], Mp[k, p]), xytext=off, textcoords="offset points",
                ha=ha, va=va, fontsize=8, zorder=5)
ax.axhline(0.0, color="0.8", lw=0.5, zorder=1)
ax.set_xlim(t[0], t[-1])
ax.set_ylim(-0.5, 0.5)
ax.set_xlabel("$t$")
ax.set_ylabel("$M_p$", fontsize=11)
ax.text(0.985, 0.95, "one realisation; three others in grey", transform=ax.transAxes,
        ha="right", va="top", fontsize=7.5, color=GREY)

args.out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out)
print(f"Saved: {args.out}")
print(f"realisation {p}; crossing at t={t[k2]:.1f}; instants "
      + ", ".join(f"t={t[k]:.0f} (M={Mp[k, p]:+.3f})" for k in kk))
print("grey companions:", ", ".join(f"{q} ({len(trans[q])} crossings)" for q in show[1:]))
