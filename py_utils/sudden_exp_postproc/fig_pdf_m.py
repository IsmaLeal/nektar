#!/usr/bin/env python3
"""Research-plan figure for subsection 3.4: probability density of the
asymmetry M, pooled over realisations and time, DO ensemble against the
two reference DNS-mode simulations at the same forcing amplitude.

python3 py_utils/sudden_exp_postproc/fig_pdf_m.py \
  --run cases/se/runs/2026_07_17_do_total_sig0p0144_t1000 \
  --dns cases/se/runs/2026_07_15_dns_clean_seed42 \
        cases/se/runs/2026_07_15_dns_clean_seed43 \
  --out ~/todisimo/studies/7.0_epfl/year1/academic/candidacy_exam/tex/figures/pdf_m.pdf
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from do_quicklook import mp_series
from do_stats_plots import dns_m_series

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9,
})

ap = argparse.ArgumentParser()
ap.add_argument("--run", type=Path, required=True)
ap.add_argument("--dns", type=Path, nargs="+", required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--mbar", type=float, default=0.1935)
ap.add_argument("--tmin", type=float, default=30.0)
ap.add_argument("--bins", type=int, default=91)
args = ap.parse_args()

t, _, Mp, _ = mp_series(args.run)
do_pool = Mp[t >= args.tmin].ravel()

dns_pool = []
for d in args.dns:
    td, Md = dns_m_series(d)
    dns_pool.append(Md[td >= args.tmin])
dns_pool = np.concatenate(dns_pool)

edges = np.linspace(-0.5, 0.5, args.bins)
ctr = 0.5 * (edges[1:] + edges[:-1])
pdo, _ = np.histogram(do_pool, bins=edges, density=True)
pdns, _ = np.histogram(dns_pool, bins=edges, density=True)
print(f"occupancy M<0: DO {np.mean(do_pool < 0):.2f}, "
      f"DNS {np.mean(dns_pool < 0):.2f}")

fig, ax = plt.subplots(figsize=(3.6, 2.4), constrained_layout=True)
ax.plot(ctr, pdns, lw=1.0, color="0.45", drawstyle="steps-mid")
ax.plot(ctr, pdo, lw=1.0, color="#0072B2", drawstyle="steps-mid")
ax.text(0.03, 0.95, "DO", color="#0072B2", fontsize=9,
        transform=ax.transAxes, va="top")
ax.text(0.03, 0.85, "DNS", color="0.45", fontsize=9,
        transform=ax.transAxes, va="top")
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(bottom=0)
ax.set_xlabel("$M$")
ax.set_ylabel("probability density", fontsize=9)
ax.set_xticks([-0.4, -0.2, 0.0, 0.2, 0.4])

args.out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out)
png = args.out.with_suffix(".png")
fig.savefig(png, dpi=200)
print(f"Saved: {args.out} and {png}")
print(f"pooled samples: DO {do_pool.size}, DNS {dns_pool.size}")
