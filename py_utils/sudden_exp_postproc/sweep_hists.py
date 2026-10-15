"""Histograms of M over the whole sweep: one panel per (Re, sigma) pair, Re down the rows and sigma along the
columns, the DNS seeds of the pair pooled (twins left out) and one curve per DO run with its forcing-seed
repeats pooled, each panel drawn by draw_hist.py like every other histogram.
    python3 sweep_hists.py                       hist_M_sweep.png and hist_absM_sweep.png in cases/se/figures/sweep
    python3 sweep_hists.py --q A --out DIR
"""
import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import draw_hist
import se
from figsave import save


def curves_of(pair):
    """The curves of one pair: the DNS seeds pooled, then one curve per DO run (its _frc repeats pooled), in order."""
    dns = [r for r in se.runs("dns", [pair]) if not se.params(r)["twin"]]
    do = defaultdict(list)
    for r in se.runs("do", [pair]):
        p = se.params(r)
        do[(p["S"], p["Np"], p["eps"], p["seed"])].append(r)
    return ([(dns, None)] if dns else []) + [(do[k], None) for k in sorted(do)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--q", nargs="+", default=["M", "absM"], choices=["M", "absM", "A", "dL"])
    ap.add_argument("--out", type=Path, default=se.FIGS / "sweep", help="directory of hist_<q>_sweep.png")
    args = ap.parse_args()

    pairs = [p for p in sorted(se.SWEEP.glob("Re*_sig*")) if p.is_dir() and float(p.name.split("_sig")[1]) > 0]
    Res = sorted({int(p.name[2:].split("_sig")[0]) for p in pairs})
    sigs = sorted({p.name.split("_sig")[1] for p in pairs}, key=float)
    for q in args.q:
        plt.rcParams.update(draw_hist.RC)
        fig, axes = plt.subplots(len(Res), len(sigs), squeeze=False, constrained_layout=True,
                                 figsize=(draw_hist.SIZE[0] * len(sigs), draw_hist.SIZE[1] * len(Res)))
        for i, Re in enumerate(Res):
            for j, sig in enumerate(sigs):
                pair = se.SWEEP / f"Re{Re}_sig{sig}"
                if pair not in pairs:
                    axes[i, j].set_axis_off()
                    continue
                print(f"{pair.name}, {q}:")
                curves = curves_of(pair)
                draw_hist.draw(axes[i, j], curves, q=q)
                if len(curves) > 8:                                    # Re 100, sigma 0.0144: every DO run of the campaign
                    axes[i, j].legend(fontsize=5, ncol=2, frameon=False, loc="upper center")
        out = save(fig, args.out / f"hist_{q}_sweep.png", dpi=draw_hist.DPI)
        plt.close(fig)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
