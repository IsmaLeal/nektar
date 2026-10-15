#!/usr/bin/env python3
"""One figure out of whichever plots and whichever runs: every --panel is one
axis, a plot kind followed by its options (key=value) and the runs drawn in it.

    --panel KIND [key=value ...] RUN [RUN ...]

Kinds, one file each (draw_<kind>.py draws its runs into the axis it is handed):
    hist    histogram of q=M, A, absM, escape_A, escape_M                 draw_hist.py
    cross   A(t) of one series with the crossings marked, tlim=T0:T1      draw_cross.py
    trace   q=A, M, dL, Y1 against time, a DNS and DO particles           draw_trace.py
    qdag    projection of q^dagger on the DO modes of one run             draw_qdag.py

Runs (se.py): a directory name of the pair or the part after
dns_<pair>_ / do_<pair>_; one argument is one curve, a glob or names joined
by + pool into it, RUN@P is particle P of a DO run. A panel without runs
takes the runs of the panel before it. pair=Re90_sig0.0072 puts one panel on
another pair than --pair.

Panels fill a grid row by row (--ncols, default one column). The figure takes
the fonts, the panel size and the dpi of the kind of its first panel. In a
column, an x label equal to that of the panel below and a title equal to that
of the panel above are written once. Writes the figure and a .cmd file with
the command line next to it (figsave.py).

Examples:
  python3 panel.py --panel hist q=A 'seed??' S6_Np1000_seed45 S8_Np1000_seed45 S12_Np1000_seed45 \
      --out cases/se/figures/Re100_sig0.0144/vary_S_Np1000_eps1e-2/hist_A.png
  python3 panel.py --sharex --panel qdag S6_Np1000_seed45_eps1e-4 --panel qdag S16_Np1000_seed45_eps1e-4 \
      --out cases/se/figures/Re100_sig0.0144/qdag_proj.pdf
  python3 panel.py --sharex --tags \
      --panel trace q=A seed45_Np1000_S8p0 S8_Np1000_seed45_eps1e-4@0 S8_Np1000_seed45_eps1e-8@0 \
      --panel trace q=M --panel trace q=dL --panel trace q=Y1 \
      --out cases/se/figures/Re100_sig0.0144/dns_Re100_sig0.0144_seed45_Np1000_S8p0/twin.pdf
"""
import argparse, importlib
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import se
from figsave import save

KINDS = ('hist', 'cross', 'trace', 'qdag')

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--pair', default='Re100_sig0.0144', help='directory of one (Re, sigma) pair of the sweep, or its name under the sweep root')
ap.add_argument('--panel', nargs='+', action='append', required=True, metavar='ARG', help='KIND [key=value ...] RUN [RUN ...], one per axis')
ap.add_argument('--ncols', type=int, default=1)
ap.add_argument('--figsize', type=float, nargs=2, metavar=('W', 'H'), help='default: the panel size of the first kind times the grid')
ap.add_argument('--sharex', action='store_true')
ap.add_argument('--tags', action='store_true', help='(a), (b), ... above the panels')
ap.add_argument('--out', type=Path, required=True, help='figure path; .pdf also writes the .png')
args = ap.parse_args()

# ---------------------------------------------------------------- which plot and which runs in every panel
panels, curves = [], None
for kind, *rest in args.panel:
    assert kind in KINDS, f'--panel {kind}: one of {", ".join(KINDS)}'
    opts = dict(a.split('=', 1) for a in rest if '=' in a)
    pair = se.pair_dir(opts.pop('pair', args.pair))
    curves = [se.find(pair, a) for a in rest if '=' not in a] or curves        # no runs: those of the panel before
    assert curves, f'--panel {kind}: no runs'
    panels.append((importlib.import_module(f'draw_{kind}'), opts, curves))

# ---------------------------------------------------------------- figure
first = panels[0][0]
plt.rcParams.update(first.RC)
nrows = -(-len(panels) // args.ncols)
fig, axs = plt.subplots(nrows, args.ncols, figsize=args.figsize or (first.SIZE[0] * args.ncols, first.SIZE[1] * nrows),
                        sharex=args.sharex, constrained_layout=True, squeeze=False)
for ax in axs.flat[len(panels):]:
    ax.set_visible(False)
for ax, tag, (mod, opts, curves) in zip(axs.flat, 'abcdefghijklmnopqrstuvwxyz', panels):
    print(f'({tag}) {mod.__name__[5:]} ' + ' '.join(f'{k}={v}' for k, v in opts.items()))
    mod.draw(ax, curves, **opts)
    if args.tags:
        ax.text(0.0, 1.01, f'({tag})', transform=ax.transAxes, va='bottom', ha='left')
for col in axs.T:
    col = [ax for ax in col if ax.get_visible()]
    xlab = [ax.get_xlabel() for ax in col]
    tit = [[ax.get_title(loc=loc) for loc in ('left', 'center', 'right')] for ax in col]
    for i, ax in enumerate(col):
        if i + 1 < len(col) and xlab[i] == xlab[i + 1]:
            ax.set_xlabel('')
        if i > 0 and tit[i] == tit[i - 1]:
            for loc in ('left', 'center', 'right'):
                ax.set_title('', loc=loc)

out = save(fig, args.out, dpi=first.DPI, also_png=True)
print(f'Saved: {out}' + (f' and {out.with_suffix(".png")}' if out.suffix != '.png' else ''))
