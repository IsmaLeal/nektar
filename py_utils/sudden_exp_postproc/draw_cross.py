"""A(t) of one series (a DNS, or particle P of a DO run, RUN@P) over one time
window, with the crossings of three candidate criteria marked, so the
criteria can be judged by eye (panel.py --panel cross).

  (1) Schmitt trigger at +-0.5 A_fp on A            (campaign trigger)
  (2) Schmitt trigger at +-A_fp on A                (cores at the fixed points)
  (3) Schmitt trigger at +-0.5 A_fp on the running mean of A over 50 t.u.
      (the in-well correlation time: 1/e lag 40-50 t.u. in DNS and DO)

Blue: A(t); gray: running mean; dotted lines +-0.5 A_fp, dashed +-A_fp.
Crossing marks: (1) at the top edge, (2) middle-top, (3) bottom, with the
count of the window in the panel corner and the count of the whole series in
the title. The window is tlim=T0:T1.

    python3 panel.py --panel cross tlim=0:4000 seed45 --panel cross tlim=4000:8000 \
        --out cases/se/figures/Re100_sig0.0144/dns_Re100_sig0.0144_seed45/cross.png
    python3 panel.py --panel cross tlim=0:4000 S6_Np1000_seed45_eps1e-4@269 --panel cross tlim=4000:8000 \
        --out cases/se/figures/Re100_sig0.0144/do_Re100_sig0.0144_S6_Np1000_seed45_eps0.0001/cross_p269.png
"""
import numpy as np

import se

RC = {"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9}
SIZE, DPI = (11, 2.6), 130                  # size of one panel, dpi of the png

W = 0.701
WIN = 50.0


def runmean(t, a, win):
    n = max(1, int(round(win / (t[1] - t[0]))))
    k = np.ones(n) / n
    pad = np.pad(a, (n // 2, n - 1 - n // 2), mode='edge')
    return np.convolve(pad, k, mode='valid')


def draw(ax, curves, tlim='0:4000'):
    """curves: one series, [([run directory], particle or None)] (se.find)."""
    assert len(curves) == 1 and len(curves[0][0]) == 1, 'cross: one series per panel'
    (run,), p = curves[0]
    par = se.params(run)
    if par['kind'] == 'dns':
        name = f'DNS {run.parent.name} seed {run.name.split("_seed")[1]}'
    else:
        assert p is not None, f'cross: {run.name} needs a particle, RUN@P'
        cfg = se.short_eps(run.name.split(f'_sig{par["sig"]}_', 1)[1].replace('_seed45', ''))     # e.g. S6_Np1000_eps1e-4
        name = f'DO {cfg.replace("_", " ")}, particle {p}'
    t, a = se.series(run, 'A', p); a = a[:, 0]
    t0, t1 = (float(s) for s in tlim.split(':'))
    am = runmean(t, a, WIN)
    c1, c2, c3 = se.crossings(t, a, 0.5 * W), se.crossings(t, a, W), se.crossings(t, am, 0.5 * W)
    ax.plot(t, a, color='#0072B2', lw=0.6)
    ax.plot(t, am, color='0.45', lw=0.9)
    for y, ls in [(0.5 * W, ':'), (W, '--')]:
        ax.axhline(y, color='0.6', lw=0.5, ls=ls); ax.axhline(-y, color='0.6', lw=0.5, ls=ls)
    ax.axhline(0, color='0.8', lw=0.4)
    for c, y, col, mk in [(c1, 1.50, '#D55E00', 'v'), (c2, 1.30, '#009E73', 'v'), (c3, -1.50, '#CC79A7', '^')]:
        cc = c[(c >= t0) & (c < t1)]
        ax.plot(cc, np.full(len(cc), y), mk, color=col, ms=4, clip_on=False)
    ax.set_xlim(t0, t1); ax.set_ylim(-1.6, 1.6)
    ax.set_ylabel('$A$')
    ax.text(0.005, 0.97, f'(1) $\\pm 0.5A_{{fp}}$ on $A$: {((c1>=t0)&(c1<t1)).sum()}', color='#D55E00', transform=ax.transAxes, va='top', fontsize=8)
    ax.text(0.005, 0.87, f'(2) $\\pm A_{{fp}}$ on $A$: {((c2>=t0)&(c2<t1)).sum()}', color='#009E73', transform=ax.transAxes, va='top', fontsize=8)
    ax.text(0.005, 0.03, f'(3) $\\pm 0.5A_{{fp}}$ on 50 t.u. mean: {((c3>=t0)&(c3<t1)).sum()}', color='#CC79A7', transform=ax.transAxes, va='bottom', fontsize=8)
    ax.set_xlabel('$t$')
    ax.set_title(f'{name}: total crossings (1) {len(c1)}, (2) {len(c2)}, (3) {len(c3)}   [t >= 0]', fontsize=9)
    print(f'  {name}, t in [{t0:g}, {t1:g}]: crossings of the whole series {len(c1)} {len(c2)} {len(c3)}')
