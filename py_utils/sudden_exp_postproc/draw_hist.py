"""Histogram of one quantity, one curve per run (panel.py --panel hist).

Quantities (q=): M (Ducimetiere, centreline v), A = <u, phi>, absM, Y1 (the
coordinate on DO mode 1, DO runs only), dL (bubble length upper - lower, the
reattachment rule of extract_lib), and the escape times (dwell between two
consecutive crossings) escape_A on A (trigger +-0.5 A_fp), escape_M on M
(campaign rule, +-c m), escape_Y1 (+-0.5 A_fp, as A: mode 1 is phi to 2%) and
escape_dL (+-0.5 dL_fp). A DNS curve pools the
seeds given to it; a DO curve pools every particle and every time of its run
(one particle with @P) and the forcing seeds (_frc) given to it. Densities, so
DNS (thousands of samples) and DO (millions) compare.

DNS in blue, the DO curves in the order given. The title holds what the DO
curves share of S, Np, eps and the label of each what differs.

    python3 panel.py --panel hist q=A 'seed??' S6_Np1000_seed45_eps0.00000001 S6_Np1000_seed45_eps0.0001 \
        S6_Np1000_seed45 --out cases/se/figures/Re100_sig0.0144/vary_eps_S6_Np1000/hist_A.png
(the figure vary_hists.py puts there; it draws every line of DO runs this way)
"""
import itertools
import matplotlib.pyplot as plt
import numpy as np

import se

RC, SIZE, DPI = {}, (6.5, 4.2), 150        # rcParams, size of one panel, dpi of the png

tmin = 300.0        # initial transient dropped (every run starts at the fixed point of its Re)
bins = 60
escape_bins = np.arange(0, 3001, 100)
M_WELL, C = 0.1706, 0.8                                                       # sweep_rates.py / do_rates.py, Re 100 well level
DL_FP = 4.95                      # dL of the Re 100 fixed point by the plane rule: grid.h5 dL at t = 0 of every DNS seed (2026-10-15)
XLABEL = {'M': 'M', 'A': 'A', 'absM': '|M|', 'Y1': '$Y_1$', 'dL': '$\\Delta L$',
          'escape_A': 'escape time on A (crossings of $\\pm${:.3f})',
          'escape_M': 'escape time on M (crossings of $\\pm${:.4f})',
          'escape_Y1': 'escape time on $Y_1$ (crossings of $\\pm${:.3f})',
          'escape_dL': 'escape time on $\\Delta L$ (crossings of $\\pm${:.3f})'}


def threshold(q, Re):
    """The crossing threshold of an escape-time quantity, None for the others."""
    return {'escape_A': 0.5 * se.a_fp(Re), 'escape_M': C * M_WELL, 'escape_Y1': 0.5 * se.a_fp(Re),
            'escape_dL': 0.5 * DL_FP}.get(q)


# ---------------------------------------------------------------- escape times
def dwells(t, X, h):
    """Durations between consecutive crossings of +-h (se.crossings), every column pooled."""
    out = [np.diff(tc) for tc in se.crossings(t, X, h) if len(tc) > 1]
    return np.concatenate(out) if out else np.zeros(0)


def samples(t, X, q, h):
    w = t >= tmin
    if q.startswith('escape'):
        return dwells(t[w], X[w], h)
    if q == 'absM':
        return np.abs(X[w]).ravel()
    return X[w].ravel()


# ---------------------------------------------------------------- one panel
def draw(ax, curves, q='M'):
    """curves: [(run directories pooled into one curve, particle or None), ...] of one pair (se.find)."""
    assert q in XLABEL, f'q={q}: one of {", ".join(XLABEL)}'
    P = [[se.params(r) for r in runs] for runs, _ in curves]
    RE, SIG = P[0][0]['Re'], P[0][0]['sig']
    assert q != 'escape_M' or RE == 100, 'M_WELL is the Re 100 well level (sweep_rates.py has the other Re)'
    h = threshold(q, RE)                                                      # crossing threshold
    fixed, vary = se.fixed_and_varied(curves)
    with_frc = any(a['frc'] for pr in P for a in pr if a['kind'] == 'do')     # then every DO label names its forcing seed
    colors = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'][1:])     # skip the default blue (DNS colour)
    blue = iter(['#0072B2'])
    for (runs, p), pr in zip(curves, P):
        rd = [se.series(r, q.replace('escape_', '').replace('abs', ''), p) for r in runs]   # the series behind q
        x = np.concatenate([samples(t, X, q, h) for t, X in rd])
        if pr[0]['kind'] == 'dns':
            seeds = [r.name.split('_seed')[1] for r in runs]
            label = f'DNS, seeds {", ".join(seeds)} ({len(x)} samples)'
            kw = dict(lw=1.6, color=next(blue, None) or next(colors))        # a second DNS curve takes the next colour
        else:
            assert len({(a['S'], a['Np'], a['eps']) for a in pr}) == 1, 'one DO curve pools the forcing seeds of one (S, Np, eps)'
            frcs = [str(a['frc'] or a['seed']) for a in pr] if with_frc else []      # no _frc: forcing seed = seed
            t_ends = ', '.join(f'{t:g}' for t in sorted({float(t[-1]) for t, _ in rd}))
            label = 'DO ' + ', '.join(f'{k} {pr[0][k]:g}' for k in (vary or fixed)) \
                + (f', forcing seed{"s" * (len(frcs) > 1)} {", ".join(frcs)}' if frcs else '') \
                + (f', particle {p}' if p is not None else '') + f', t_end {t_ends} ({len(x)} samples)'
            kw = dict(lw=1.1, color=next(colors))
        ax.hist(x, bins=escape_bins if q.startswith('escape') else bins, density=True, histtype='step', label=label, **kw)
        print(f'  {label}')
    ax.set_xlabel(XLABEL[q].format(h)); ax.set_ylabel('density')
    what = [f'{k} = {v:g}' for k, v in fixed.items()] + [f'{k} varied' for k in vary]
    ax.set_title(f'Re {RE}, sigma {SIG}' + ('; ' + ', '.join(what) if what else ''), fontsize=9)
    ax.legend(fontsize=7, frameon=False)
