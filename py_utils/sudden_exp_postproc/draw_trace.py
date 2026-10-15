"""A, M, Delta L or Y_1 against time, one line per run: a DNS in gray,
particle P of a DO run (RUN@P) in blue (panel.py --panel trace).

Made for a DNS twin against the DO particle it copies: a twin DNS starts from
the initial field of particle 0 of a DO run and receives the same noise at
every step, so the two agree at t = 0 and part only because the particle
lives in mean + S modes. eps enters the DO dynamics, not the initial
condition, so one twin is compared with the eps 1e-4 and the eps 1e-8 DO runs
of the same (S, Np).

One panel per quantity (q=A, M, dL, Y1): DNS in gray, the DO runs in blue in
the order given (first solid, second dashed). The DNS analogue of Y_1 exists
only where the DO mean and modes were archived on the plane (grid/t, every
1000 t.u.): y1_dns = <u_dns - ubar, u_1> with the mean and first mode of each
DO run, drawn as open markers.

Printed: the t = 0 agreement of the quantity and, for A, the first time at
which |A_dns - A_p| exceeds 0.05 and 0.2 (DO sampled at the DNS times).

Measures come from cases/se/templates/extract_lib.py (ip, m_grid,
reattachment) through se.py; nothing is recomputed here.

    python3 panel.py --sharex --tags \
        --panel trace q=A seed45_Np1000_S8p0 S8_Np1000_seed45_eps1e-4@0 S8_Np1000_seed45_eps1e-8@0 \
        --panel trace q=M --panel trace q=dL --panel trace q=Y1 \
        --out cases/se/figures/Re100_sig0.0144/dns_Re100_sig0.0144_seed45_Np1000_S8p0/twin.pdf
"""
import h5py
import numpy as np
from matplotlib.transforms import offset_copy

import se

RC = {"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9}
SIZE, DPI = (6.5, 1.75), 200                # size of one panel, dpi of the png

STYLE = [('-', 'o', 'circles'), ('--', 's', 'squares')]     # line, marker of the DNS projection: DO runs in the order given
YLABEL = {'A': '$A$', 'M': '$M$', 'dL': r'$\Delta L$', 'Y1': '$Y_1$'}
BLUE, GRAY = '#0072B2', '0.45'


def y1_dns(f, g):
    """<u_dns - ubar, u_1> at the instants where the DO plane was archived."""
    tg, td = g['grid/t'][:], f['t'][:]
    out = []
    for j, tj in enumerate(tg):
        i = int(np.argmin(np.abs(td - tj)))
        assert td[i] == tj, f'no DNS snapshot at t = {tj}'
        du = f['u'][i].astype(float) - g['grid/mean_u'][j]
        dv = f['v'][i].astype(float) - g['grid/mean_v'][j]
        out.append(se.ip(du, dv, g['grid/mode_u'][j, 0], g['grid/mode_v'][j, 0]))
    return tg, np.array(out)


def end_labels(ax, items, sep=0.10):
    """Text at the right edge, level with each curve's end, pushed apart vertically."""
    y0, y1 = ax.get_ylim()
    pos = np.array([(y - y0) / (y1 - y0) for y, _, _ in items])
    order = np.argsort(pos)
    for a, b in zip(order[:-1], order[1:]):
        pos[b] = max(pos[b], pos[a] + sep)
    pos -= max(0.0, pos.max() - 0.95)
    tr = offset_copy(ax.get_xaxis_transform(), fig=ax.figure, x=3, units='points')
    for (_, s, c), p in zip(items, pos):
        ax.text(ax.get_xlim()[1], p, s, color=c, transform=tr, ha='left', va='center', fontsize=8, clip_on=False)


def draw(ax, curves, q='A'):
    """curves: [([run directory], particle or None), ...] of one pair (se.find)."""
    assert q in YLABEL, f'q={q}: one of {", ".join(YLABEL)}'
    assert all(len(runs) == 1 for runs, _ in curves), 'trace: one run per line, nothing pooled'
    dns = [runs[0] for runs, _ in curves if se.params(runs[0])['kind'] == 'dns']
    do = [(runs[0], p) for runs, p in curves if se.params(runs[0])['kind'] == 'do']
    assert all(p is not None for _, p in do), 'trace: a DO run needs a particle, RUN@P'
    assert len(do) <= len(STYLE), f'trace: at most {len(STYLE)} DO runs in a panel (STYLE)'
    fixed, vary = se.fixed_and_varied(curves)
    labels, names = [], []
    if q != 'Y1':
        for run in dns:
            td, x = se.series(run, q)
            ax.plot(td, x[:, 0], color=GRAY, lw=0.8)
            labels.append((x[-1, 0], 'DNS', GRAY))
            print(f'  {run.name}: t in [{td[0]:g}, {td[-1]:g}], {td.size} rows, NaN in {q}: {int(np.isnan(x).sum())}')
    for (run, p), (ls, mk, shape) in zip(do, STYLE):
        t, x = se.series(run, q, p); x = x[:, 0]
        name = 'DO ' + ', '.join(f'{k} ' + (se.eps_str(se.params(run)[k]) if k == 'eps' else f'{se.params(run)[k]:g}')
                                 for k in (vary or fixed)) + (f', particle {p}' if len({pp for _, pp in do}) > 1 else '')
        names.append(f'{shape} {name[3:]}')
        ax.plot(t, x, color=BLUE, lw=0.8, ls=ls)
        labels.append((x[-1], name, BLUE))
        print(f'  {run.name}@{p}: t in [{t[0]:g}, {t[-1]:g}], {t.size} rows, NaN in {q}: {int(np.isnan(x).sum())}')
        for d in dns:
            if q == 'Y1':
                with h5py.File(se.h5(d), 'r') as f, h5py.File(se.h5(run), 'r') as g:
                    tg, y1d = y1_dns(f, g)
                ax.plot(tg, y1d, mk, mfc='none', mec=GRAY, mew=0.8, ms=3.5, ls='none')
                if len(labels) == 1:                                   # the DNS label once, at the markers of the first DO run
                    labels.append((y1d[-1], 'DNS', GRAY))
                x0 = y1d[0]
            else:
                td, xd = se.series(d, q); xd = xd[:, 0]; x0 = xd[0]
            print(f'   t=0 {q:>2s}: DNS {x0:+.6f}  DO {x[0]:+.6f}  diff {x0-x[0]:+.2e}')
            if q == 'A':
                # DO sampled at the DNS times (DO t is every 1 t.u., DNS every 4)
                it = np.rint(td).astype(int); ok = it <= int(t[-1])
                assert np.array_equal(t[it[ok]], td[ok]), 'DO and DNS time grids do not line up'
                dd = np.abs(xd[ok] - x[it[ok]])
                for thr in (0.05, 0.2):
                    hit = np.flatnonzero(dd > thr)
                    print(f'   t* |A_dns - A_{p}| > {thr}: ' + (f't = {td[ok][hit[0]]:g}' if hit.size else f'never (t <= {td[ok][-1]:g})'))
    ax.set_ylabel(YLABEL[q]); ax.set_xlabel('$t$')
    ax.axhline(0, color='0.8', lw=0.4)
    tx = se.dns(dns[0])['t'] if dns else se.do_one(*do[0])['t']        # the time span of the DNS when there is one
    ax.set_xlim(tx[0], tx[-1])
    end_labels(ax, labels)
    if q == 'Y1' and dns:
        # a free strip under the Y_1 curves for the marker note
        y0, y1 = ax.get_ylim(); tk = ax.get_yticks()
        ax.set_yticks(tk[(tk >= y0) & (tk <= y1)]); ax.set_ylim(y0 - 0.2 * (y1 - y0), y1)
        ax.text(0.5, 0.02, f'open markers: DNS projected on mode 1 of each DO run ({", ".join(names)})',
                color=GRAY, fontsize=7, transform=ax.transAxes, ha='center', va='bottom')
