"""Free-mode reader of the JED sweep extraction plus the 3 x 3 histogram grid
of M (or |M|) over every (Re, sigma) pair.

First part: load every run of one (run_type, Re, sigma[, S, Np, eps]) and print
the pooled sample count of M after tmin (the loaders load_dns / load_do are the
reference for what grid.h5 / measures.h5 hold). Second part: one panel per
(Re, sigma) pair, DNS = every seed pooled, DO = every (S, Np, eps) run of that
pair, one curve each; pairs without DO just show the DNS. Densities, so DNS
(thousands of samples) and DO (millions) compare.

File: OUT/hist_{M,absM}_sweep.png (+ .cmd, figsave.py)

Usage:
    python3 jed_sweep_freemode.py [--sweep DIR] [--re 90] [--sig weak|mid|strong]
        [--run-type dns|do] [--seed all|N] [--S 6] [--Np 1000] [--eps 0.01] [--absM]
        [--out cases/se/figures/sweep]
"""
import argparse, glob, sys, h5py, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRAP = np.trapezoid if hasattr(np, "trapezoid") else np.trapz   # numpy 2 has no np.trapz

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from figsave import save  # noqa: E402

REPO = HERE.parents[1]
SWEEP = REPO / 'cases/se/runs/2026_09_13_jed_sweep'
FIGS = REPO / 'cases/se/figures'

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--sweep', type=Path, default=SWEEP, help='sweep root holding the Re<Re>_sig<sig>/ pair directories')
ap.add_argument('--re', type=int, default=90)
ap.add_argument('--sig', default='weak', choices=('weak', 'mid', 'strong'))
ap.add_argument('--run-type', default='dns', choices=('dns', 'do'))
ap.add_argument('--seed', default='all', help="int for one seed, or 'all' to pool every seed of this (run_type, re, sig)")
ap.add_argument('--S', type=int, default=6, help='do only: which convergence run type (base = 6, 1000, 0.01); ignored for dns')
ap.add_argument('--Np', type=int, default=1000)
ap.add_argument('--eps', type=float, default=0.01)
ap.add_argument('--tmin', type=float, default=300.0, help='initial transient dropped (every run starts at the fixed point of its Re)')
ap.add_argument('--absM', action='store_true', help='histogram figure: |M| instead of signed M')
ap.add_argument('--out', type=Path, default=FIGS / 'sweep')
args = ap.parse_args()
re, sig, run_type, seed, S, Np, eps, tmin, absM = (args.re, args.sig, args.run_type, args.seed, args.S, args.Np,
                                                    args.eps, args.tmin, args.absM)
ROOT = str(args.sweep)

sig_dict = {
    'weak': 0.0072,
    'mid': 0.0144,
    'strong': 0.0288
}

# ---------------------------------------------------------------- which files
# runs live in ROOT/Re<re>_sig<sig>/ (one directory per (Re, sigma) pair):
# dns:  dns_Re<re>_sig<sig>_seed<seed>/output/extract/grid.h5
# do:   do_Re<re>_sig<sig>_S<S>_Np<Np>_seed<seed>[_eps<eps>]/output/extract/measures.h5
if run_type == 'do':
    tail = 'measures.h5'
    cfg = f'_S{S}_Np{Np}'
    suffix = '' if eps == 0.01 else f'_eps{eps}'
else:
    tail = 'grid.h5'
    cfg = suffix = ''
pattern = f'{ROOT}/Re{re}_sig{sig_dict[sig]}/{run_type}_Re{re}_sig{sig_dict[sig]}{cfg}_seed{"*" if seed == "all" else seed}{suffix}/output/extract/{tail}'
files = sorted(glob.glob(pattern))
if run_type == 'do' and suffix == '':
    files = [f for f in files if '_eps' not in f]      # seed* would also catch the _eps runs
assert files, f'nothing matches {pattern}'
print(f'{len(files)} run(s):', *[f.split('/')[-4] for f in files], sep='\n  ')


# ---------------------------------------------------------------- loaders
def load_dns(f):
    """Everything in grid.h5 except the u, v planes (T, 121, 901): those stay
    on disk, read one snapshot with h5py.File(f)['u'][k]."""
    with h5py.File(f, 'r') as h:
        return dict(
            t=h['t'][:],                 # (T,)  every 4 t.u.
            M=h['M_grid'][:],            # (T,)  Ducimetiere M from the centreline v
            A=h['A'][:],                 # (T,)  <u, phi_Re>, phi = antisym part of the fixed point
            dL=h['dL'][:],               # (T,)  reattachment length upper - lower
            norm2=h['norm2'][:],         # (T,)  energy of the plane field
            v_axis=h['v_axis'][:],       # (T, 801)      v on y = 0, x in [0, 40]
            u_wall=h['u_wall'][:],       # (T, 2, 800)   u one row off each wall (lower, upper)
            p_wall=h['p_wall'][:],       # (T, 2, 801)   pressure on each wall
        )


def load_do(f):
    """Everything in measures.h5 except the mode/mean lines on the walls and
    the plane snapshots (group grid/, every 1000 t.u.). Adds the per-particle
    M_p and A_p, rebuilt from the mean, the modes and the Yi."""
    with h5py.File(f, 'r') as h:
        d = dict(
            t=h['t'][:],                        # (T,)  every 1 t.u.
            yi=h['yi'][:],                      # (T, Np, S)  particle coefficients
            a=h['a'][:],                        # (T, S)      <mean, mode_i>
            G=h['G'][:],                        # (T, S, S)   <mode_i, mode_j>  (identity on the plane)
            A0=h['A0'][:],                      # (T,)        <mean, phi>
            Ai=h['Ai'][:],                      # (T, S)      <mode_i, phi>
            v_axis_mean=h['v_axis_mean'][:],    # (T, 801)
            v_axis_modes=h['v_axis_modes'][:],  # (T, S, 801)
            S=int(h.attrs['S']),
        )
    # per-particle measures: field of particle p = mean + sum_i Y_pi mode_i
    d['A_p'] = d['A0'][:, None] + np.einsum('tps,ts->tp', d['yi'], d['Ai'])         # (T, Np)
    x = np.linspace(0.0, 40.0, 801)
    i2 = int(np.argmin(np.abs(x - 2.0)))                                             # sensor x = 2
    T = len(d['t'])
    Mp = np.empty((T, d['yi'].shape[1]), dtype=np.float32)
    for a_ in range(0, T, 100):                                                      # chunked: (100, Np, 801)
        b = min(a_ + 100, T)
        v = d['yi'][a_:b].astype(np.float32) @ d['v_axis_modes'][a_:b] + d['v_axis_mean'][a_:b][:, None, :]
        Mp[a_:b] = np.sign(v[..., i2]) * np.sqrt(TRAP(v ** 2, x, axis=-1))
    d['M_p'] = Mp                                                                    # (T, Np)
    return d


runs = [(load_do if run_type == 'do' else load_dns)(f) for f in files]

# ---------------------------------------------------------------- pooled samples (t >= tmin)
# per seed: keep the series separate (time-series plots, autocorrelations);
# pooled: concatenate the samples (pdfs, histograms, rates)
if run_type == 'dns':
    M_samples = np.concatenate([r['M'][r['t'] >= tmin] for r in runs])
    A_samples = np.concatenate([r['A'][r['t'] >= tmin] for r in runs])
    T_total = sum(float(r['t'][-1] - tmin) for r in runs)
else:
    r = runs[0]
    w = r['t'] >= tmin
    M_samples = r['M_p'][w].ravel()          # all particles, all times
    A_samples = r['A_p'][w].ravel()
    T_total = float(r['t'][-1] - tmin) * r['yi'].shape[1]   # particle-time

print(f'{len(M_samples)} samples of M after tmin, total observed time {T_total:.0f}, median |M| {np.median(np.abs(M_samples)):.4f}')

# ---------------------------------------------------------------- histogram of M (or |M|) for every (Re, sigma)
RES = (90, 100, 110)
SIGS = ('weak', 'mid', 'strong')
fig, axes = plt.subplots(len(RES), len(SIGS), figsize=(11, 8), sharex=True, constrained_layout=True)
for i, re_ in enumerate(RES):
    for j, sg in enumerate(SIGS):
        ax = axes[i, j]
        sv = sig_dict[sg]
        dns_files = sorted(glob.glob(f'{ROOT}/Re{re_}_sig{sv}/dns_Re{re_}_sig{sv}_seed*/output/extract/grid.h5'))
        do_files = sorted(glob.glob(f'{ROOT}/Re{re_}_sig{sv}/do_Re{re_}_sig{sv}_*/output/extract/measures.h5'))
        if dns_files:
            rr = [load_dns(f) for f in dns_files]
            x = np.concatenate([r['M'][r['t'] >= tmin] for r in rr])
            ax.hist(np.abs(x) if absM else x, bins=60, density=True, histtype='step', lw=1.4,
                    color='#0072B2', label=f'DNS, {len(rr)} seed(s)')
        for f in do_files:
            r = load_do(f)
            x = r['M_p'][r['t'] >= tmin].ravel()
            name = f.split('/')[-4].split('_sig')[1].split('_', 1)[1].replace('_seed45', '')   # e.g. S6_Np1000 or S6_Np1000_eps0.0001
            ax.hist(np.abs(x) if absM else x, bins=60, density=True, histtype='step', lw=1.0, label=f'DO {name}')
        ax.set_title(f'Re {re_}, sigma {sv}', fontsize=9)
        if dns_files or do_files:
            ax.legend(fontsize=6, frameon=False)
        if i == len(RES) - 1:
            ax.set_xlabel('|M|' if absM else 'M')
out = save(fig, args.out / f'hist_{"absM" if absM else "M"}_sweep.png', dpi=150)
print(out)
