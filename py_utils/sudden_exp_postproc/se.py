"""The basic tools of the sudden-expansion (se) campaign, in one module.

From a console, a script or PyCharm, in any directory: import se
(py_utils is installed once with pip, py_utils/README.md).

Sections:

  plane        the 901 x 121 extraction plane, x in [-5, 40], y in [-1.5, 1.5]:
               x, y, X, Y, DX, DY, INSIDE (False on the inlet walls), and the
               measures on it: ip (area inner product), m_grid (M), reattachment
               (dL), mirror, antisym. They are defined in
               cases/se/templates/extract_lib.py, the module JED runs for the
               extraction; se re-exports them, so each is defined once.
  runs         run directories of the JED sweep: params(run) reads the name,
               h5(run) is its measures file, find(pair, 'seed45') finds runs.
  rules        C, M_WELL, DL_FP, a_fp(Re): the thresholds of every transition count.
  load         load(f): every dataset of a measures file in a dict.
  series       dns(run), do_all(run), do_one(run, p), series(run, q, p):
               A, M, dL, Y1 against time.
  particle     particle_plane(h, p, k): u, v of DO particle p on the plane.
  derivatives  wall_dudn(u), dL_shear(u), vorticity(u, v).
  crossings    crossings(t, X, h): the hysteresis rule of every transition count.
  raw runs     the archives themselves (FieldConvert on them): FIELDCONVERT,
               fc_env, interp, casefile_param, load_dat, yi_from_fld, yi_series.

Examples:

    import h5py, matplotlib.pyplot as plt, se
    run = se.SWEEP / 'Re100_sig0.0144/dns_Re100_sig0.0144_seed45'
    f = h5py.File(se.h5(run), 'r')
    d = se.load(f)                      # d['t'], d['A'], d['M_grid'], d['u'] (u, v: slice them)
    tau = se.wall_dudn(d['u'][1000])    # (2, 901): du/dn on the lower and upper wall
    plt.plot(se.x, tau[0]); plt.plot(se.x, tau[1])

    g = h5py.File(se.h5(se.SWEEP / 'Re100_sig0.0144/do_Re100_sig0.0144_S8_Np1000_seed45_eps0.0001'), 'r')
    u, v = se.particle_plane(g, p=6, k=3)        # particle 6 at g['grid/t'][3]

    t, X = se.series(run, 'A')          # A(t) of the DNS, X of shape (T, 1)
    tc = se.crossings(t, X[:, 0], 0.5 * se.a_fp(100))
"""
import functools
import glob
import os
import re
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SWEEP = REPO / 'cases/se/runs/2026_09_13_jed_sweep'
TPL = REPO / 'cases/se/templates'
FIGS = REPO / 'cases/se/figures'
sys.path.insert(0, str(TPL))
import extract_lib as xl  # noqa: E402

TRAP = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz

# ---------------------------------------------------------------- plane (extract_lib)
NX, NY, x, y, X, Y, DX, DY = xl.NX, xl.NY, xl.x, xl.y, xl.X, xl.Y, xl.DX, xl.DY
INSIDE, W, BOX = xl.INSIDE, xl.W, xl.BOX                 # inlet-wall mask, area weights, FieldConvert box
JAXIS, JLOW, JUP = xl.JAXIS, xl.JLOW, xl.JUP             # rows: centreline, first row off each wall
XPOS, XWALL, I2 = xl.XPOS, xl.XWALL, xl.I2               # columns x >= 0, x > 0; sensor x = 2 within XPOS
ip, m_grid, reattachment = xl.ip, xl.m_grid, xl.reattachment
mirror, antisym, read_dat = xl.mirror, xl.antisym, xl.read_dat
yi_from_fld = xl.yi_from_info                            # (Np, S) Yi from the metadata of one DO archive

# ---------------------------------------------------------------- runs of the JED sweep
# A run is a directory of one (Re, sigma) pair, SWEEP/Re<Re>_sig<sig>/<name>/:
#   dns_Re<Re>_sig<sig>_seed<seed>                    forced DNS
#   dns_Re<Re>_sig<sig>_seed<seed>_Np<Np>_S<S>p0      DNS twin of particle 0 of the DO run (S, Np, seed)
#   do_Re<Re>_sig<sig>_S<S>_Np<Np>_seed<seed>[_eps<eps>][_frc<fseed>]
#       no _eps suffix = the default 0.01; _frc<fseed> = forcing seed differs from the POD seed
# Its measures file (h5): output/extract/grid.h5 (DNS) or measures.h5 (DO), the python extraction JED ran on
# the archives (cases/se/templates/extract.sbatch), or output/measures/measures.h5, written by the solver's
# DOMeasures filter (exact quadrature, format DO_MEASURES_V1). The series functions read either.
# On a command line (panel.py) a run is its directory name or the part after
# dns_<pair>_ / do_<pair>_ (seed45, S6_Np1000_seed45_eps1e-4). One argument
# is one curve: a glob ('seed*') or names joined by + pool every match into
# that curve, and @P after a DO run takes particle P instead of every particle.
_DNS = re.compile(r'dns_Re(\d+)_sig([0-9.]+)_seed(\d+)(?:_Np(\d+)_S(\d+)p(\d+))?$')
_DO = re.compile(r'do_Re(\d+)_sig([0-9.]+)_S(\d+)_Np(\d+)_seed(\d+)(?:_eps([0-9.]+))?(?:_frc(\d+))?$')


def params(run):
    """What the name of a run directory says."""
    name = Path(run).name
    m = _DNS.match(name)
    if m:
        return dict(kind='dns', Re=int(m.group(1)), sig=m.group(2), seed=int(m.group(3)),
                    twin=m.group(4) is not None)               # twin: the DNS copy of particle 0 of a DO run
    m = _DO.match(name)
    assert m, f'{name}: not a run of the sweep'
    return dict(kind='do', Re=int(m.group(1)), sig=m.group(2), S=int(m.group(3)), Np=int(m.group(4)), seed=int(m.group(5)),
                eps=float(m.group(6)) if m.group(6) else 0.01,
                frc=int(m.group(7)) if m.group(7) else None)


def eps_str(eps):
    """eps as it is written everywhere but in the run directories: 1e-4, 1e-8, 1e-2."""
    return f'{eps:.0e}'.replace('e-0', 'e-')


def short_eps(name):
    """A run name with its eps written as eps_str: ..._eps0.0001 -> ..._eps1e-4."""
    return re.sub(r'eps([0-9.e-]+)', lambda m: 'eps' + eps_str(float(m.group(1))), name)


def disk_eps(name):
    """A run name with its eps as the run directories spell it: eps1e-4 -> eps0.0001, eps1e-2 -> nothing."""
    def spell(m):
        eps = float(m.group(2))
        return '' if eps == 0.01 else m.group(1) + 'eps' + np.format_float_positional(eps, trim='-')
    return re.sub(r'(_?)eps([0-9.e-]+)', spell, name)


def h5(run):
    """The measures file of a run: the python extraction, output/extract/grid.h5 (DNS) or measures.h5 (DO), when
    the run has it, else the DOMeasures filter's output/measures/measures.h5."""
    run = Path(run)
    f = run / 'output/extract' / ('grid.h5' if params(run)['kind'] == 'dns' else 'measures.h5')
    return f if f.exists() else run / 'output/measures/measures.h5'


def is_filter(h):
    """True when the open h5py.File h was written by the DOMeasures filter, False for the python extraction."""
    return 'header' in h


def pair_dir(pair):
    """The directory of one (Re, sigma) pair: a path, or its name under SWEEP (Re100_sig0.0144)."""
    p = SWEEP / pair if (SWEEP / pair).is_dir() else Path(pair)      # the sweep's pair first: cases/se/figures/ has the same names
    assert p.is_dir() and re.match(r'Re(\d+)_sig([0-9.]+)$', p.name), f'{pair}: not a Re<Re>_sig<sig> directory'
    return p


def find(pair, arg):
    """One curve of a panel: (run directories pooled into it, particle or None)."""
    names, _, p = arg.partition('@')
    names = disk_eps(names)                                       # eps typed as 1e-4 or as the directory spells it
    runs = []
    for name in names.split('+'):
        hits = sorted(d for pat in (name, f'dns_{pair.name}_{name}', f'do_{pair.name}_{name}') for d in pair.glob(pat)
                      if (_DNS.match(d.name) or _DO.match(d.name)) and h5(d).exists())
        assert hits, f'{name}: no run in {pair}'
        runs += hits
    return runs, (int(p) if p else None)


def runs(kind, pairs=None):
    """Every run of one kind ('dns' or 'do') that has its extraction file, in the pairs given (names under SWEEP
    or paths; default every pair of SWEEP). Runs moved out of SWEEP (2026_09_13_jed_sweep_absTol) are not seen."""
    pairs = [pair_dir(p) for p in pairs] if pairs else sorted(p for p in SWEEP.glob('Re*_sig*') if p.is_dir())
    rx = _DNS if kind == 'dns' else _DO
    return [d for p in pairs for d in sorted(p.glob(f'{kind}_*')) if rx.match(d.name) and h5(d).exists()]


def fixed_and_varied(curves):
    """Of S, Np, eps: those every DO curve shares (name -> value) and the names of those that differ."""
    vals = {k: sorted({params(r)[k] for runs, _ in curves for r in runs if params(r)['kind'] == 'do'}) for k in ('S', 'Np', 'eps')}
    return {k: v[0] for k, v in vals.items() if len(v) == 1}, [k for k, v in vals.items() if len(v) > 1]


# ---------------------------------------------------------------- rules of every transition count
C = 0.8             # a transition: M (or A, dL, Y1) crosses -C * well from beyond +C * well, or the reverse (crossings)
M_WELL = 0.1706     # |M| of the Re 100 well: median |M| of the sigma 0.0072 DNS (sweep_rates.py has the other Re)
DL_FP = 4.95        # dL of the Re 100 fixed point on the plane: grid.h5 dL at t = 0 of every DNS seed (2026-10-15)


def a_fp(Re):
    """A of the unforced fixed point of that Re (0.701 at Re 100)."""
    return float(np.load(TPL / f'phi_Re{Re}.npz')['antisym_norm'])


# ---------------------------------------------------------------- load
def load(f, big=1e8):
    """Every dataset of a measures file as {name: value}, plus its attributes
    (mode = 'dns' or 'do' of the python extraction; format, n_modes, n_particles ... of the filter).

    f: an open h5py.File, or a path (then opened read-only and left open).
    Datasets under big bytes are read into numpy arrays; the larger ones (u, v
    of a DNS, about 870 MB each; yi of a DO run, 500 MB and up) stay h5py
    datasets, which read only the part sliced: d['u'][1000] is one instant.
    Group paths become prefixes: grid/mean_u -> grid_mean_u.

        globals().update(se.load(f))      # every dataset as a console variable
    """
    if not isinstance(f, h5py.File):
        f = h5py.File(f, 'r')
    out = dict(f.attrs)
    if is_filter(f):
        out.update(f['header'].attrs)

    def take(name, d):
        if isinstance(d, h5py.Dataset):
            out[name.replace('/', '_')] = d[()] if d.nbytes < big else d

    f.visititems(take)
    return out


# ---------------------------------------------------------------- series
@functools.lru_cache(maxsize=None)
def dns(run):
    """t, M, A, dL of a DNS, each (T,) (the u, v planes stay on disk)."""
    with h5py.File(h5(run), 'r') as h:
        if is_filter(h):                                                     # one particle
            return dict(t=h['t'][:], M=h['M_p'][:, 0], A=h['A_p'][:, 0], dL=h['dL_p'][:, 0])
        return dict(t=h['t'][:], M=h['M_grid'][:], A=h['A'][:], dL=h['dL'][:])


@functools.lru_cache(maxsize=None)
def do_all(run):
    """t (T,) and the per-particle M_p and A_p, (T, Np): the filter's, or rebuilt from the mean, the modes and the Yi."""
    with h5py.File(h5(run), 'r') as h:
        if is_filter(h):
            return dict(t=h['t'][:], M=h['M_p'][:], A=h['A_p'][:])
        t, Y = h['t'][:], h['yi'][:]
        vm, phi = h['v_axis_mean'][:], h['v_axis_modes'][:]
        A0, Ai = h['A0'][:], h['Ai'][:]
    # int v_p^2 dx is quadratic in Y_p: c + 2 b.Y + Y^T G Y with c = int vbar^2, b_i = int vbar phi_i,
    # G_ij = int phi_i phi_j (trapz weights), so the (T, Np, 801) axis lines are never formed.
    xa = np.linspace(0.0, 40.0, 801)
    i2 = int(np.argmin(np.abs(xa - 2.0)))                                            # sensor x = 2
    w = np.full(801, xa[1] - xa[0]); w[[0, -1]] *= 0.5                               # trapz weights
    c = np.einsum('ti,i,ti->t', vm, w, vm)
    b = np.einsum('ti,i,tsi->ts', vm, w, phi)
    G = np.einsum('tsi,i,tri->tsr', phi, w, phi)
    q = c[:, None] + 2 * np.einsum('tps,ts->tp', Y, b) + np.einsum('tps,tsr,tpr->tp', Y, G, Y)
    v2 = vm[:, i2][:, None] + np.einsum('tps,ts->tp', Y, phi[:, :, i2])
    return dict(t=t, M=(np.sign(v2) * np.sqrt(np.maximum(q, 0.0))).astype(np.float32),
                A=(A0[:, None] + np.einsum('tps,ts->tp', Y, Ai)).astype(np.float32))


@functools.lru_cache(maxsize=None)
def do_all_Y1(run):
    """t (T,) and the per-particle Y_1 (T, Np)."""
    with h5py.File(h5(run), 'r') as g:
        return dict(t=g['t'][:], Y1=g['Y1_p'][:] if is_filter(g) else g['yi'][:, :, 0])


@functools.lru_cache(maxsize=None)
def do_all_dL(run, block=50):
    """t (T,) and the per-particle Delta L (T, Np), from the near-wall u of every particle (mean + Y . modes, the
    reattachment rule of extract_lib on every row at once), in blocks of instants: the whole product would be
    T x Np x 2 x 800 floats. About a minute per run. The filter's dL_p is read as is."""
    with h5py.File(h5(run), 'r') as g:
        if is_filter(g):
            return dict(t=g['t'][:], dL=g['dL_p'][:])
        t, Y = g['t'][:], g['yi'][:]
        um, modes = g['u_wall_mean'], g['u_wall_modes']                  # (T, 2, 800) and (T, S, 2, 800), on disk
        T, Np, S = Y.shape
        xw = xl.x[xl.XWALL]
        dL = np.empty((T, Np), dtype=np.float32)
        for a in range(0, T, block):
            b = slice(a, min(a + block, T))
            n = b.stop - b.start
            uw = um[b].reshape(n, 1, 1600) + Y[b] @ modes[b].reshape(n, S, 1600)   # (n, Np, 1600): lower wall, upper wall
            neg = (uw < 0).reshape(n, Np, 2, 800)
            k = 799 - np.argmax(neg[..., ::-1], axis=-1)                   # last column with u < 0; none -> 799 (x = 40)
            L = xw[k]
            dL[b] = L[..., 1] - L[..., 0]                                  # upper - lower
    return dict(t=t, dL=dL)


@functools.lru_cache(maxsize=None)
def do_one(run, p):
    """t, A, M, Delta L and Y_1 of particle p along a DO run, each (T,); M and Delta L from extract_lib (m_grid, reattachment)."""
    with h5py.File(h5(run), 'r') as g:
        t = g['t'][:]
        if is_filter(g):
            return dict(t=t, A=g['A_p'][:, p], M=g['M_p'][:, p], dL=g['dL_p'][:, p], Y1=g['Y1_p'][:, p])
        Y = g['yi'][:, p, :]                                       # (T, S), never the full yi
        A = g['A0'][:] + np.einsum('tk,tk->t', Y, g['Ai'][:])
        vax = g['v_axis_mean'][:] + np.einsum('tk,tkx->tx', Y, g['v_axis_modes'][:])
        M = np.array([m_grid(r) for r in vax])
        uw = g['u_wall_mean'][:] + np.einsum('tk,tkcx->tcx', Y, g['u_wall_modes'][:])
        dL = np.array([reattachment(r[1]) - reattachment(r[0]) for r in uw])   # upper - lower
    return dict(t=t, A=A, M=M, dL=dL, Y1=Y[:, 0])


def series(run, q, p=None):
    """t (T,) and the quantity q of one run, (T, n): n = 1 for a DNS and for particle p of a DO run, n = Np for every particle."""
    kind = params(run)['kind']
    assert kind == 'do' or p is None, f'{Path(run).name}: @{p} on a DNS'
    if kind == 'dns':
        d = dns(run)
    elif p is not None:
        d = do_one(run, p)
    else:
        d = {'Y1': do_all_Y1, 'dL': do_all_dL}.get(q, do_all)(run)
    assert q in d, f'{Path(run).name}: no {q} here (has {", ".join(k for k in d if k != "t")})'
    return d['t'], d[q].reshape(len(d['t']), -1)


# ---------------------------------------------------------------- one DO particle on the plane
def particle_plane(h, p, k):
    """u, v (NY, NX) of particle p of a DO run at h['grid/t'][k]: mean + sum_i Y_pi mode_i.

    h: the open measures.h5 of the python extraction (the filter keeps no planes). Only the instants of grid/t
    (every 1000 time units) keep the mean and the modes on the whole plane; between them measures.h5 holds
    the centreline and the near-wall rows only (do_one)."""
    assert not is_filter(h), 'particle_plane: the filter file has no planes'
    i = int(np.flatnonzero(h['step'][:] == h['grid/step'][k])[0])
    Yp = h['yi'][i, p]                                                    # (S,)
    u = h['grid/mean_u'][k] + np.einsum('i,iyx->yx', Yp, h['grid/mode_u'][k])
    v = h['grid/mean_v'][k] + np.einsum('i,iyx->yx', Yp, h['grid/mode_v'][k])
    return u, v


# ---------------------------------------------------------------- derivatives on the plane
def wall_dudn(u):
    """du/dn on the first row off each wall (|y| = 1.475), n pointing into the fluid.

    Central difference between the wall row (u = 0 there) and the second row
    off the wall, the shear criterion of the bubble length (dLs,
    asym_measures.py, 2026-09-01). u: (..., NY, NX) -> (..., 2, NX), with
    [..., 0, :] the lower wall and [..., 1, :] the upper wall; negative where
    the near-wall flow goes upstream (inside the bubble)."""
    u = np.asarray(u, dtype=float)
    low = (u[..., JLOW + 1, :] - u[..., JLOW - 1, :]) / (2 * DY)
    up = -(u[..., JUP + 1, :] - u[..., JUP - 1, :]) / (2 * DY)
    return np.stack([low, up], axis=-2)


def dL_shear(u):
    """Bubble length difference upper - lower (dLs) of one plane u (NY, NX):
    the reattachment rule of extract_lib (last negative column with x > 0)
    applied to wall_dudn instead of to the near-wall u."""
    tau = wall_dudn(u)
    return reattachment(tau[1, XWALL]) - reattachment(tau[0, XWALL])


def vorticity(u, v, dx=DX, dy=DY):
    """dv/dx - du/dy by second-order finite differences (np.gradient) on the
    last two axes, (..., NY, NX). The inlet walls of the plane are zeros, so
    the values next to the step corner mix flow and wall."""
    return np.gradient(v, dx, axis=-1, edge_order=2) - np.gradient(u, dy, axis=-2, edge_order=2)


# ---------------------------------------------------------------- crossings
def crossings(t, X, h):
    """Times at which X crosses from beyond +h to beyond -h or back.

    Hysteresis (Schmitt trigger): the state is the last threshold passed, and
    a crossing is the first sample beyond the other one. X (T,) -> array of
    times; X (T, n) -> list of n arrays, one per column. A transition count is
    len() of the array, an escape (dwell) time np.diff of it."""
    t, X = np.asarray(t), np.asarray(X)
    col = X[:, None] if X.ndim == 1 else X
    s = np.where(col > h, 1, np.where(col < -h, -1, 0)).astype(np.int8)
    idx = np.where(s != 0, np.arange(len(t))[:, None], 0)
    idx = np.maximum.accumulate(idx, axis=0)                     # index of the last nonzero state
    st = np.take_along_axis(s, idx, axis=0)                      # state carried forward (0 before the first hit)
    cross = (st[1:] != st[:-1]) & (st[:-1] != 0)
    out = [t[1:][cross[:, p]] for p in range(col.shape[1])]
    return out[0] if X.ndim == 1 else out


# ---------------------------------------------------------------- raw runs (no JED extraction)
FIELDCONVERT = os.environ.get('NEKTAR_FIELDCONVERT', '/home/isma/nektar_repro/build/utilities/FieldConvert/FieldConvert')


def fc_env():
    """Environment for FIELDCONVERT: on this laptop it exists only in the ASan
    build, so libasan is preloaded and leak detection is off."""
    env = os.environ.copy()
    try:
        asan = subprocess.run(['gcc', '-print-file-name=libasan.so'], capture_output=True, text=True).stdout.strip()
        if '/' in asan:
            env['LD_PRELOAD'] = asan
    except OSError:
        pass
    env['ASAN_OPTIONS'] = 'detect_leaks=0'
    return env


def interp(xml, fld, points, out, fieldconvert=FIELDCONVERT):
    """FieldConvert interppoints of the field file fld (on the mesh xml) at
    points ('box=...', 'plane=...', 'line=...'), written to the .dat out.
    Cached: nothing runs when out exists. Returns out."""
    out = Path(out)
    if not out.exists():
        subprocess.run([fieldconvert, '-f', '-m', f'interppoints:fromxml={xml}:fromfld={fld}:{points}', str(out)],
                       env=fc_env(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out


def casefile_param(run, name, default=None):
    """Value of the parameter name = ... in run/casefile.xml (a float; expressions like 1/100 are evaluated)."""
    txt = (Path(run) / 'casefile.xml').read_text()
    m = re.search(rf'{name}\s*=\s*([0-9eE.+-/]+)', txt)
    if not m:
        return default
    try:
        return float(eval(m.group(1), {'__builtins__': {}}))
    except Exception:
        return default


def load_dat(path, ncol):
    """Rows of a FieldConvert .dat with at least ncol numbers, first ncol columns (header lines skipped)."""
    rows = []
    for ln in Path(path).read_text().splitlines():
        p = ln.split()
        if len(p) >= ncol:
            try:
                rows.append([float(v) for v in p[:ncol]])
            except ValueError:
                pass
    return np.array(rows)


def yi_series(do_dir, nmodes, pattern='*.do_*.fld', dt=0.01):
    """t (T,) = step * dt and yi (T, Np, S) from every DO archive of a directory."""
    files = sorted(glob.glob(str(Path(do_dir) / pattern)))
    if not files:
        raise FileNotFoundError(f'no archives matching {pattern} in {do_dir}')
    steps = np.array([int(re.search(r'do_(\d+)', f).group(1)) for f in files])
    yi = np.stack([yi_from_fld(f, nmodes) for f in files])
    return steps * dt, yi
