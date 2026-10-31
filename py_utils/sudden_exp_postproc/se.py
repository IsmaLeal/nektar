"""The basic tools of the sudden-expansion (se) campaign, in one module.

From a console, a script or PyCharm, in any directory: import se
(py_utils is installed once with pip, py_utils/README.md).

Sections:

  plane        the 901 x 121 extraction plane, x in [-5, 40], y in [-1.5, 1.5]:
               x, y, X, Y, DX, DY, INSIDE (False on the inlet walls), read_dat. They
               are defined in cases/se/templates/extract_lib.py, the module JED runs
               for the extraction; se re-exports them, so each is defined once.
  runs         run directories of the JED sweep: params(run) reads the name,
               h5(run) is its measures file (the DOMeasures filter; before it, the
               extraction, whose series legacy.py computes), planes(run) its planes
               file (the JED extraction), find(pair, 'seed45') finds runs.
  rules        C, M_WELL, a_fp(Re), dl_fp(Re): the thresholds of every transition count.
  load         load(f): every dataset of a measures file in a dict.
  series       dns(run), do_all(run), do_one(run, p), series(run, q, p):
               A, M, dL, Y1 against time.
  crossings    crossings(t, X, h): the hysteresis rule of every transition count.

Examples:

    import h5py, matplotlib.pyplot as plt, se
    run = se.RUNS / 'Re100_sig0.0144/dns_Re100_sig0.0144_seed45'
    d = se.load(se.h5(run))             # d['t'], d['A_p'], d['M_p'] ...

    t, X = se.series(run, 'A')          # A(t) of the DNS, X of shape (T, 1)
    tc = se.crossings(t, X[:, 0], 0.5 * se.a_fp(100))
"""
import functools
import re
import sys
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUNS = REPO / 'cases/se/runs'
TPL = REPO / 'cases/se/templates'
FIGS = REPO / 'cases/se/figures'
sys.path.insert(0, str(TPL))
import extract_lib as xl  # noqa: E402
sys.path.insert(0, str(HERE))
import legacy  # noqa: E402                     # the series of the runs extracted before the filter

# ---------------------------------------------------------------- plane (extract_lib)
NX, NY, x, y, X, Y, DX, DY = xl.NX, xl.NY, xl.x_vals, xl.y_vals, xl.X, xl.Y, xl.DX, xl.DY
INSIDE, BOX, read_dat = xl.INSIDE, xl.BOX, xl.read_dat    # inlet-wall mask, FieldConvert box, its .dat reader

# ---------------------------------------------------------------- runs of the JED sweep
# A run is a directory of one (Re, sigma) pair, RUNS/Re<Re>_sig<sig>/<name>/ (RUNS = cases/se/runs; its
# folder cerrado/ holds the closed campaigns and is not part of the sweep):
#   dns_Re<Re>_sig<sig>_seed<seed>                    forced DNS
#   dns_Re<Re>_sig<sig>_seed<seed>_Np<Np>_S<S>p0      DNS twin of particle 0 of the DO run (S, Np, seed)
#   do_Re<Re>_sig<sig>_S<S>_Np<Np>_seed<seed>[_eps<eps>][_frc<fseed>]
#       no _eps suffix = the default 0.01; _frc<fseed> = forcing seed differs from the POD seed
# Inside a run: casefile.xml, run.log, series/ (the DOMeasures filter's measures.h5, energy.mdl), campos/ (the
# planes JED extracted from the archives, cases/se/templates/extract.sbatch: grid.h5 of a DNS, measures.h5
# of a DO run) and, on JED only, fuentes/ (chks, archives). Its measures file, h5(run), is series/measures.h5
# (exact quadrature, format DO_MEASURES_V1); its planes file, planes(run), is the campos/ one.
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


SYM = dict(Re=r'\mathrm{Re}', sig=r'\sigma', S='S', Np='N_p')


def label(run):
    p = params(run)
    keys = ('Re', 'sig', 'S', 'Np')[:2 if p['kind'] == 'dns' else 4]
    return p['kind'].upper() + ' ' + ', '.join(f'${SYM[k]} = {p[k]}$' for k in keys) + f' (seed {p["seed"]})'


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
    """The measures file of a run: series/measures.h5, written by the DOMeasures filter, when the run has it;
    else the planes file of the extraction before the filter (legacy.py computes the series from it)."""
    f = Path(run) / 'series/measures.h5'
    return f if f.exists() else planes(run)


def planes(run):
    """The planes file of a run: campos/grid.h5 (DNS: u, v) or campos/measures.h5 (DO: grid/)."""
    run = Path(run)
    return run / 'campos' / ('grid.h5' if params(run)['kind'] == 'dns' else 'measures.h5')


def pair_dir(pair):
    """The directory of one (Re, sigma) pair: a path, or its name under RUNS (Re100_sig0.0144)."""
    p = RUNS / pair if (RUNS / pair).is_dir() else Path(pair)      # the sweep's pair first: cases/se/figures/ has the same names
    assert p.is_dir() and re.match(r'Re(\d+)_sig([0-9.]+)$', p.name), f'{pair}: not a Re<Re>_sig<sig> directory'
    return p


def find(pair, arg):
    """One curve of a panel: (run directories pooled into it, particle or None)."""
    names, _, p = arg.partition('@')
    names = disk_eps(names) # eps typed as 1e-4 or as the directory spells it
    runs = []
    for name in names.split('+'):
        hits = sorted(d for pat in (name, f'dns_{pair.name}_{name}', f'do_{pair.name}_{name}') for d in pair.glob(pat)
                      if (_DNS.match(d.name) or _DO.match(d.name)) and h5(d).exists())
        assert hits, f'{name}: no run in {pair}'
        runs += hits
    return runs, (int(p) if p else None)


def runs(kind, pairs=None):
    """Every run of one kind ('dns' or 'do') that has its measures file, in the pairs given (names under RUNS
    or paths; default every pair of RUNS). Nothing under cerrado/ (the absTol runs among it) is seen."""
    pairs = [pair_dir(p) for p in pairs] if pairs else sorted(p for p in RUNS.glob('Re*_sig*') if p.is_dir())
    rx = _DNS if kind == 'dns' else _DO
    return [d for p in pairs for d in sorted(p.glob(f'{kind}_*')) if rx.match(d.name) and h5(d).exists()]


def fixed_and_varied(curves):
    """Of S, Np, eps: those every DO curve shares (name -> value) and the names of those that differ."""
    vals = {k: sorted({params(r)[k] for runs, _ in curves for r in runs if params(r)['kind'] == 'do'}) for k in ('S', 'Np', 'eps')}
    return {k: v[0] for k, v in vals.items() if len(v) == 1}, [k for k, v in vals.items() if len(v) > 1]


# ---------------------------------------------------------------- rules of every transition count
C = 0.8             # a transition: M (or A, dL, Y1) crosses -C * well from beyond +C * well, or the reverse (crossings)
M_WELL = 0.1696     # |M| of the Re 100 well: M at t = 0 of every Re 100 DNS, whose IC is the unforced fixed point (sweep_rates.py has the other Re)


def a_fp(Re):
    """A of the unforced fixed point of that Re (0.701 at Re 100): antisym_norm of the DOMeasures filter (exact
    quadrature), header attribute of series/measures.h5 of the first run of that Re that has it."""
    for f in sorted(RUNS.glob(f'Re{Re}_sig*/*/series/measures.h5')):
        with h5py.File(f, 'r') as h:
            return float(h['header'].attrs['antisym_norm'])
    raise FileNotFoundError(f'a_fp({Re}): no series/measures.h5 of the filter under {RUNS}/Re{Re}_sig*/')


def dl_fp(Re):
    """Delta L of the unforced fixed point of that Re: dL_p[0, 0] of the first forced DNS of that Re with the filter
    (its IC is ic.chk, the fixed point, and the filter's first row is step 0; a DO particle starts off it)."""
    for f in sorted(RUNS.glob(f'Re{Re}_sig*/dns_*/series/measures.h5')):
        with h5py.File(f, 'r') as h:
            return float(h['dL_p'][0, 0])
    raise FileNotFoundError(f'dl_fp({Re}): no dns_*/series/measures.h5 of the filter under {RUNS}/Re{Re}_sig*/')


# ---------------------------------------------------------------- load
def load(f, big=1e8):
    """Every dataset of a measures file as {name: value}, plus its attributes
    (format, n_modes, n_particles ... of the header).

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
    if 'header' in f:
        out.update(f['header'].attrs)

    def take(name, d):
        if isinstance(d, h5py.Dataset):
            out[name.replace('/', '_')] = d[()] if d.nbytes < big else d

    f.visititems(take)
    return out


# ---------------------------------------------------------------- series
@functools.lru_cache(maxsize=None)
def dns(run):
    """t, M, A, dL of a DNS, each (T,): particle 0 of its measures file."""
    with h5py.File(h5(run), 'r') as h:
        if 'header' not in h:
            return legacy.dns(h)
        return dict(t=h['t'][:], M=h['M_p'][:, 0], A=h['A_p'][:, 0], dL=h['dL_p'][:, 0])


@functools.lru_cache(maxsize=None)
def do_all(run):
    """t (T,) and the per-particle M, A, dL, Y1, each (T, Np)."""
    with h5py.File(h5(run), 'r') as h:
        if 'header' not in h:
            return legacy.do_all(h)
        return dict(t=h['t'][:], M=h['M_p'][:], A=h['A_p'][:], dL=h['dL_p'][:], Y1=h['Y_p'][:, :, 0])


@functools.lru_cache(maxsize=None)
def do_one(run, p):
    """t, M, A, dL, Y1 of particle p along a DO run, each (T,)."""
    with h5py.File(h5(run), 'r') as h:
        if 'header' not in h:
            return legacy.do_one(h, p)
        return dict(t=h['t'][:], M=h['M_p'][:, p], A=h['A_p'][:, p], dL=h['dL_p'][:, p], Y1=h['Y_p'][:, p, 0])


def series(run, q, p=None):
    """t (T,) and the quantity q of one run, (T, n): n = 1 for a DNS and for particle p of a DO run, n = Np for every particle."""
    kind = params(run)['kind']
    assert kind == 'do' or p is None, f'{Path(run).name}: @{p} on a DNS'
    if kind == 'dns':
        d = dns(run)
    elif p is not None:
        d = do_one(run, p)
    else:
        d = do_all(run)
    assert q in d, f'{Path(run).name}: no {q} here (has {", ".join(k for k in d if k != "t")})'
    return d['t'], d[q].reshape(len(d['t']), -1)


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
