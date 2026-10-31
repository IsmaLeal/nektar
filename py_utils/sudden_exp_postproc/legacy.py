"""What the DOMeasures filter made obsolete and a few things still use.

The plane inner product, <a, b> = sum over the masked 901 x 121 plane of (a_u b_u + a_v b_v) dx dy (W, ip),
mirror and antisym (y -> -y), make_phi and the make-phi command (the phi of the plane, phi_Re<Re>.npz, drawn
by plot_plane.py and do_fields.py):

    python3 legacy.py make-phi DAT OUT.npz          (make_phi.sh, on JED)

The series of a run extracted before the filter (campos/grid.h5 of a DNS, campos/measures.h5 of a DO run),
computed as the python extraction of 2026-09 did:

    M   = sgn v(2, 0) sqrt(int v^2 dx) on the centreline (Ducimetiere Eq. 25), stored for a DNS (M_grid),
          rebuilt for a DO particle from v_axis_mean + Y . v_axis_modes
    A   = <u, phi>, stored for a DNS, A0 + Y . Ai for a DO particle
    dL  = L_upper - L_lower, L = x of the last column with u < 0 on the row |y| = 1.475 (u_wall rows)
    Y_1 = yi[:, :, 0]

FieldConvert on the archives themselves, on this laptop (FIELDCONVERT, fc_env, interp, load_dat, yi_from_fld,
yi_series, casefile_param): the validation checks of July 2026 (rankinv_check.py, convergence_orders.py,
forcing_gain_check.py, vortex_mc_check.py) use them; the planes of the campaign come from JED (extract.sbatch).

se.dns, se.do_all and se.do_one call dns, do_all, do_one when the measures file has no header group; those
three take the open h5py.File. qdag_projection.py (q^dagger on the DO modes) and mean_check.py use ip.
"""
import glob
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'cases/se/templates'))
import extract_lib as xl  # noqa: E402

XA = xl.X[0][xl.X[0] >= 0]                     # the 801 centreline points, x >= 0
XW = xl.X[0][xl.X[0] > 0]                      # the 800 near-wall points, x > 0
I2 = int(np.argmin(np.abs(XA - 2.0)))          # the sensor x = 2
TRAP = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
W = xl.INSIDE * xl.DX * xl.DY                  # area of each cell, 0 on the inlet walls


# ---------------------------------------------------------------- the plane inner product and phi
def ip(au, av, bu, bv):
    return float(((au * bu + av * bv) * W).sum())


def mirror(u, v):
    """R: y -> -y. The grid is symmetric in y, so this is a row flip; v changes sign."""
    return u[::-1, :], -v[::-1, :]


def antisym(u, v):
    mu, mv = mirror(u, v)
    return (u - mu) / 2, (v - mv) / 2


def make_phi(u, v):
    au, av = antisym(u, v)
    n = np.sqrt(ip(au, av, au, av))
    return au / n, av / n


# ---------------------------------------------------------------- the series of the runs extracted before the filter


def m_grid(v_axis):
    return float(np.sign(v_axis[I2]) * np.sqrt(TRAP(v_axis ** 2, XA)))


def reattachment(u_wall_row):
    """x of the last column with u < 0; no backflow -> x = 40."""
    neg = u_wall_row < 0
    return XW[XW.size - 1 - int(np.argmax(neg[::-1]))]


def dns(h):
    return dict(t=h['t'][:], M=h['M_grid'][:], A=h['A'][:], dL=h['dL'][:])


def do_all(h, block=50):
    """t (T,) and M, A, dL, Y1 of every particle, (T, Np). int v_p^2 dx is quadratic in Y_p (c + 2 b.Y + Y^T G Y,
    trapezoid weights), so the (T, Np, 801) lines are never formed; dL goes in blocks of instants."""
    t, Y = h['t'][:], h['yi'][:]
    vm, phi = h['v_axis_mean'][:], h['v_axis_modes'][:]
    T, Np, S = Y.shape
    w = np.full(801, XA[1] - XA[0]); w[[0, -1]] *= 0.5
    c = np.einsum('ti,i,ti->t', vm, w, vm)
    b = np.einsum('ti,i,tsi->ts', vm, w, phi)
    G = np.einsum('tsi,i,tri->tsr', phi, w, phi)
    q = c[:, None] + 2 * np.einsum('tps,ts->tp', Y, b) + np.einsum('tps,tsr,tpr->tp', Y, G, Y)
    v2 = vm[:, I2][:, None] + np.einsum('tps,ts->tp', Y, phi[:, :, I2])
    M = (np.sign(v2) * np.sqrt(np.maximum(q, 0.0))).astype(np.float32)
    A = (h['A0'][:][:, None] + np.einsum('tps,ts->tp', Y, h['Ai'][:])).astype(np.float32)
    um, modes = h['u_wall_mean'], h['u_wall_modes']                     # (T, 2, 800) and (T, S, 2, 800), on disk
    dL = np.empty((T, Np), dtype=np.float32)
    for a in range(0, T, block):
        bl = slice(a, min(a + block, T))
        n = bl.stop - bl.start
        uw = um[bl].reshape(n, 1, 1600) + Y[bl] @ modes[bl].reshape(n, S, 1600)   # (n, Np, 1600): lower wall, upper wall
        neg = (uw < 0).reshape(n, Np, 2, 800)
        L = XW[799 - np.argmax(neg[..., ::-1], axis=-1)]                   # last column with u < 0; none -> x = 40
        dL[bl] = L[..., 1] - L[..., 0]
    return dict(t=t, M=M, A=A, dL=dL, Y1=Y[:, :, 0])


def do_one(h, p):
    Y = h['yi'][:, p, :]                                                   # (T, S), never the full yi
    A = h['A0'][:] + np.einsum('tk,tk->t', Y, h['Ai'][:])
    vax = h['v_axis_mean'][:] + np.einsum('tk,tkx->tx', Y, h['v_axis_modes'][:])
    uw = h['u_wall_mean'][:] + np.einsum('tk,tkcx->tcx', Y, h['u_wall_modes'][:])
    return dict(t=h['t'][:], A=A, M=np.array([m_grid(r) for r in vax]),
                dL=np.array([reattachment(r[1]) - reattachment(r[0]) for r in uw]), Y1=Y[:, 0])



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

_YI_TAG = re.compile(r"<DOVelocityCorrectionScheme_Yi_hex>([0-9a-fA-F]+)<")


def yi_from_fld(path, S):
    """(Np, S) Yi from the hex metadata of one DO archive (Info.xml of an MPI run, the .fld of a serial one)."""
    p = Path(path)
    info = p / "Info.xml" if p.is_dir() else p
    m = _YI_TAG.search(info.read_text())
    if m is None:
        raise ValueError(f"no Yi_hex metadata in {info}")
    flat = np.frombuffer(bytes.fromhex(m.group(1)), dtype="<f8")
    if flat.size % S:
        raise ValueError(f"{info}: {flat.size} doubles not divisible by S={S}")
    return flat.reshape(-1, S)


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "make-phi":
        sys.exit("usage: legacy.py make-phi DAT OUT.npz")
    f = xl.read_dat(sys.argv[2])
    pu, pv = make_phi(f["u"], f["v"])
    au, av = antisym(f["u"], f["v"])
    ru, rv = mirror(pu, pv)
    su, sv = pu + ru, pv + rv                      # phi + R phi, zero for an antisymmetric phi
    na = np.sqrt(ip(au, av, au, av))
    nt = np.sqrt(ip(f["u"], f["v"], f["u"], f["v"]))
    np.savez(sys.argv[3], phi_u=pu, phi_v=pv, source=sys.argv[2], antisym_norm=na, total_norm=nt)
    print(f"phi written to {sys.argv[3]}: |antisym(u_fp)| = {na:.4f}, |u_fp| = {nt:.4f}, "
          f"antisymmetry residual of phi = {np.sqrt(ip(su, sv, su, sv)):.2e}")