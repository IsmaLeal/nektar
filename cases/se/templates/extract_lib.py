#!/usr/bin/env python3
"""Reductions for the JED extraction of the sudden-expansion campaign.

Every DNS checkpoint and every DO archive is interpolated by FieldConvert on
the same 901 x 121 plane (x in [-5, 40], y in [-1.5, 1.5]; the inlet walls
|y| > 0.5 for x < 0 are masked). From that plane this module keeps only what
the asymmetry measures need:

  DNS checkpoint  -> u, v on the plane (float32), the centreline v, the
                     near-wall u of both walls, the wall pressure of both
                     walls, and the scalars A, dL, M_grid.
  DO archive      -> Yi (from Info.xml), a_i = <ubar, u_i>, the Gram matrix
                     G_ij = <u_i, u_j>, A0 = <ubar, phi>, A_i = <u_i, phi>,
                     the same three lines for the mean and for every mode,
                     and optionally the full plane of mean and modes.

Inner product: <a, b> = sum over the plane of (a_u b_u + a_v b_v) dx dy with
the mask, the same convention as Desktop/asym_measures.py (2026-09-01).

phi is the fixed reference direction of the measure
    A(t) = <u(t), phi>,  phi = antisym(u_fp) / ||antisym(u_fp)||,
with u_fp the unforced asymmetric fixed point at that Re and
antisym(u) = (u - R u)/2, R the mirror y -> -y (u even, v odd).

Sub-commands (see main):
    reduce-dns  DAT OUT.npz [--phi PHI.npz]
    reduce-do   DAT OUT.npz --info INFO [--phi PHI.npz] [--keep-grid]
    make-phi    DAT OUT.npz
Only numpy is required.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- plane grid
NX, NY = 901, 121
X0, X1, Y0, Y1 = -5.0, 40.0, -1.5, 1.5
x = np.linspace(X0, X1, NX)
y = np.linspace(Y0, Y1, NY)
DX = x[1] - x[0]
DY = y[1] - y[0]
X, Y = np.meshgrid(x, y)                       # (NY, NX), x fastest
INSIDE = ~((X < 0) & (np.abs(Y) > 0.5))        # inlet channel walls masked
W = INSIDE * DX * DY

JAXIS = int(np.argmin(np.abs(y)))              # centreline row (y = 0)
JLOW, JUP = 1, NY - 2                          # first row off each wall (|y| = 1.475)
JWLOW, JWUP = 0, NY - 1                        # wall rows (|y| = 1.5)
XPOS = x >= 0                                  # 801 columns, centreline and wall pressure
XWALL = x > 0                                  # 800 columns, reattachment search
I2 = int(np.argmin(np.abs(x[XPOS] - 2.0)))     # sensor column x = 2 within XPOS

BOX = f"box={NX},{NY},1,{X0:g},{X1:g},{Y0:g},{Y1:g},0,0"   # FieldConvert interppoints argument


# ---------------------------------------------------------------- readers
def read_dat(path):
    """FieldConvert interppoints output -> {name: (NY, NX) float64}.

    Header: 'Variables = x, y, z, u, v, p, ...' then a Zone line, then
    NX*NY rows, x fastest. Points outside the domain come back as 0.
    """
    with open(path) as f:
        head = f.readline()
    names = [s.strip() for s in head.split("=", 1)[1].split(",")]
    data = np.loadtxt(path, skiprows=3)
    if data.shape[0] != NX * NY:
        raise ValueError(f"{path}: {data.shape[0]} rows, expected {NX * NY}")
    out = {}
    for j, n in enumerate(names):
        if n in ("x", "y", "z"):
            continue
        out[n] = data[:, j].reshape(NY, NX)
    # sanity: the grid FieldConvert produced is the one assumed above
    if abs(data[0, 0] - X0) > 1e-9 or abs(data[NX, 1] - (Y0 + DY)) > 1e-9:
        raise ValueError(f"{path}: grid ordering is not x-fastest on the assumed box")
    return out


_YI_TAG = re.compile(r"<DOVelocityCorrectionScheme_Yi_hex>([0-9a-fA-F]+)<")


def yi_from_info(path, S):
    """(Np, S) Yi matrix from the hex metadata of a Field-format archive.

    path: the archive (directory with Info.xml for MPI runs, or the single
    .fld file of a serial run). Identical in every partition file.
    """
    p = Path(path)
    info = p / "Info.xml" if p.is_dir() else p
    m = _YI_TAG.search(info.read_text())
    if m is None:
        raise ValueError(f"no Yi_hex metadata in {info}")
    flat = np.frombuffer(bytes.fromhex(m.group(1)), dtype="<f8")
    if flat.size % S:
        raise ValueError(f"{info}: {flat.size} doubles not divisible by S={S}")
    return flat.reshape(-1, S)


def modes_in(fields):
    """Sorted list of mode indices present in a DO archive plane."""
    ks = sorted({int(m.group(1)) for n in fields for m in [re.match(r"mode_(\d+)_u$", n)] if m})
    return ks


# ---------------------------------------------------------------- algebra
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


def load_phi(path):
    z = np.load(path)
    return z["phi_u"], z["phi_v"]


# ---------------------------------------------------------------- measures on one field
def lines(u, v, p):
    """The three line samples of one (u, v, p) field."""
    return dict(
        v_axis=v[JAXIS, XPOS],                       # (801,) centreline v, x >= 0
        u_wall=np.stack([u[JLOW, XWALL], u[JUP, XWALL]]),    # (2, 800) near-wall u, x > 0
        p_wall=np.stack([p[JWLOW, XPOS], p[JWUP, XPOS]]),    # (2, 801) wall pressure, x >= 0
    )


def reattachment(u_wall_row):
    """Downstream end of the near-wall backflow: x of the last column with
    u < 0 (Desktop/asym_measures.py, 2026-09-01). No backflow -> x = 40."""
    xw = x[XWALL]
    neg = u_wall_row < 0
    return xw[xw.size - 1 - int(np.argmax(neg[::-1]))]


def m_grid(v_axis):
    """Ducimetiere Eq. 25 on the plane centreline: sgn v(x=2) sqrt(int v^2 dx)."""
    xa = x[XPOS]
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(np.sign(v_axis[I2]) * np.sqrt(trap(v_axis ** 2, xa)))


def reduce_dns(fields, phi=None):
    u, v, p = fields["u"], fields["v"], fields["p"]
    out = dict(u=u.astype(np.float32), v=v.astype(np.float32))
    out.update(lines(u, v, p))
    out["dL"] = reattachment(out["u_wall"][1]) - reattachment(out["u_wall"][0])   # upper - lower
    out["M_grid"] = m_grid(out["v_axis"])
    out["norm2"] = ip(u, v, u, v)
    if phi is not None:
        out["A"] = ip(u, v, phi[0], phi[1])
    return out


def reduce_do(fields, phi=None, keep_grid=False):
    ks = modes_in(fields)
    S = len(ks)
    if S == 0:
        raise ValueError("no mode_i_u fields in this plane: not a DO archive")
    ub, vb, pb = fields["u"], fields["v"], fields["p"]
    mu = [fields[f"mode_{k}_u"] for k in ks]
    mv = [fields[f"mode_{k}_v"] for k in ks]
    mp = [fields.get(f"mode_{k}_p") for k in ks]
    out = dict(S=S)
    out["a"] = np.array([ip(ub, vb, mu[i], mv[i]) for i in range(S)])
    out["G"] = np.array([[ip(mu[i], mv[i], mu[j], mv[j]) for j in range(S)] for i in range(S)])
    out["mean_norm2"] = ip(ub, vb, ub, vb)
    if phi is not None:
        out["A0"] = ip(ub, vb, phi[0], phi[1])
        out["Ai"] = np.array([ip(mu[i], mv[i], phi[0], phi[1]) for i in range(S)])
    L = lines(ub, vb, pb)
    out["v_axis_mean"], out["u_wall_mean"], out["p_wall_mean"] = L["v_axis"], L["u_wall"], L["p_wall"]
    ML = [lines(mu[i], mv[i], mp[i] if mp[i] is not None else np.zeros_like(mu[i])) for i in range(S)]
    out["v_axis_modes"] = np.stack([m["v_axis"] for m in ML])     # (S, 801)
    out["u_wall_modes"] = np.stack([m["u_wall"] for m in ML])     # (S, 2, 800)
    out["p_wall_modes"] = np.stack([m["p_wall"] for m in ML])     # (S, 2, 801)
    if keep_grid:
        out["mean_u"], out["mean_v"] = ub.astype(np.float32), vb.astype(np.float32)
        out["mode_u"] = np.stack(mu).astype(np.float32)           # (S, NY, NX)
        out["mode_v"] = np.stack(mv).astype(np.float32)
    return out


# ---------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("reduce-dns"); a.add_argument("dat"); a.add_argument("out"); a.add_argument("--phi")
    b = sub.add_parser("reduce-do"); b.add_argument("dat"); b.add_argument("out"); b.add_argument("--info", required=True)
    b.add_argument("--phi"); b.add_argument("--keep-grid", action="store_true")
    c = sub.add_parser("make-phi"); c.add_argument("dat"); c.add_argument("out")
    args = ap.parse_args(argv)

    if args.cmd == "make-phi":
        f = read_dat(args.dat)
        pu, pv = make_phi(f["u"], f["v"])
        au, av = antisym(f["u"], f["v"])
        ru, rv = mirror(pu, pv)
        su, sv = pu + ru, pv + rv                      # phi + R phi, zero for an antisymmetric phi
        na = np.sqrt(ip(au, av, au, av))
        nt = np.sqrt(ip(f["u"], f["v"], f["u"], f["v"]))
        np.savez(args.out, phi_u=pu, phi_v=pv, source=str(args.dat), antisym_norm=na, total_norm=nt)
        print(f"phi written to {args.out}: |antisym(u_fp)| = {na:.4f}, |u_fp| = {nt:.4f}, "
              f"antisymmetry residual of phi = {np.sqrt(ip(su, sv, su, sv)):.2e}")
        return 0

    phi = load_phi(args.phi) if args.phi else None
    fields = read_dat(args.dat)
    if args.cmd == "reduce-dns":
        out = reduce_dns(fields, phi)
    else:
        out = reduce_do(fields, phi, keep_grid=args.keep_grid)
        out["yi"] = yi_from_info(args.info, out["S"])
    np.savez(args.out, **out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
