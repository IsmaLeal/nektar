#!/usr/bin/env python3
from __future__ import annotations

import sys
import argparse
import numpy as np


def read_mesh_pts_xml(path):
    xs, ys = [], []
    in_pts = False
    with open(path, "r") as f:
        for line in f:
            if "<POINTS" in line:
                in_pts = True
                continue
            if "</POINTS>" in line:
                break
            if not in_pts:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            xs.append(float(parts[0]))
            ys.append(float(parts[1]))
    if not xs:
        raise RuntimeError("No points found in mesh .pts XML")
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def mollifier_discrete(Ny: int, Nx: int, passes: int = 4) -> np.ndarray:
    """
    Edge-taper mollifier that matches the reference discrete construction:
    build (Ny+1, Nx+1), smooth 4 passes, then drop last row/col.
    """
    Ny1, Nx1 = Ny + 1, Nx + 1
    mol = np.ones((Ny1, Nx1), dtype=float)
    mol[0, :] = 0.0
    mol[-1, :] = 0.0
    mol[:, 0] = 0.0
    mol[:, -1] = 0.0
    for _ in range(max(0, passes)):
        mol_old = mol.copy()
        mol[1:-1, 1:-1] = mol_old[1:-1, 1:-1] * (
            mol_old[0:-2, 1:-1] + mol_old[0:-2, 0:-2] + mol_old[0:-2, 2:] +
            mol_old[1:-1, 0:-2] + mol_old[1:-1, 2:] +
            mol_old[2:, 0:-2] + mol_old[2:, 1:-1] + mol_old[2:, 2:]
        ) / 8.0
    return mol[:-1, :-1]


def sample_bilinear_clamped(F, xs, ys, Lx, Ly):
    """
    Bilinear sampling of a uniform-grid scalar field F[j,i]
    defined on x_i = i*Lx/Nx, y_j = j*Ly/Ny (endpoint=False),
    with clamped (non-periodic) bounds.
    """
    Ny, Nx = F.shape
    xs = np.where(np.isclose(xs, Lx), 0.0, xs)
    ys = np.where(np.isclose(ys, Ly), 0.0, ys)
    xs = np.clip(xs, 0.0, np.nextafter(Lx, -np.inf))
    ys = np.clip(ys, 0.0, np.nextafter(Ly, -np.inf))

    fx = xs * (Nx / Lx)
    fy = ys * (Ny / Ly)

    i0 = np.floor(fx).astype(int)
    j0 = np.floor(fy).astype(int)
    i1 = np.minimum(i0 + 1, Nx - 1)
    j1 = np.minimum(j0 + 1, Ny - 1)

    tx = fx - i0
    ty = fy - j0

    f00 = F[j0, i0]
    f10 = F[j0, i1]
    f01 = F[j1, i0]
    f11 = F[j1, i1]

    return (1 - tx) * (1 - ty) * f00 + tx * (1 - ty) * f10 + (1 - tx) * ty * f01 + tx * ty * f11


def build_lamb_oseen_velocity(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    Lx: float,
    Ly: float,
    gamma: float,
    radius: float,
    x0: float,
    y0: float,
    moll_nx: int,
    moll_ny: int,
    moll_passes: int,
) -> tuple[np.ndarray, np.ndarray]:
    mol = mollifier_discrete(moll_ny, moll_nx, passes=moll_passes)
    dx = xs - x0
    dy = ys - y0
    rr2 = dx * dx + dy * dy
    rr = np.sqrt(rr2)
    rr_safe = np.where(rr == 0.0, 1.0, rr)

    theta = np.arctan2(dy, dx)
    rc = max(radius, 1.0e-12)
    vtheta = gamma / (2.0 * np.pi * rr_safe) * (1.0 - np.exp(-rr2 / (rc * rc)))
    vtheta = np.where(rr == 0.0, 0.0, vtheta)

    m = sample_bilinear_clamped(mol, xs, ys, Lx, Ly)
    u = -vtheta * np.sin(theta) * m
    v =  vtheta * np.cos(theta) * m
    return u, v


def write_pts(path: str, xs: np.ndarray, ys: np.ndarray, u: np.ndarray, v: np.ndarray) -> None:
    with open(path, "w") as f:
        f.write('<?xml version="1.0" encoding="utf-8" ?>\n')
        f.write("<NEKTAR>\n")
        f.write('  <POINTS DIM="2" FIELDS="u,v,p">\n')
        for x, y, ui, vi in zip(xs, ys, u, v):
            f.write(f"    {x:.16e} {y:.16e} {ui:.16e} {vi:.16e} 0.0\n")
        f.write("  </POINTS>\n")
        f.write("</NEKTAR>\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mesh_pts", help="Input mesh points XML (.pts)")
    ap.add_argument("out_pts", help="Output IC points XML (.pts)")
    ap.add_argument("--Lx", type=float, default=2.0 * np.pi)
    ap.add_argument("--Ly", type=float, default=2.0 * np.pi)
    ap.add_argument("--gamma", type=float, default=10.0)
    ap.add_argument("--radius", type=float, default=0.2)
    ap.add_argument("--x0", type=float, default=None, help="Vortex center x (default Lx/2)")
    ap.add_argument("--y0", type=float, default=None, help="Vortex center y (default Ly/2)")
    ap.add_argument("--mollifier-nx", type=int, default=400)
    ap.add_argument("--mollifier-ny", type=int, default=400)
    ap.add_argument("--mollifier-passes", type=int, default=4)
    args = ap.parse_args()

    xs, ys = read_mesh_pts_xml(args.mesh_pts)
    x0 = 0.5 * args.Lx if args.x0 is None else args.x0
    y0 = 0.5 * args.Ly if args.y0 is None else args.y0
    u, v = build_lamb_oseen_velocity(
        xs,
        ys,
        Lx=args.Lx,
        Ly=args.Ly,
        gamma=args.gamma,
        radius=args.radius,
        x0=x0,
        y0=y0,
        moll_nx=args.mollifier_nx,
        moll_ny=args.mollifier_ny,
        moll_passes=args.mollifier_passes,
    )

    print(np.mean(u), np.mean(v))
    write_pts(args.out_pts, xs, ys, u, v)


if __name__ == "__main__":
    main()
