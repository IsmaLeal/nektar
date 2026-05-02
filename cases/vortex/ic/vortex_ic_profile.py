#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

FIELDS = ("u", "v", "p")


def mollifier_discrete(ny: int, nx: int, passes: int = 4) -> np.ndarray:
    ny1, nx1 = ny + 1, nx + 1
    mol = np.ones((ny1, nx1), dtype=float)
    mol[0, :] = 0.0
    mol[-1, :] = 0.0
    mol[:, 0] = 0.0
    mol[:, -1] = 0.0
    for _ in range(max(0, passes)):
        old = mol.copy()
        mol[1:-1, 1:-1] = old[1:-1, 1:-1] * (
            old[0:-2, 1:-1]
            + old[0:-2, 0:-2]
            + old[0:-2, 2:]
            + old[1:-1, 0:-2]
            + old[1:-1, 2:]
            + old[2:, 0:-2]
            + old[2:, 1:-1]
            + old[2:, 2:]
        ) / 8.0
    return mol[:-1, :-1]


def sample_bilinear_clamped(F, xs, ys, Lx, Ly):
    ny, nx = F.shape
    xs = np.where(np.isclose(xs, Lx), 0.0, xs)
    ys = np.where(np.isclose(ys, Ly), 0.0, ys)
    xs = np.clip(xs, 0.0, np.nextafter(Lx, -np.inf))
    ys = np.clip(ys, 0.0, np.nextafter(Ly, -np.inf))

    fx = xs * (nx / Lx)
    fy = ys * (ny / Ly)
    i0 = np.floor(fx).astype(int)
    j0 = np.floor(fy).astype(int)
    i1 = np.minimum(i0 + 1, nx - 1)
    j1 = np.minimum(j0 + 1, ny - 1)
    tx = fx - i0
    ty = fy - j0

    f00 = F[j0, i0]
    f10 = F[j0, i1]
    f01 = F[j1, i0]
    f11 = F[j1, i1]
    return (1 - tx) * (1 - ty) * f00 + tx * (1 - ty) * f10 + (1 - tx) * ty * f01 + tx * ty * f11


def sample_parameters(rng: np.random.Generator, sample_index: int) -> dict:
    return {
        "gamma": 10.0,
        "radius": max(0.01, float(rng.normal(0.2, 0.1))),
        "x0": float(np.pi),
        "y0": float(np.pi),
        "Lx": float(2.0 * np.pi),
        "Ly": float(2.0 * np.pi),
        "moll_nx": 400,
        "moll_ny": 400,
        "moll_passes": 4,
    }


def evaluate_fields(coords: dict[str, np.ndarray], params: dict) -> dict[str, np.ndarray]:
    x = coords["x"]
    y = coords["y"]
    Lx = float(params["Lx"])
    Ly = float(params["Ly"])
    gamma = float(params["gamma"])
    rc = max(1.0e-12, float(params["radius"]))
    x0 = float(params["x0"])
    y0 = float(params["y0"])

    mol = mollifier_discrete(
        int(params["moll_ny"]),
        int(params["moll_nx"]),
        int(params["moll_passes"]),
    )

    dx = x - x0
    dy = y - y0
    rr2 = dx * dx + dy * dy
    rr = np.sqrt(rr2)
    rr_safe = np.where(rr == 0.0, 1.0, rr)

    theta = np.arctan2(dy, dx)
    vtheta = gamma / (2.0 * np.pi * rr_safe) * (1.0 - np.exp(-rr2 / (rc * rc)))
    vtheta = np.where(rr == 0.0, 0.0, vtheta)

    m = sample_bilinear_clamped(mol, x, y, Lx, Ly)
    u = -vtheta * np.sin(theta) * m
    v = vtheta * np.cos(theta) * m
    p = np.zeros_like(u)

    return {"u": u, "v": v, "p": p}
