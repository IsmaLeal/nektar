#!/usr/bin/env python3
# IC for Mowlavi & Sapsis (2018) §4.6 / Fig. 10:
#   u0 = Lamb-Oseen vortex at (pi, pi) + uniform flow (cos psi, sin psi),
#   rc  ~ N(0.2, var=1e-4),  psi ~ U(0, pi/2),  Gamma = 10.
from __future__ import annotations

import numpy as np

FIELDS = ("u", "v", "p")


def sample_parameters(rng: np.random.Generator, sample_index: int) -> dict:
    return {
        "gamma": 10.0,
        "radius": max(1.0e-6, float(rng.normal(0.2, 0.01))),
        "psi": float(rng.uniform(0.0, 0.5 * np.pi)),
        "x0": float(np.pi),
        "y0": float(np.pi),
        "Lx": float(2.0 * np.pi),
        "Ly": float(2.0 * np.pi),
    }


def evaluate_fields(coords: dict[str, np.ndarray], params: dict) -> dict[str, np.ndarray]:
    x = coords["x"]
    y = coords["y"]
    gamma = float(params["gamma"])
    rc = max(1.0e-12, float(params["radius"]))
    psi = float(params["psi"])
    x0 = float(params["x0"])
    y0 = float(params["y0"])

    dx = x - x0
    dy = y - y0
    rr2 = dx * dx + dy * dy
    rr = np.sqrt(rr2)
    rr_safe = np.where(rr == 0.0, 1.0, rr)

    theta = np.arctan2(dy, dx)
    vtheta = gamma / (2.0 * np.pi * rr_safe) * (1.0 - np.exp(-rr2 / (rc * rc)))
    vtheta = np.where(rr == 0.0, 0.0, vtheta)

    u = -vtheta * np.sin(theta) + np.cos(psi)
    v = vtheta * np.cos(theta) + np.sin(psi)
    p = np.zeros_like(u)

    return {"u": u, "v": v, "p": p}
