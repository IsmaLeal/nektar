#!/usr/bin/env python3
"""
Is the DO mean field the long-time average of the DNS? For every DO run of the pair:

    d = ||DO mean - DNS mean|| / ||DNS mean||

DNS mean: the u, v planes of grid.h5 averaged over every stored instant, the seeds pooled (twins left out).
DO mean: grid/mean_u, grid/mean_v of measures.h5 at the run's last stored instant.
|| ||: the plane inner product of extract_lib.py (se.ip).

    python3 mean_check.py [--pair Re100_sig0.0144]
"""
import argparse

import h5py
import numpy as np

import se


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", default="Re100_sig0.0144")
    args = ap.parse_args()

    dns = [r for r in se.runs("dns", [args.pair]) if not se.params(r)["twin"]]
    U, V, n = np.zeros((se.NY, se.NX)), np.zeros((se.NY, se.NX)), 0
    for r in dns:
        with h5py.File(se.h5(r), "r") as f:
            for i in range(0, len(f["t"]), 500):
                U += f["u"][i:i + 500].sum(axis=0, dtype=np.float64)
                V += f["v"][i:i + 500].sum(axis=0, dtype=np.float64)
            n += len(f["t"])
    U, V = U / n, V / n
    norm = np.sqrt(se.ip(U, V, U, V))
    print(f"DNS mean: {len(dns)} seeds ({', '.join(r.name.split('_seed')[1] for r in dns)}), {n} planes")

    for r in se.runs("do", [args.pair]):
        with h5py.File(se.h5(r), "r") as f:
            t, du, dv = f["grid/t"][-1], f["grid/mean_u"][-1] - U, f["grid/mean_v"][-1] - V
        print(f"{r.name:58s} t {t:6g}   d = {np.sqrt(se.ip(du, dv, du, dv)) / norm:.3e}")


if __name__ == "__main__":
    main()
