#!/usr/bin/env python3
"""Compose per-particle initial conditions u_p = ubar + sum_i Y_pi u_i from a
DO archive, as Nektar chk files -- the matched-ensemble MC rig of report sec 35.

Machinery gate M0 (registered, runs first, aborts on failure):
  (i)  Y=0 composition must reproduce the archived mean bit-exactly through
       the writer roundtrip;
  (ii) re-decoding each composed IC and subtracting the mean must recover
       sum_i Y_pi u_i to <= 1e-14 relative.

Usage:
    python3 do_compose_mc_ics.py --archive RUN/output/do/casefile.do_000000.fld \
        --template-chk RUN/output/chks/casefile_0.chk --outdir DIR [--nmodes 6]
"""
from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "sudden_exp_postproc"))

from load_chk import (_build_chk_mode_writer, _coeffs_per_element,
                      _decode_chk_elements_blob, _parse_elem_id_count,
                      _parse_num_modes_per_dir)
from do_quicklook import yi_from_fld


def archive_coeffs(archive: Path) -> dict[str, np.ndarray]:
    """Field name -> concatenated coefficient vector, from a serial DO fld."""
    root = ET.parse(archive).getroot()
    out: dict[str, list[np.ndarray]] = {}
    for el in root.iter("ELEMENTS"):
        fields = [f.strip() for f in el.attrib["FIELDS"].split(",") if f.strip()]
        ne = _parse_elem_id_count(el.attrib["ID"])
        cpe = _coeffs_per_element(el.attrib["SHAPE"],
                                  _parse_num_modes_per_dir(el.attrib["NUMMODESPERDIR"]))
        arr = _decode_chk_elements_blob(el).reshape(len(fields), ne * cpe)
        for j, f in enumerate(fields):
            out.setdefault(f, []).append(arr[j])
    return {f: np.concatenate(parts) for f, parts in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--template-chk", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--nmodes", type=int, default=6)
    args = ap.parse_args()
    S = args.nmodes

    coeffs = archive_coeffs(args.archive)
    Y = yi_from_fld(args.archive, S)
    Np = Y.shape[0]
    mean = {v: coeffs[v] for v in ("u", "v", "p")}
    modes = {v: np.column_stack([coeffs[f"mode_{i}_{v}"] for i in range(S)])
             for v in ("u", "v", "p")}
    write, totals = _build_chk_mode_writer(args.template_chk)
    for v in ("u", "v", "p"):
        if totals.get(v) != mean[v].size:
            raise SystemExit(f"M0 FAIL: template field {v} size {totals.get(v)} "
                             f"!= archive {mean[v].size}")
    args.outdir.mkdir(parents=True, exist_ok=True)

    # gate M0(i): zero-coefficient composition == archived mean, bit-exact
    probe = args.outdir / "_m0_mean_roundtrip.chk"
    write(probe, {v: mean[v] for v in ("u", "v", "p")})
    back = archive_coeffs(probe)
    for v in ("u", "v", "p"):
        if not np.array_equal(back[v], mean[v]):
            raise SystemExit(f"M0(i) FAIL: mean roundtrip not bit-exact in {v}")
    print("M0(i) PASS: mean roundtrip bit-exact")

    worst = 0.0
    for pidx in range(Np):
        cmap = {v: mean[v] + modes[v] @ Y[pidx] for v in ("u", "v", "p")}
        out = args.outdir / f"ic_p{pidx:03d}.chk"
        write(out, cmap)
        back = archive_coeffs(out)
        for v in ("u", "v"):
            fluct = modes[v] @ Y[pidx]
            err = np.linalg.norm(back[v] - mean[v] - fluct)
            rel = err / max(np.linalg.norm(fluct), 1e-300)
            worst = max(worst, rel)
    print(f"M0(ii) worst relative recovery error over {Np} ICs: {worst:.2e}")
    if worst > 1e-14:
        raise SystemExit("M0(ii) FAIL (> 1e-14)")
    print(f"M0 PASS -- {Np} ICs written to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
