#!/usr/bin/env python3
"""Create a starter IC profile file for py_utils/ic_sampler/generate_ic_samples.py."""

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

TEMPLATE = '''#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

# Fields to generate in each sample. Must match the case variables order/names.
FIELDS = ({fields})


def sample_parameters(
    rng: np.random.Generator,
    sample_index: int,
    coords_meta: dict | None = None,
) -> dict:
    """Return random/sample-dependent parameters for one sample.

    coords_meta (optional) contains:
      - dim: int
      - npts: int
      - bbox: dict like {{"x": [min, max], "y": [min, max], ...}}
    """
    dim = int(coords_meta.get("dim", 2)) if coords_meta else 2
    n_terms = 8
    length_frac = float(np.clip(rng.normal(0.15, 0.05), 0.03, 0.6))
    return {{
        "seed": int(rng.integers(0, 2**31 - 1)),
        "dim": dim,
        "n_terms": n_terms,
        "length_frac": length_frac,
        # Per-field RMS amplitude scales (edit for your case physics):
        "field_rms": {{name: float(rng.normal(0.05, 0.02)) for name in FIELDS}},
    }}


def evaluate_fields(coords: dict[str, np.ndarray], params: dict) -> dict[str, np.ndarray]:
    """Evaluate fields on mesh points (geometry-agnostic random smooth fields).

    coords has keys x/y/(z) and optionally dim.
    Return dict: field_name -> 1D array (npts,)
    """
    dim = int(np.asarray(coords.get("dim", np.asarray([2], dtype=int)))[0])
    dim = max(1, min(3, dim))
    keys = ["x", "y", "z"][:dim]
    pts = np.stack([np.asarray(coords[k], dtype=float) for k in keys], axis=1)
    npts = pts.shape[0]

    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    span = np.maximum(hi - lo, 1e-12)
    diag = float(np.linalg.norm(span))

    rng = np.random.default_rng(int(params["seed"]))
    n_terms = int(params.get("n_terms", 8))
    ell = max(1e-12, float(params.get("length_frac", 0.15)) * diag)
    centers = lo + rng.random((n_terms, dim)) * span

    out: dict[str, np.ndarray] = {{}}
    for name in FIELDS:
        w = rng.normal(0.0, 1.0, size=n_terms)
        acc = np.zeros(npts, dtype=float)
        for j in range(n_terms):
            d2 = np.sum((pts - centers[j]) ** 2, axis=1)
            acc += w[j] * np.exp(-0.5 * d2 / (ell * ell))

        acc -= float(acc.mean())
        rms = float(np.sqrt(np.mean(acc * acc)))
        if rms > 1e-14:
            acc /= rms

        target = float(params.get("field_rms", {{}}).get(name, 0.0))
        out[name] = target * acc

    return out
'''


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fields", default=None, help="Comma-separated field names")
    ap.add_argument("--case-xml", type=Path, default=None, help="Infer fields from case XML <VARIABLES>")
    args = ap.parse_args()

    fields: list[str] = []
    if args.fields:
        fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    elif args.case_xml is not None:
        root = ET.parse(args.case_xml).getroot()
        for v in root.findall(".//VARIABLES/V"):
            if v.text:
                name = v.text.strip()
                if name:
                    fields.append(name)
    else:
        fields = ["u", "v", "p"]

    if not fields:
        raise ValueError(
            "Could not determine fields. Pass --fields or --case-xml with a valid <VARIABLES> block."
        )

    fields_literal = ", ".join(f'"{f}"' for f in fields)
    text = TEMPLATE.format(fields=fields_literal)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"Wrote profile template: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
