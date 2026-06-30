#!/usr/bin/env python3
"""Generate DO IC sample files from a user-defined Python profile.

Profile API (in --profile file):
- FIELDS: tuple/list, e.g. ("u", "v", "p")
- sample_parameters(rng, sample_index[, coords_meta]) -> dict
- evaluate_fields(coords, params) -> dict[field_name -> np.ndarray(npts,)]
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from py_utils.ic_sampler.core import (  # type: ignore
        build_manifest,
        convert_pts_to_fld,
        read_mesh_pts_xml,
        resolve_fieldconvert,
        write_pts_fields,
    )
else:
    from .core import (
        build_manifest,
        convert_pts_to_fld,
        read_mesh_pts_xml,
        resolve_fieldconvert,
        write_pts_fields,
    )


def _load_profile(path: Path):
    spec = importlib.util.spec_from_file_location("ic_profile", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import profile from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", type=Path, required=True, help="Path to profile .py")
    ap.add_argument("--mesh-pts", type=Path, required=True, help="Mesh points XML (.pts)")
    ap.add_argument("--case-xml", type=Path, required=True, help="Case XML used by FieldConvert")
    ap.add_argument("--n-samples", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--samples-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--fieldconvert", type=Path, default=None)
    ap.add_argument("--keep-pts", action="store_true")
    ap.add_argument("--params-jsonl", type=Path, default=None)
    args = ap.parse_args()

    if args.n_samples <= 0:
        raise ValueError("--n-samples must be > 0")

    profile = _load_profile(args.profile)
    if not hasattr(profile, "FIELDS"):
        raise AttributeError("Profile must define FIELDS")
    if not hasattr(profile, "sample_parameters"):
        raise AttributeError("Profile must define sample_parameters(rng, sample_index)")
    if not hasattr(profile, "evaluate_fields"):
        raise AttributeError("Profile must define evaluate_fields(coords, params)")

    field_order = list(profile.FIELDS)
    if not field_order:
        raise ValueError("FIELDS cannot be empty")

    coords = read_mesh_pts_xml(args.mesh_pts)
    fieldconvert = resolve_fieldconvert(args.fieldconvert)

    args.samples_dir.mkdir(parents=True, exist_ok=True)
    if args.params_jsonl is None:
        args.params_jsonl = args.samples_dir / "sample_params.jsonl"

    rng = np.random.default_rng(args.seed)
    dim = int(np.asarray(coords.get("dim", np.asarray([2], dtype=int)))[0])
    coord_keys = ["x", "y", "z"][: max(1, min(3, dim))]
    bbox = {
        k: [float(np.min(np.asarray(coords[k], dtype=float))), float(np.max(np.asarray(coords[k], dtype=float)))]
        for k in coord_keys
    }
    coords_meta = {
        "dim": dim,
        "npts": int(np.asarray(coords["x"], dtype=float).size),
        "bbox": bbox,
    }

    sample_sig = inspect.signature(profile.sample_parameters)
    n_sample_args = len(sample_sig.parameters)
    if n_sample_args not in (2, 3):
        raise TypeError(
            "sample_parameters must accept (rng, sample_index) or "
            "(rng, sample_index, coords_meta)"
        )
    out_files: list[Path] = []

    with args.params_jsonl.open("w", encoding="utf-8") as params_fp:
        for i in range(args.n_samples):
            if n_sample_args == 3:
                params = profile.sample_parameters(rng, i, coords_meta)
            else:
                params = profile.sample_parameters(rng, i)
            fields = profile.evaluate_fields(coords, params)

            pts_file = args.samples_dir / f"sample_{i:05d}.pts"
            fld_file = args.samples_dir / f"sample_{i:05d}.fld"

            write_pts_fields(pts_file, coords, fields, field_order)
            convert_pts_to_fld(fieldconvert, args.case_xml, pts_file, fld_file)

            if not args.keep_pts:
                pts_file.unlink(missing_ok=True)

            out_files.append(fld_file)
            rec = {"sample": i, "params": params}
            params_fp.write(json.dumps(rec) + "\n")

            if (i + 1) % max(1, args.n_samples // 10) == 0 or (i + 1) == args.n_samples:
                print(f"[{i + 1:4d}/{args.n_samples:4d}] generated {fld_file.name}")

    build_manifest(args.manifest, out_files)
    print(f"Generated {len(out_files)} samples in {args.samples_dir}")
    print(f"Manifest: {args.manifest}")
    print(f"Params: {args.params_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
