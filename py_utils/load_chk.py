#!/usr/bin/env python3
"""
Load Nektar++ checkpoint (.chk) files and optional DO archive/state files.

Mean fields (`u,v,p`) are read from checkpoints by calling FieldConvert and
interpolating to a fixed Cartesian grid (`interppoints`), then stored as:

    u[ny, nx, t], v[ny, nx, t], p[ny, nx, t]

Optional DO archive (`DO_ARCHIVE_V1`) or legacy DO state files (`DO_STATE_V1`)
are parsed and stored as:

    mode_u_native[S, nPts, t_do], mode_v_native[S, nPts, t_do]
    yi[nParticles, S, t_do]
    ou_state[nParticles, S, t_do]

Important limitation:
    DO mode fields in `.dat` are stored on Nektar internal physical points, and
    the files do not contain those point coordinates or modal coefficients.
    Therefore this script cannot exactly interpolate mode fields to a Cartesian
    grid from `.dat` alone. It stores them on their native DO physical-point
    ordering instead.

Use like (bash)
python3 py_utils/load_chk.py \
  --xml cases/vortex/casefile.xml \
  --chk-dir cases/vortex/output \
  --do-archive-file cases/vortex/do_archive_v1.dat \
  --nx 200 --ny 120 \
  --out /tmp/vortex_do_series.npz   # or .h5/.hdf5
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import subprocess
import tempfile
import base64
import zlib
import hashlib
from pathlib import Path
from typing import Iterable, Iterator
from xml.etree import ElementTree as ET

import numpy as np
try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover - optional dependency at runtime
    h5py = None


def _parse_parameters(xml_root: ET.Element) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in xml_root.findall(".//CONDITIONS/PARAMETERS/P"):
        txt = " ".join((p.text or "").split())
        if "=" not in txt:
            continue
        name, value = txt.split("=", 1)
        out[name.strip()] = value.strip()
    return out


def _float_or_none(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _find_checkpoint_frequency(xml_root: ET.Element) -> int | None:
    for filt in xml_root.findall(".//FILTERS/FILTER"):
        if filt.attrib.get("TYPE") != "Checkpoint":
            continue
        for prm in filt.findall("PARAM"):
            if prm.attrib.get("NAME") == "OutputFrequency":
                txt = " ".join((prm.text or "").split())
                try:
                    return int(float(txt))
                except ValueError:
                    return None
    return None


def _sorted_chk_files(output_dir: Path, pattern: str) -> list[Path]:
    files = sorted(output_dir.glob(pattern))

    def _key(path: Path) -> tuple[int, str]:
        m = re.search(r"_(\d+)\.chk$", path.name)
        idx = int(m.group(1)) if m else -1
        return idx, path.name

    return sorted(files, key=_key)


def _sorted_do_dat_files(dat_dir: Path, pattern: str) -> list[Path]:
    files = sorted(dat_dir.glob(pattern))

    def _key(path: Path) -> tuple[int, str]:
        m = re.search(r"_(\d+)\.dat$", path.name)
        idx = int(m.group(1)) if m else -1
        return idx, path.name

    return sorted(files, key=_key)


def _sorted_archive_part_files(archive_file: Path) -> list[Path]:
    parent = archive_file.parent if archive_file.parent != Path("") else Path(".")
    prefix = archive_file.name + ".part_"
    suffix = ".tmp"
    files = [p for p in parent.glob(f"{archive_file.name}.part_*.tmp") if p.is_file()]

    def _key(path: Path) -> tuple[int, str]:
        m = re.search(r"\.part_(\d+)\.tmp$", path.name)
        idx = int(m.group(1)) if m else -1
        return idx, path.name

    return sorted(files, key=_key)


def _infer_do_archive_file(case_dir: Path) -> Path:
    """Best-effort archive path inference from case output directory."""
    out_dir = (case_dir / "output").resolve()
    candidates: list[Path] = []
    if out_dir.is_dir():
        candidates.extend([p.resolve() for p in out_dir.glob("*.do_archive.h5") if p.is_file()])
        candidates.extend([p.resolve() for p in out_dir.glob("*.do_archive")    if p.is_file()])
        candidates.extend([p.resolve() for p in out_dir.glob("*_do_archive_v*.dat") if p.is_file()])
        candidates.extend([p.resolve() for p in out_dir.glob("do_archive_v*.dat") if p.is_file()])
    if candidates:
        # Prefer higher version number, then newest mtime.
        def _key(p: Path) -> tuple[int, float, str]:
            m = re.search(r"_v(\d+)\.dat$", p.name)
            ver = int(m.group(1)) if m else -1
            try:
                mt = p.stat().st_mtime
            except OSError:
                mt = 0.0
            return ver, mt, p.name

        return sorted(candidates, key=_key, reverse=True)[0]

    # Fallback legacy default.
    return (out_dir / "do_archive_v1.dat").resolve()


def _decode_chk_elements_blob(el: ET.Element) -> np.ndarray:
    comp = el.attrib.get("COMPRESSED", "")
    bits = int(el.attrib.get("BITSIZE", "64"))
    text = "".join((el.text or "").split())
    raw = base64.b64decode(text)
    if "B64Z" in comp:
        raw = zlib.decompress(raw)
    if bits == 64:
        return np.frombuffer(raw, dtype="<f8").copy()
    if bits == 32:
        return np.frombuffer(raw, dtype="<f4").copy()
    raise ValueError(f"Unsupported BITSIZE={bits}")


def _encode_chk_elements_blob(arr: np.ndarray, el: ET.Element) -> str:
    comp = el.attrib.get("COMPRESSED", "")
    bits = int(el.attrib.get("BITSIZE", "64"))
    if bits == 64:
        raw = np.asarray(arr, dtype="<f8").tobytes(order="C")
    elif bits == 32:
        raw = np.asarray(arr, dtype="<f4").tobytes(order="C")
    else:
        raise ValueError(f"Unsupported BITSIZE={bits}")
    if "B64Z" in comp:
        raw = zlib.compress(raw)
    return base64.b64encode(raw).decode("ascii")


def _parse_elem_id_count(id_str: str) -> int:
    total = 0
    for part in id_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            total += int(b) - int(a) + 1
        else:
            total += 1
    return total


def _parse_num_modes_per_dir(num_str: str) -> list[int]:
    if ":" in num_str:
        num_str = num_str.split(":", 1)[1]
    return [int(x.strip()) for x in num_str.split(",") if x.strip()]


def _coeffs_per_element(shape: str, nmd: list[int]) -> int:
    s = shape.lower()
    if s in ("quadrilateral", "quad"):
        return nmd[0] * nmd[1]
    if s in ("triangle", "tri"):
        p, q = nmd
        if p == q:
            return p * (p + 1) // 2
        m, M = min(p, q), max(p, q)
        return m * M - m * (m - 1) // 2
    if s in ("segment", "line"):
        return nmd[0]
    if s in ("hexahedron", "hex"):
        return nmd[0] * nmd[1] * nmd[2]
    raise ValueError(f"Unsupported shape for coeff split: {shape}")


def _build_chk_mode_writer(template_chk: Path):
    root = ET.parse(template_chk).getroot()
    template_root = root
    blocks = []
    totals: dict[str, int] = {}
    for el in template_root.iter("ELEMENTS"):
        fields = [v.strip() for v in el.attrib["FIELDS"].split(",") if v.strip()]
        ne = _parse_elem_id_count(el.attrib["ID"])
        cpe = _coeffs_per_element(el.attrib["SHAPE"], _parse_num_modes_per_dir(el.attrib["NUMMODESPERDIR"]))
        n_per_field = ne * cpe
        blocks.append((el, fields, n_per_field))
        for f in fields:
            totals[f] = totals.get(f, 0) + n_per_field

    def _write(out_chk: Path, coeff_map: dict[str, np.ndarray]) -> None:
        r = copy.deepcopy(template_root)
        src_to_dst = list(zip(template_root.iter("ELEMENTS"), r.iter("ELEMENTS")))
        offsets = {k: 0 for k in totals}
        coeff_map_local = {k: np.asarray(v) for k, v in coeff_map.items()}
        for src_el, dst_el in src_to_dst:
            fields = [v.strip() for v in dst_el.attrib["FIELDS"].split(",") if v.strip()]
            ne = _parse_elem_id_count(dst_el.attrib["ID"])
            cpe = _coeffs_per_element(dst_el.attrib["SHAPE"], _parse_num_modes_per_dir(dst_el.attrib["NUMMODESPERDIR"]))
            n_per_field = ne * cpe
            arr = _decode_chk_elements_blob(src_el).reshape(len(fields), n_per_field)
            for j, f in enumerate(fields):
                if f in coeff_map_local:
                    start = offsets[f]
                    end = start + n_per_field
                    if end > coeff_map_local[f].size:
                        raise ValueError(f"Not enough coeffs supplied for field {f}")
                    arr[j, :] = coeff_map_local[f][start:end]
                    offsets[f] = end
            dst_el.text = _encode_chk_elements_blob(arr.reshape(-1), dst_el)
        ET.ElementTree(r).write(out_chk, encoding="utf-8", xml_declaration=True)

    return _write, totals


def _write_points_xml(path: Path, x: np.ndarray, y: np.ndarray) -> None:
    if x.shape != y.shape:
        raise ValueError("x and y grids must have the same shape.")
    ny, nx = x.shape

    lines = [
        '<?xml version="1.0" encoding="utf-8" ?>',
        "<NEKTAR>",
        '  <POINTS DIM="2" FIELDS="">',
    ]
    for j in range(ny):
        for i in range(nx):
            lines.append(f"    {x[j, i]:.16e} {y[j, i]:.16e}")
    lines += ["  </POINTS>", "</NEKTAR>", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _read_interppoints_file(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8", errors="replace")
    rows: list[list[float]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "<" in line or ">" in line:
            continue
        # Support csv and whitespace output with the same parser.
        line = line.replace(",", " ")
        parts = line.split()
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            continue
        if vals:
            rows.append(vals)

    if not rows:
        raise ValueError(f"No numeric interpolation data found in {path}")

    arr = np.asarray(rows, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Unexpected interpolated data shape: {arr.shape}")
    return arr


def _guess_velocity_columns(ncols: int, dim: int = 2) -> tuple[int, int]:
    # Typical interppoints output:
    # 2D -> x y u v p       (u=2, v=3)
    # 3D -> x y z u v w p   (u=3, v=4)
    if dim == 2 and ncols >= 4:
        return 2, 3
    if dim == 3 and ncols >= 5:
        return 3, 4
    # Conservative fallback: assume first two columns are coords.
    if ncols >= 4:
        return 2, 3
    raise ValueError(
        f"Cannot infer velocity columns from {ncols} columns. "
        "Use --u-col and --v-col."
    )


def _guess_pressure_column(ncols: int, dim: int = 2) -> int:
    # Typical interppoints output:
    # 2D -> x y u v p       (p=4)
    # 3D -> x y z u v w p   (p=6)
    if dim == 2 and ncols >= 5:
        return 4
    if dim == 3 and ncols >= 7:
        return 6
    if ncols >= 5:
        return ncols - 1
    raise ValueError(
        f"Cannot infer pressure column from {ncols} columns. Use --p-col."
    )


def _run_fieldconvert(
    fieldconvert: Path,
    xml_path: Path,
    chk_path: Path,
    pts_path: Path,
    out_path: Path,
) -> None:
    # FieldConvert needs both geometry and conditions XMLs. Conventional
    # case-dir layout has geometry.xml alongside casefile.xml; if present,
    # comma-prepend it so SessionReader / MeshGraphIO finds the geometry.
    geom = xml_path.parent / "geometry.xml"
    fromxml = (
        f"{geom},{xml_path}"
        if geom.exists() and geom.resolve() != xml_path.resolve()
        else str(xml_path)
    )
    cmd = [
        str(fieldconvert),
        "-f",
        "-e",
        "-m",
        f"interppoints:fromxml={fromxml}:fromfld={chk_path}:topts={pts_path}",
        str(out_path),
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "FieldConvert failed.\n"
            f"Command: {' '.join(cmd)}\n\n"
            f"Output:\n{proc.stdout}"
        )


def _try_run_text(cmd: list[str], cwd: Path | None = None) -> str | None:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _file_sha256(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _git_provenance(repo_dir: Path) -> dict[str, object]:
    out: dict[str, object] = {}
    git_root = _try_run_text(["git", "rev-parse", "--show-toplevel"], cwd=repo_dir)
    if git_root is None:
        out["git_available"] = False
        return out
    out["git_available"] = True
    out["git_root"] = git_root
    out["git_commit"] = _try_run_text(["git", "rev-parse", "HEAD"], cwd=repo_dir)
    out["git_branch"] = _try_run_text(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_dir
    )
    # `git status --porcelain` is robust and cheap enough for metadata.
    status = _try_run_text(["git", "status", "--porcelain"], cwd=repo_dir)
    if status is not None:
        out["git_dirty"] = bool(status.strip())
    return out


def _is_hdf5_path(path: Path) -> bool:
    return path.suffix.lower() in {".h5", ".hdf5"}


def _write_hdf5_dataset(h5, key: str, value) -> None:
    if isinstance(value, str):
        dt = h5py.string_dtype(encoding="utf-8")
        h5.create_dataset(key, data=value, dtype=dt)
        return
    arr = np.asarray(value)
    if arr.dtype == object:
        flat = [str(x) for x in arr.reshape(-1)]
        dt = h5py.string_dtype(encoding="utf-8")
        ds = h5.create_dataset(key, shape=arr.shape, dtype=dt)
        ds[...] = np.asarray(flat, dtype=object).reshape(arr.shape)
        return
    if arr.dtype.kind in {"U", "S"}:
        dt = h5py.string_dtype(encoding="utf-8")
        ds = h5.create_dataset(key, shape=arr.shape, dtype=dt)
        ds[...] = arr.astype(str)
        return
    h5.create_dataset(key, data=arr, compression="gzip")


def _save_arrays(path: Path, arrays: dict[str, object]) -> None:
    if _is_hdf5_path(path):
        if h5py is None:
            raise RuntimeError(
                "h5py is required to write HDF5 output (.h5/.hdf5). "
                "Install h5py or use a .npz output path."
            )
        with h5py.File(path, "w") as h5:
            for k, v in arrays.items():
                _write_hdf5_dataset(h5, k, v)
    else:
        np.savez_compressed(path, **arrays)


def _infer_times(
    chk_files: Iterable[Path],
    session_root: ET.Element,
    dt_override: float | None,
    stride_override: int | None,
) -> np.ndarray:
    pmap = _parse_parameters(session_root)
    dt = dt_override if dt_override is not None else _float_or_none(pmap.get("TimeStep"))
    stride = (
        stride_override
        if stride_override is not None
        else _find_checkpoint_frequency(session_root)
    )
    if stride is None:
        stride = 1
    if dt is None:
        dt = 1.0

    times: list[float] = []
    for chk in chk_files:
        m = re.search(r"_(\d+)\.chk$", chk.name)
        idx = int(m.group(1)) if m else 0
        times.append(float(idx * stride) * dt)
    return np.asarray(times, dtype=float)


def _infer_do_times(dat_files: Iterable[Path], dt: float | None) -> np.ndarray:
    if dt is None:
        # If unknown, use suffix index as time-like coordinate.
        scale = 1.0
    else:
        scale = dt

    times: list[float] = []
    for dat in dat_files:
        m = re.search(r"_(\d+)\.dat$", dat.name)
        idx = int(m.group(1)) if m else 0
        times.append(idx * scale)
    return np.asarray(times, dtype=float)


def _load_do_state_file(dat_path: Path) -> dict[str, object]:
    with dat_path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split()
        if len(header) != 5 or header[0] != "DO_STATE_V1":
            raise ValueError(f"Invalid DO state header in {dat_path}")
        n_vel, n_pts, n_modes, n_particles = map(int, header[1:])

        rng_state = int(f.readline().strip())
        vals = np.fromiter((float(line) for line in f if line.strip()),
                           dtype=np.float64)

    n_mode_vals = n_modes * n_vel * n_pts
    n_y_vals = n_particles * n_modes
    n_ou_vals = n_particles * n_modes
    expected = n_mode_vals + n_y_vals + n_ou_vals
    if vals.size != expected:
        raise ValueError(
            f"{dat_path.name}: expected {expected} values after header, got {vals.size}"
        )

    k0 = 0
    k1 = k0 + n_mode_vals
    k2 = k1 + n_y_vals
    k3 = k2 + n_ou_vals
    mode_phys = vals[k0:k1].reshape(n_modes, n_vel, n_pts)
    yi = vals[k1:k2].reshape(n_particles, n_modes)
    ou_state = vals[k2:k3].reshape(n_particles, n_modes)

    return {
        "n_vel": n_vel,
        "n_pts": n_pts,
        "n_modes": n_modes,
        "n_particles": n_particles,
        "rng_state": rng_state,
        "mode_phys": mode_phys,
        "yi": yi,
        "ou_state": ou_state,
    }


def _parse_archive_snapshot_lines(lines: Iterator[str], ctx: str) -> dict[str, object]:
    line = next(lines, None)
    if line is None:
        raise ValueError(f"{ctx}: unexpected EOF while reading RNG line")
    toks = line.strip().split()
    if len(toks) != 2 or toks[0] != "RNG":
        raise ValueError(f"{ctx}: expected 'RNG <state>', got {line!r}")
    rng_state = int(toks[1])

    line = next(lines, None)
    if line is None or line.strip() != "MEAN_FIELDS_BEGIN":
        raise ValueError(f"{ctx}: expected MEAN_FIELDS_BEGIN")
    mean_coeffs: dict[str, np.ndarray] = {}
    while True:
        line = next(lines, None)
        if line is None:
            raise ValueError(f"{ctx}: unexpected EOF in mean fields")
        s = line.strip()
        if s == "MEAN_FIELDS_END":
            break
        toks = s.split()
        if len(toks) != 3 or toks[0] != "FIELD":
            raise ValueError(f"{ctx}: expected 'FIELD <name> <ncoeffs>', got {s!r}")
        name = toks[1]
        ncoeffs = int(toks[2])
        vals = np.empty((ncoeffs,), dtype=np.float64)
        for i in range(ncoeffs):
            vline = next(lines, None)
            if vline is None:
                raise ValueError(f"{ctx}: unexpected EOF in FIELD {name}")
            vals[i] = float(vline.strip())
        mean_coeffs[name] = vals

    line = next(lines, None)
    if line is None:
        raise ValueError(f"{ctx}: unexpected EOF before MODE_FIELDS_BEGIN")
    toks = line.strip().split()
    if len(toks) != 4 or toks[0] != "MODE_FIELDS_BEGIN":
        raise ValueError(f"{ctx}: expected MODE_FIELDS_BEGIN, got {line!r}")
    n_modes, n_vel, n_pts = map(int, toks[1:])
    n_mode_vals = n_modes * n_vel * n_pts
    vals = np.empty((n_mode_vals,), dtype=np.float64)
    for i in range(n_mode_vals):
        vline = next(lines, None)
        if vline is None:
            raise ValueError(f"{ctx}: unexpected EOF in mode fields")
        vals[i] = float(vline.strip())
    mode_phys = vals.reshape(n_modes, n_vel, n_pts)
    line = next(lines, None)
    if line is None or line.strip() != "MODE_FIELDS_END":
        raise ValueError(f"{ctx}: expected MODE_FIELDS_END")

    # Optional V2 mode coefficient block (for exact Nektar reconstruction).
    mode_coeffs = None
    line = next(lines, None)
    if line is None:
        raise ValueError(f"{ctx}: unexpected EOF after MODE_FIELDS_END")
    s = line.strip()
    if s.startswith("MODE_COEFFS_BEGIN"):
        toks = s.split()
        if len(toks) != 3:
            raise ValueError(f"{ctx}: malformed MODE_COEFFS_BEGIN")
        n_modes_c, n_vel_c = map(int, toks[1:])
        if n_modes_c != n_modes or n_vel_c != n_vel:
            raise ValueError(f"{ctx}: MODE_COEFFS dims mismatch")
        mode_coeffs_list: list[list[np.ndarray]] = [
            [None for _ in range(n_vel)] for _ in range(n_modes)
        ]  # type: ignore[list-item]
        for _ in range(n_modes * n_vel):
            hdr = next(lines, None)
            if hdr is None:
                raise ValueError(f"{ctx}: unexpected EOF in MODE_COEFFS")
            ht = hdr.strip().split()
            if len(ht) != 4 or ht[0] != "MODE_COEFF":
                raise ValueError(f"{ctx}: malformed MODE_COEFF header {hdr!r}")
            m_idx, c_idx, ncoeffs = int(ht[1]), int(ht[2]), int(ht[3])
            vals_c = np.empty((ncoeffs,), dtype=np.float64)
            for i in range(ncoeffs):
                vline = next(lines, None)
                if vline is None:
                    raise ValueError(f"{ctx}: unexpected EOF in MODE_COEFF values")
                vals_c[i] = float(vline.strip())
            mode_coeffs_list[m_idx][c_idx] = vals_c
        end = next(lines, None)
        if end is None or end.strip() != "MODE_COEFFS_END":
            raise ValueError(f"{ctx}: expected MODE_COEFFS_END")
        mode_coeffs = mode_coeffs_list
        line = next(lines, None)
        if line is None:
            raise ValueError(f"{ctx}: unexpected EOF before YI_BEGIN")

    toks = line.strip().split()
    if len(toks) != 3 or toks[0] != "YI_BEGIN":
        raise ValueError(f"{ctx}: expected YI_BEGIN, got {line!r}")
    n_particles, n_modes_y = map(int, toks[1:])
    if n_modes_y != n_modes:
        raise ValueError(f"{ctx}: YI nModes mismatch ({n_modes_y} != {n_modes})")
    yi_vals = np.empty((n_particles * n_modes,), dtype=np.float64)
    for i in range(yi_vals.size):
        vline = next(lines, None)
        if vline is None:
            raise ValueError(f"{ctx}: unexpected EOF in Yi")
        yi_vals[i] = float(vline.strip())
    yi = yi_vals.reshape(n_particles, n_modes)
    line = next(lines, None)
    if line is None or line.strip() != "YI_END":
        raise ValueError(f"{ctx}: expected YI_END")

    line = next(lines, None)
    if line is None:
        raise ValueError(f"{ctx}: unexpected EOF before OU_BEGIN")
    toks = line.strip().split()
    if len(toks) != 3 or toks[0] != "OU_BEGIN":
        raise ValueError(f"{ctx}: expected OU_BEGIN, got {line!r}")
    n_particles_ou, n_modes_ou = map(int, toks[1:])
    if n_particles_ou != n_particles or n_modes_ou != n_modes:
        raise ValueError(f"{ctx}: OU dims mismatch")
    ou_vals = np.empty((n_particles * n_modes,), dtype=np.float64)
    for i in range(ou_vals.size):
        vline = next(lines, None)
        if vline is None:
            raise ValueError(f"{ctx}: unexpected EOF in OU state")
        ou_vals[i] = float(vline.strip())
    ou_state = ou_vals.reshape(n_particles, n_modes)
    line = next(lines, None)
    if line is None or line.strip() != "OU_END":
        raise ValueError(f"{ctx}: expected OU_END")

    diag_cov = None
    mode_norms = None
    ortho_err = None
    while True:
        line = next(lines, None)
        if line is None:
            raise ValueError(f"{ctx}: unexpected EOF before END_SNAPSHOT")
        s = line.strip()
        if s == "END_SNAPSHOT":
            break
        toks = s.split()
        if not toks:
            continue
        if toks[0] == "DO_DIAG_COV_BEGIN":
            if len(toks) != 2:
                raise ValueError(f"{ctx}: malformed DO_DIAG_COV_BEGIN")
            nvals = int(toks[1])
            vals = np.empty((nvals,), dtype=np.float64)
            for i in range(nvals):
                vline = next(lines, None)
                if vline is None:
                    raise ValueError(f"{ctx}: unexpected EOF in DO_DIAG_COV")
                vals[i] = float(vline.strip())
            end = next(lines, None)
            if end is None or end.strip() != "DO_DIAG_COV_END":
                raise ValueError(f"{ctx}: expected DO_DIAG_COV_END")
            diag_cov = vals
            continue
        if toks[0] == "DO_MODE_NORMS_BEGIN":
            if len(toks) != 2:
                raise ValueError(f"{ctx}: malformed DO_MODE_NORMS_BEGIN")
            nvals = int(toks[1])
            vals = np.empty((nvals,), dtype=np.float64)
            for i in range(nvals):
                vline = next(lines, None)
                if vline is None:
                    raise ValueError(f"{ctx}: unexpected EOF in DO_MODE_NORMS")
                vals[i] = float(vline.strip())
            end = next(lines, None)
            if end is None or end.strip() != "DO_MODE_NORMS_END":
                raise ValueError(f"{ctx}: expected DO_MODE_NORMS_END")
            mode_norms = vals
            continue
        if toks[0] == "DO_ORTHO_ERR":
            if len(toks) != 2:
                raise ValueError(f"{ctx}: malformed DO_ORTHO_ERR line")
            ortho_err = float(toks[1])
            continue
        raise ValueError(f"{ctx}: unexpected snapshot trailer line {s!r}")

    return {
        "rng_state": rng_state,
        "mean_coeffs": mean_coeffs,
        "n_modes": n_modes,
        "n_vel": n_vel,
        "n_pts": n_pts,
        "n_particles": n_particles,
        "mode_phys": mode_phys,
        "mode_coeffs": mode_coeffs,
        "yi": yi,
        "ou_state": ou_state,
        "diag_cov": diag_cov,
        "mode_norms": mode_norms,
        "ortho_err": ortho_err,
    }


def _load_do_archive_h5(archive_path: Path) -> dict[str, object]:
    """Read DO_ARCHIVE_H5_V1 (HDF5) and return the same dict shape that
    `_load_do_archive_file` produces from the text format."""
    import h5py
    with h5py.File(archive_path, "r") as f:
        hdr = f["header"]
        n_modes     = int(hdr.attrs["n_modes"])
        n_particles = int(hdr.attrs["n_particles"])
        n_vel       = int(hdr.attrs["n_vel"])
        n_phys      = int(hdr.attrs["n_phys"])
        dt          = float(hdr.attrs["dt"])
        var_names   = [v.decode() if isinstance(v, bytes) else str(v)
                       for v in hdr.attrs["var_names"]]
        snap_names  = sorted(
            [k for k in f.keys() if k.startswith("snap_")],
            key=lambda k: int(k.split("_")[1]))
        snapshots = []
        for sn in snap_names:
            g = f[sn]
            mean_coeffs = {v: g[f"mean_{v}"][...] for v in var_names}
            mode_phys_flat   = g["mode_phys"][...]
            mode_coeffs_flat = g["mode_coeffs"][...]
            yi_flat          = g["Yi"][...]
            n_pts = mode_phys_flat.size // (n_modes * n_vel)
            n_co  = mode_coeffs_flat.size // (n_modes * n_vel)
            mode_phys = mode_phys_flat.reshape(n_modes, n_vel, n_pts)
            mode_coeffs_list = [
                [mode_coeffs_flat[(m * n_vel + c) * n_co:
                                  (m * n_vel + c + 1) * n_co].copy()
                 for c in range(n_vel)]
                for m in range(n_modes)
            ]
            yi = yi_flat.reshape(n_particles, n_modes)
            snapshots.append({
                "step":        int(g.attrs["step"]),
                "time":        float(g.attrs["time"]),
                "rng_state":   0,
                "mean_coeffs": mean_coeffs,
                "n_modes":     n_modes,
                "n_vel":       n_vel,
                "n_pts":       n_pts,
                "n_particles": n_particles,
                "mode_phys":   mode_phys,
                "mode_coeffs": mode_coeffs_list,
                "yi":          yi,
                "ou_state":    np.zeros((n_particles, n_modes), dtype=np.float64),
                "diag_cov":    None,
                "mode_norms":  None,
                "ortho_err":   None,
            })
    return {
        "header":   {"N_MODES": [str(n_modes)], "N_PARTICLES": [str(n_particles)],
                     "N_VEL": [str(n_vel)], "N_PTS": [str(n_phys)],
                     "DT": [str(dt)]},
        "snapshots": snapshots,
        "version":   "DO_ARCHIVE_H5_V1",
    }


def _load_do_archive_file(archive_path: Path) -> dict[str, object]:
    # Auto-detect: HDF5 if filename ends in .h5 OR file starts with the HDF5
    # magic 0x894844460d0a1a0a; otherwise treat as text DO_ARCHIVE_V1/V2.
    is_h5 = str(archive_path).endswith(".h5")
    if not is_h5:
        try:
            with open(archive_path, "rb") as fb:
                is_h5 = fb.read(8) == b"\x89HDF\r\n\x1a\n"
        except OSError:
            is_h5 = False
    if is_h5:
        return _load_do_archive_h5(archive_path)
    with archive_path.open("r", encoding="utf-8") as f:
        lines = iter(f)
        line = next(lines, None)
        if line is None or line.strip() not in {"DO_ARCHIVE_V1", "DO_ARCHIVE_V2"}:
            raise ValueError(f"{archive_path}: missing DO_ARCHIVE_V1/V2 header")
        archive_version = line.strip()

        header: dict[str, object] = {}
        while True:
            line = next(lines, None)
            if line is None:
                raise ValueError(f"{archive_path}: unexpected EOF in header")
            s = line.strip()
            if s == "END_HEADER":
                break
            toks = s.split()
            if not toks:
                continue
            key, vals = toks[0], toks[1:]
            header[key] = vals

        snapshots = []
        for line in lines:
            s = line.strip()
            if not s:
                continue
            toks = s.split()
            if toks[0] != "SNAPSHOT":
                raise ValueError(f"{archive_path}: expected SNAPSHOT, got {s!r}")
            if len(toks) != 3:
                raise ValueError(f"{archive_path}: malformed SNAPSHOT line {s!r}")
            step = int(toks[1])
            time = float(toks[2])
            snap = _parse_archive_snapshot_lines(lines, f"{archive_path.name}:snapshot {step}")
            snap["step"] = step
            snap["time"] = time
            snapshots.append(snap)

    return {"header": header, "snapshots": snapshots, "version": archive_version}


def _load_do_archive_parts(part_files: list[Path]) -> dict[str, object]:
    snapshots = []
    for p in part_files:
        with p.open("r", encoding="utf-8") as f:
            lines = iter(f)
            first = next(lines, None)
            if first is None:
                continue
            toks = first.strip().split()
            if len(toks) != 3 or toks[0] != "SNAPSHOT":
                raise ValueError(f"{p}: expected SNAPSHOT line at top")
            step = int(toks[1])
            time = float(toks[2])
            snap = _parse_archive_snapshot_lines(lines, f"{p.name}:snapshot {step}")
            snap["step"] = step
            snap["time"] = time
            snapshots.append(snap)
    snapshots.sort(key=lambda s: (int(s["step"]), float(s["time"])))
    return {"header": None, "snapshots": snapshots}


def _default_xml() -> Path | None:
    cands = [
        Path("casefile.xml"),
        Path("mesh") / "geometry.xml",
    ]
    for c in cands:
        p = c.resolve()
        if p.is_file():
            return p
    return None


def _infer_case_dir_from_xml(xml_path: Path) -> Path:
    # Typical layouts:
    # 1) <case_dir>/casefile.xml
    # 2) <case_dir>/mesh/geometry.xml
    if xml_path.name == "casefile.xml":
        return xml_path.parent
    if xml_path.parent.name == "mesh":
        return xml_path.parent.parent
    return xml_path.parent


def _resolve_chk_dir(chk_dir_arg: Path | None, xml_path: Path) -> Path:
    if chk_dir_arg is not None:
        chk_dir = chk_dir_arg.resolve()
        if not chk_dir.is_dir():
            raise FileNotFoundError(f"Checkpoint directory not found: {chk_dir}")
        return chk_dir

    case_dir = _infer_case_dir_from_xml(xml_path)
    out_dir = case_dir / "output"
    if out_dir.is_dir():
        return out_dir

    # Backward-compatible fallback for cases with checkpoints directly in case_dir.
    if any(case_dir.glob("*.chk")):
        return case_dir

    raise FileNotFoundError(
        f"Could not find checkpoint directory. Tried: {out_dir} and {case_dir}. "
        "Pass --chk-dir explicitly."
    )


def _read_field_format_metadata(fld_path: Path) -> dict:
    """Read DOVelocityCorrectionScheme_* metadata + Yi from a Field-format .fld
    snapshot. Handles both HDF5 .fld (parallel writes) and XML .fld (serial)."""
    with open(fld_path, "rb") as fb:
        is_h5 = fb.read(8) == b"\x89HDF\r\n\x1a\n"
    if is_h5:
        import h5py
        with h5py.File(fld_path, "r") as f:
            attrs = dict(f["NEKTAR/Metadata"].attrs)
        md = {k: (v.decode() if isinstance(v, bytes) else str(v))
              for k, v in attrs.items()}
    else:
        from xml.etree import ElementTree as ET
        meta_node = ET.parse(fld_path).getroot().find("Metadata")
        if meta_node is None:
            raise ValueError(f"{fld_path}: no <Metadata> element")
        md = {child.tag: (child.text or "").strip() for child in meta_node}
    nM = int(md["DOVelocityCorrectionScheme_n_modes"])
    nP = int(md["DOVelocityCorrectionScheme_n_particles"])
    yi = np.frombuffer(bytes.fromhex(md["DOVelocityCorrectionScheme_Yi_hex"]),
                       dtype=np.float64).reshape(nP, nM)
    return {"step":        int(md["DOVelocityCorrectionScheme_step"]),
            "time":        float(md["DOVelocityCorrectionScheme_time"]),
            "n_modes":     nM,
            "n_particles": nP,
            "n_vel":       int(md["DOVelocityCorrectionScheme_n_vel"]),
            "dt":          float(md["DOVelocityCorrectionScheme_dt"]),
            "yi":          yi}


def _field_format_files(case_dir: Path) -> list[Path]:
    """Discover Field-format DO archive snapshot files."""
    out_dir = (case_dir / "output").resolve()
    if not out_dir.is_dir():
        return []
    files = sorted([p for p in out_dir.glob("*.do_*.fld") if p.is_file()],
                   key=lambda p: int(re.search(r"\.do_(\d+)\.fld$", p.name).group(1)))
    return files


def _main_field_format(field_files: list[Path], fieldconvert: Path,
                       xml_path: Path, args: argparse.Namespace) -> int:
    """End-to-end loader for Field-format DO archives. Each .fld snapshot is
    a self-contained Nektar HDF5 file holding mean fields, mode coefficients
    (named mode_<i>_<v>), and Yi as MetaData hex. We run FieldConvert once per
    snapshot to extract everything onto the user's Cartesian grid in a single
    sweep, and write the standard out.h5 schema."""
    if args.limit is not None:
        field_files = field_files[: args.limit]
    nT = len(field_files)
    if nT == 0:
        raise FileNotFoundError("No Field-format snapshot files found.")

    # Cartesian grid points (same convention as the legacy path).
    x1 = np.linspace(args.xmin, args.xmax, args.nx)
    y1 = np.linspace(args.ymin, args.ymax, args.ny)
    xg, yg = np.meshgrid(x1, y1)

    # Probe the first snapshot to discover field names + sizes.
    md0 = _read_field_format_metadata(field_files[0])
    nM, nP, nV = md0["n_modes"], md0["n_particles"], md0["n_vel"]
    var_names = ["u", "v", "p"] if nV == 2 else ["u", "v", "w", "p"]
    mode_field_names = [
        f"mode_{m}_{var_names[c]}" for m in range(nM) for c in range(nV)
    ]

    # Storage.
    mean_u = np.empty((args.ny, args.nx, nT), dtype=np.float64)
    mean_v = np.empty_like(mean_u)
    mean_p = np.empty_like(mean_u)
    mode_u_grid = np.empty((nM, args.ny, args.nx, nT), dtype=np.float64)
    mode_v_grid = np.empty_like(mode_u_grid)
    yi_full = np.empty((nP, nM, nT), dtype=np.float64)
    times   = np.empty(nT, dtype=np.float64)
    steps   = np.empty(nT, dtype=np.int64)

    tmp_root = Path(tempfile.mkdtemp(prefix="load_chk_field_", dir="/tmp"))
    pts_path = tmp_root / "grid.pts"
    _write_points_xml(pts_path, xg, yg)
    try:
        for k, fld in enumerate(field_files):
            md = _read_field_format_metadata(fld)
            yi_full[:, :, k] = md["yi"]
            times[k] = md["time"]
            steps[k] = md["step"]
            out_csv = tmp_root / f"snap_{k:04d}.csv"
            _run_fieldconvert(fieldconvert, xml_path, fld, pts_path, out_csv)
            arr = _read_interppoints_file(out_csv)
            ncol = arr.shape[1]
            # Column layout: x, y, [z], <FIELDS in same order as written>.
            # Means came first, then modes (per writer order in DOVelocityCorrectionScheme).
            base_offset = 2 if nV == 2 else 3
            ucol = base_offset + 0
            vcol = base_offset + 1
            pcol = base_offset + 2
            mean_u[:, :, k] = arr[:, ucol].reshape(args.ny, args.nx)
            mean_v[:, :, k] = arr[:, vcol].reshape(args.ny, args.nx)
            mean_p[:, :, k] = arr[:, pcol].reshape(args.ny, args.nx)
            mode_offset = base_offset + (3 if "p" in var_names else 2)
            for m in range(nM):
                mode_u_grid[m, :, :, k] = arr[
                    :, mode_offset + m * nV + 0
                ].reshape(args.ny, args.nx)
                mode_v_grid[m, :, :, k] = arr[
                    :, mode_offset + m * nV + 1
                ].reshape(args.ny, args.nx)
            print(f"[FIELD {k+1:4d}/{nT:4d}] {fld.name}  "
                  f"step={md['step']} time={md['time']:.4g}")
    finally:
        if not args.keep_temp:
            shutil.rmtree(tmp_root, ignore_errors=True)

    # ---- Write out.h5 ----
    out = (
        args.out.resolve()
        if args.out is not None
        else (xml_path.parent / "output" / "out.h5").resolve()
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "schema":      "DO_ARCHIVE_FIELD_V1",
        "xml":         str(xml_path),
        "n_modes":     nM,
        "n_particles": nP,
        "n_vel":       nV,
        "dt":          md0["dt"],
        "n_snapshots": nT,
        "snapshot_files": [str(f) for f in field_files],
    }
    save_kwargs = {
        "x": xg, "y": yg,
        "u": mean_u, "v": mean_v, "p": mean_p,
        "mode_u": mode_u_grid, "mode_v": mode_v_grid,
        "yi":   np.transpose(yi_full,   (0, 1, 2)),  # (Np, S, T)
        "t":    times,
        "t_do": times,
        "metadata_json": json.dumps(meta, indent=2),
    }
    if str(out).endswith((".h5", ".hdf5")):
        import h5py
        with h5py.File(out, "w") as hf:
            for k, v in save_kwargs.items():
                if isinstance(v, str):
                    hf.create_dataset(k, data=np.string_(v))
                else:
                    hf.create_dataset(k, data=np.asarray(v))
    else:
        np.savez(out, **save_kwargs)
    print(f"\nSaved: {out}")
    print(f"u/v/p shape (Ny,Nx,T): {mean_u.shape}")
    print(f"mode_u/mode_v shape (S,Ny,Nx,T): {mode_u_grid.shape}")
    print(f"yi shape (Np,S,T): {yi_full.shape}")
    print(f"t range: [{times.min():.4g}, {times.max():.4g}]")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--xml",
        type=Path,
        default=None,
        help="XML to use with FieldConvert (default: ./casefile.xml or ./mesh/geometry.xml)",
    )
    ap.add_argument(
        "--fieldconvert",
        type=Path,
        default=Path("/home/isma/nektar_repro/build/utilities/FieldConvert/FieldConvert"),
        help="Path to FieldConvert binary",
    )
    ap.add_argument(
        "--chk-dir",
        type=Path,
        default=None,
        help=(
            "Checkpoint directory. "
            "Default: infer from --xml as <case-dir>/output, then <case-dir>"
        ),
    )
    ap.add_argument(
        "--chk-pattern",
        default="casefile_*.chk",
        help="Glob pattern under checkpoint directory",
    )
    ap.add_argument(
        "--do-archive-file",
        type=Path,
        default=None,
        help=(
            "Path to merged DO archive file (DO_ARCHIVE_V1/V2). "
            "Default: auto-detect under <case-dir>/output (prefers highest version)."
        ),
    )
    ap.add_argument(
        "--do-dat-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing legacy DO state *.dat files "
            "(fallback if no DO archive is found; default: case dir)"
        ),
    )
    ap.add_argument(
        "--do-dat-pattern",
        default="do_state_*.dat",
        help="Glob pattern under DO state directory",
    )
    ap.add_argument("--nx", type=int, default=100, help="Cartesian grid points in x")
    ap.add_argument("--ny", type=int, default=100, help="Cartesian grid points in y")
    ap.add_argument("--xmin", type=float, default=0.0)
    ap.add_argument("--xmax", type=float, default=6.283185307179586)
    ap.add_argument("--ymin", type=float, default=0.0)
    ap.add_argument("--ymax", type=float, default=6.283185307179586)
    ap.add_argument(
        "--u-col",
        type=int,
        default=None,
        help="0-based column index for u in FieldConvert output",
    )
    ap.add_argument(
        "--v-col",
        type=int,
        default=None,
        help="0-based column index for v in FieldConvert output",
    )
    ap.add_argument(
        "--p-col",
        type=int,
        default=None,
        help="0-based column index for p in FieldConvert output",
    )
    ap.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Override TimeStep when constructing time array",
    )
    ap.add_argument(
        "--chk-stride",
        type=int,
        default=None,
        help="Override checkpoint step interval when constructing time array",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process first N checkpoints (for quick tests)",
    )
    ap.add_argument(
        "--chk-step",
        type=int,
        default=1,
        help=(
            "Take every Nth checkpoint/DO snapshot (default 1 = use all). "
            "Applied uniformly to chk files, DO .dat files, DO archive part "
            "files, and DO archive snapshots so their time axes stay aligned."
        ),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path (.npz or .h5/.hdf5). Default: <case-dir>/output/out.h5",
    )
    ap.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary interpolation files",
    )
    ap.add_argument(
        "--interp-modes-from-archive",
        dest="interp_modes_from_archive",
        action="store_true",
        default=True,
        help=(
            "If archive mode coefficients are available (DO_ARCHIVE_V2), "
            "reconstruct and interpolate modes to the same Cartesian grid "
            "(default: enabled; use --no-interp-modes-from-archive to disable)."
        ),
    )
    ap.add_argument(
        "--no-interp-modes-from-archive",
        dest="interp_modes_from_archive",
        action="store_false",
        help="Disable --interp-modes-from-archive.",
    )
    args = ap.parse_args()

    xml_path = args.xml.resolve() if args.xml else _default_xml()
    if xml_path is None or not xml_path.is_file():
        raise FileNotFoundError(
            "Could not find XML. Pass --xml explicitly "
            "(e.g. casefile.xml or mesh/geometry.xml)."
        )
    case_dir = _infer_case_dir_from_xml(xml_path)
    chk_dir = _resolve_chk_dir(args.chk_dir, xml_path)
    do_archive_file = (
        args.do_archive_file.resolve()
        if args.do_archive_file is not None
        else _infer_do_archive_file(case_dir)
    )
    do_dat_dir = (
        args.do_dat_dir.resolve()
        if args.do_dat_dir is not None
        else case_dir.resolve()
    )

    fieldconvert = args.fieldconvert.resolve()
    if not fieldconvert.is_file():
        raise FileNotFoundError(f"FieldConvert not found: {fieldconvert}")

    chk_step = max(1, int(args.chk_step))

    # Field-format archives subsume both chk and DOArchive: each .do_<step>.fld
    # contains means + modes + Yi (Yi as MetaData hex). Take this fast path
    # when present — skip the legacy chk-loop entirely.
    field_files = _field_format_files(case_dir)
    if field_files:
        return _main_field_format(field_files, fieldconvert, xml_path, args)

    chk_files = _sorted_chk_files(chk_dir, args.chk_pattern)
    if chk_step > 1:
        chk_files = chk_files[::chk_step]
    if args.limit is not None:
        chk_files = chk_files[: args.limit]
    if not chk_files:
        raise FileNotFoundError(
            f"No checkpoint files matching '{args.chk_pattern}' in {chk_dir}"
        )
    do_dat_files = _sorted_do_dat_files(do_dat_dir, args.do_dat_pattern)
    if chk_step > 1:
        do_dat_files = do_dat_files[::chk_step]
    if args.limit is not None:
        do_dat_files = do_dat_files[: args.limit]
    archive_part_files = _sorted_archive_part_files(do_archive_file)
    if chk_step > 1:
        archive_part_files = archive_part_files[::chk_step]
    if args.limit is not None:
        archive_part_files = archive_part_files[: args.limit]

    try:
        session_root = ET.parse((case_dir / "casefile.xml")).getroot()
    except Exception:
        # Fall back to xml_path if casefile.xml is unavailable.
        session_root = ET.parse(xml_path).getroot()

    x1 = np.linspace(args.xmin, args.xmax, args.nx, dtype=float)
    y1 = np.linspace(args.ymin, args.ymax, args.ny, dtype=float)
    xg, yg = np.meshgrid(x1, y1)
    npts = xg.size

    tmp_dir = Path(tempfile.mkdtemp(prefix="load_chk_velocity_", dir="/tmp"))
    pts_path = tmp_dir / "grid.pts"
    _write_points_xml(pts_path, xg, yg)

    u_stack = np.empty((len(chk_files), args.ny, args.nx), dtype=float)
    v_stack = np.empty((len(chk_files), args.ny, args.nx), dtype=float)
    p_stack = np.empty((len(chk_files), args.ny, args.nx), dtype=float)

    inferred_u_col: int | None = args.u_col
    inferred_v_col: int | None = args.v_col
    inferred_p_col: int | None = args.p_col

    try:
        for k, chk in enumerate(chk_files):
            out_txt = tmp_dir / f"{chk.stem}_interp.csv"
            _run_fieldconvert(fieldconvert, xml_path, chk, pts_path, out_txt)

            arr = _read_interppoints_file(out_txt)
            if arr.shape[0] != npts:
                raise ValueError(
                    f"Unexpected number of interpolated rows for {chk.name}: "
                    f"{arr.shape[0]} (expected {npts})"
                )

            if inferred_u_col is None or inferred_v_col is None:
                uc, vc = _guess_velocity_columns(arr.shape[1], dim=2)
                inferred_u_col = uc if inferred_u_col is None else inferred_u_col
                inferred_v_col = vc if inferred_v_col is None else inferred_v_col
            if inferred_p_col is None:
                inferred_p_col = _guess_pressure_column(arr.shape[1], dim=2)

            if (
                inferred_u_col >= arr.shape[1]
                or inferred_v_col >= arr.shape[1]
                or inferred_p_col >= arr.shape[1]
            ):
                raise ValueError(
                    f"Column index out of bounds for {chk.name}: "
                    f"u_col={inferred_u_col}, v_col={inferred_v_col}, "
                    f"p_col={inferred_p_col}, ncols={arr.shape[1]}"
                )

            u_flat = arr[:, inferred_u_col]
            v_flat = arr[:, inferred_v_col]
            p_flat = arr[:, inferred_p_col]
            u_stack[k] = u_flat.reshape(args.ny, args.nx)
            v_stack[k] = v_flat.reshape(args.ny, args.nx)
            p_stack[k] = p_flat.reshape(args.ny, args.nx)

            print(
                f"[{k + 1:4d}/{len(chk_files):4d}] loaded {chk.name} "
                f"(cols u={inferred_u_col}, v={inferred_v_col}, p={inferred_p_col})"
            )

    finally:
        if args.keep_temp:
            print(f"Temporary files kept in: {tmp_dir}")
        else:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    t = _infer_times(chk_files, session_root, args.dt, args.chk_stride)
    dt_for_do = args.dt
    if dt_for_do is None:
        pmap = _parse_parameters(session_root)
        dt_for_do = _float_or_none(pmap.get("TimeStep"))

    mode_u_native_t = None
    mode_v_native_t = None
    yi_t = None
    ou_state_t = None
    rng_state_t = None
    do_dims = None
    do_source = "none"
    archive_mean_coeffs_u_t = None
    archive_mean_coeffs_v_t = None
    archive_mean_coeffs_p_t = None
    archive_mode_coeffs_u_t = None
    archive_mode_coeffs_v_t = None
    archive_diag_cov_t = None
    archive_mode_norms_t = None
    archive_ortho_err_t = None
    if do_archive_file.is_file():
        do_source = "archive"
        arch = _load_do_archive_file(do_archive_file)
        snaps = arch["snapshots"]
        if chk_step > 1:
            snaps = snaps[::chk_step]
        if args.limit is not None:
            snaps = snaps[: args.limit]
        if snaps:
            first = snaps[0]
            n_vel = int(first["n_vel"])
            n_pts = int(first["n_pts"])
            n_modes = int(first["n_modes"])
            n_particles = int(first["n_particles"])
            do_dims = np.array([n_vel, n_pts, n_modes, n_particles], dtype=np.int64)
            nd = len(snaps)
            mode_u_native_t = np.empty((n_modes, n_pts, nd), dtype=np.float64)
            mode_v_native_t = np.empty((n_modes, n_pts, nd), dtype=np.float64)
            yi_t = np.empty((n_particles, n_modes, nd), dtype=np.float64)
            ou_state_t = np.empty((n_particles, n_modes, nd), dtype=np.float64)
            rng_state_t = np.empty((nd,), dtype=np.int64)
            t_do = np.empty((nd,), dtype=np.float64)

            # Optional mean coeff snapshots from archive (useful for exact recovery).
            coeff_u0 = first["mean_coeffs"].get("u")
            coeff_v0 = first["mean_coeffs"].get("v")
            coeff_p0 = first["mean_coeffs"].get("p")
            if coeff_u0 is not None:
                archive_mean_coeffs_u_t = np.empty((coeff_u0.size, nd), dtype=np.float64)
            if coeff_v0 is not None:
                archive_mean_coeffs_v_t = np.empty((coeff_v0.size, nd), dtype=np.float64)
            if coeff_p0 is not None:
                archive_mean_coeffs_p_t = np.empty((coeff_p0.size, nd), dtype=np.float64)
            mc0 = first.get("mode_coeffs")
            if mc0 is not None and mc0[0][0] is not None and mc0[0][1] is not None:
                archive_mode_coeffs_u_t = np.empty(
                    (n_modes, mc0[0][0].size, nd), dtype=np.float64
                )
                archive_mode_coeffs_v_t = np.empty(
                    (n_modes, mc0[0][1].size, nd), dtype=np.float64
                )
            diag0 = first.get("diag_cov")
            if diag0 is not None:
                archive_diag_cov_t = np.empty((diag0.size, nd), dtype=np.float64)
            norms0 = first.get("mode_norms")
            if norms0 is not None:
                archive_mode_norms_t = np.empty((norms0.size, nd), dtype=np.float64)
            if first.get("ortho_err") is not None:
                archive_ortho_err_t = np.empty((nd,), dtype=np.float64)

            for k, snap in enumerate(snaps):
                mode_phys = snap["mode_phys"]
                mode_u_native_t[:, :, k] = mode_phys[:, 0, :]
                mode_v_native_t[:, :, k] = mode_phys[:, 1, :]
                yi_t[:, :, k] = snap["yi"]
                ou_state_t[:, :, k] = snap["ou_state"]
                rng_state_t[k] = int(snap["rng_state"])
                t_do[k] = float(snap["time"])
                mc = snap["mean_coeffs"]
                if archive_mean_coeffs_u_t is not None and "u" in mc:
                    archive_mean_coeffs_u_t[:, k] = mc["u"]
                if archive_mean_coeffs_v_t is not None and "v" in mc:
                    archive_mean_coeffs_v_t[:, k] = mc["v"]
                if archive_mean_coeffs_p_t is not None and "p" in mc:
                    archive_mean_coeffs_p_t[:, k] = mc["p"]
                mcoeff = snap.get("mode_coeffs")
                if (
                    mcoeff is not None
                    and archive_mode_coeffs_u_t is not None
                    and archive_mode_coeffs_v_t is not None
                ):
                    for m in range(n_modes):
                        archive_mode_coeffs_u_t[m, :, k] = mcoeff[m][0]
                        archive_mode_coeffs_v_t[m, :, k] = mcoeff[m][1]
                if archive_diag_cov_t is not None and snap.get("diag_cov") is not None:
                    archive_diag_cov_t[:, k] = snap["diag_cov"]
                if (
                    archive_mode_norms_t is not None
                    and snap.get("mode_norms") is not None
                ):
                    archive_mode_norms_t[:, k] = snap["mode_norms"]
                if archive_ortho_err_t is not None and snap.get("ortho_err") is not None:
                    archive_ortho_err_t[k] = float(snap["ortho_err"])
            print(f"Loaded DO archive: {do_archive_file} ({len(snaps)} snapshots)")
        else:
            t_do = np.empty((0,), dtype=float)
    elif archive_part_files:
        do_source = "archive_parts"
        arch = _load_do_archive_parts(archive_part_files)
        snaps = arch["snapshots"]
        if snaps:
            first = snaps[0]
            n_vel = int(first["n_vel"])
            n_pts = int(first["n_pts"])
            n_modes = int(first["n_modes"])
            n_particles = int(first["n_particles"])
            do_dims = np.array([n_vel, n_pts, n_modes, n_particles], dtype=np.int64)
            nd = len(snaps)
            mode_u_native_t = np.empty((n_modes, n_pts, nd), dtype=np.float64)
            mode_v_native_t = np.empty((n_modes, n_pts, nd), dtype=np.float64)
            yi_t = np.empty((n_particles, n_modes, nd), dtype=np.float64)
            ou_state_t = np.empty((n_particles, n_modes, nd), dtype=np.float64)
            rng_state_t = np.empty((nd,), dtype=np.int64)
            t_do = np.empty((nd,), dtype=np.float64)
            diag0 = first.get("diag_cov")
            if diag0 is not None:
                archive_diag_cov_t = np.empty((diag0.size, nd), dtype=np.float64)
            norms0 = first.get("mode_norms")
            if norms0 is not None:
                archive_mode_norms_t = np.empty((norms0.size, nd), dtype=np.float64)
            if first.get("ortho_err") is not None:
                archive_ortho_err_t = np.empty((nd,), dtype=np.float64)
            for k, snap in enumerate(snaps):
                mode_phys = snap["mode_phys"]
                mode_u_native_t[:, :, k] = mode_phys[:, 0, :]
                mode_v_native_t[:, :, k] = mode_phys[:, 1, :]
                yi_t[:, :, k] = snap["yi"]
                ou_state_t[:, :, k] = snap["ou_state"]
                rng_state_t[k] = int(snap["rng_state"])
                t_do[k] = float(snap["time"])
                if archive_diag_cov_t is not None and snap.get("diag_cov") is not None:
                    archive_diag_cov_t[:, k] = snap["diag_cov"]
                if (
                    archive_mode_norms_t is not None
                    and snap.get("mode_norms") is not None
                ):
                    archive_mode_norms_t[:, k] = snap["mode_norms"]
                if archive_ortho_err_t is not None and snap.get("ortho_err") is not None:
                    archive_ortho_err_t[k] = float(snap["ortho_err"])
            print(f"Loaded DO archive parts: {len(archive_part_files)} files ({len(snaps)} snapshots)")
        else:
            t_do = np.empty((0,), dtype=float)
    elif do_dat_files:
        do_source = "legacy_dat"
        t_do = _infer_do_times(do_dat_files, dt_for_do) if do_dat_files else np.empty((0,), dtype=float)
        first = _load_do_state_file(do_dat_files[0])
        n_vel = int(first["n_vel"])
        n_pts = int(first["n_pts"])
        n_modes = int(first["n_modes"])
        n_particles = int(first["n_particles"])
        do_dims = np.array([n_vel, n_pts, n_modes, n_particles], dtype=np.int64)
        if n_vel < 2:
            raise ValueError("DO state file has n_vel < 2; expected at least u,v")

        nd = len(do_dat_files)
        mode_u_native_t = np.empty((n_modes, n_pts, nd), dtype=np.float64)
        mode_v_native_t = np.empty((n_modes, n_pts, nd), dtype=np.float64)
        yi_t = np.empty((n_particles, n_modes, nd), dtype=np.float64)
        ou_state_t = np.empty((n_particles, n_modes, nd), dtype=np.float64)
        rng_state_t = np.empty((nd,), dtype=np.int64)

        for k, dat in enumerate(do_dat_files):
            ds = _load_do_state_file(dat)
            dims_here = (
                int(ds["n_vel"]),
                int(ds["n_pts"]),
                int(ds["n_modes"]),
                int(ds["n_particles"]),
            )
            if dims_here != (n_vel, n_pts, n_modes, n_particles):
                raise ValueError(
                    f"DO state dimensions changed at {dat.name}: {dims_here} "
                    f"!= {(n_vel, n_pts, n_modes, n_particles)}"
                )
            mode_phys = ds["mode_phys"]
            mode_u_native_t[:, :, k] = mode_phys[:, 0, :]
            mode_v_native_t[:, :, k] = mode_phys[:, 1, :]
            yi_t[:, :, k] = ds["yi"]
            ou_state_t[:, :, k] = ds["ou_state"]
            rng_state_t[k] = int(ds["rng_state"])
            print(f"[DO {k + 1:4d}/{nd:4d}] loaded {dat.name}")
    else:
        t_do = np.empty((0,), dtype=float)

    out = (
        args.out.resolve()
        if args.out is not None
        else (case_dir / "output" / "out.h5").resolve()
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    mode_u_grid = None
    mode_v_grid = None
    if (
        args.interp_modes_from_archive
        and archive_mode_coeffs_u_t is not None
        and archive_mode_coeffs_v_t is not None
        and chk_files
    ):
        # Rebuild temporary chk snapshots from archived mode coefficients and
        # interpolate with FieldConvert, using the first checkpoint as template.
        write_mode_chk, coeff_totals = _build_chk_mode_writer(chk_files[0])
        p_zeros = np.zeros(coeff_totals.get("p", 0), dtype=float)
        nd = archive_mode_coeffs_u_t.shape[2]
        nm = archive_mode_coeffs_u_t.shape[0]
        mode_u_grid = np.empty((nm, args.ny, args.nx, nd), dtype=float)
        mode_v_grid = np.empty((nm, args.ny, args.nx, nd), dtype=float)
        mode_interp_root = Path(tempfile.mkdtemp(prefix="load_chk_mode_interp_", dir="/tmp"))
        mode_pts_path = mode_interp_root / "grid.pts"
        _write_points_xml(mode_pts_path, xg, yg)
        mode_tmp_dir = mode_interp_root / "mode_interp"
        mode_tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            for k in range(nd):
                for m in range(nm):
                    mode_chk = mode_tmp_dir / f"mode_m{m:03d}_t{k:04d}.chk"
                    mode_out = mode_tmp_dir / f"mode_m{m:03d}_t{k:04d}.csv"
                    write_mode_chk(
                        mode_chk,
                        {
                            "u": archive_mode_coeffs_u_t[m, :, k],
                            "v": archive_mode_coeffs_v_t[m, :, k],
                            "p": p_zeros,
                        },
                    )
                    _run_fieldconvert(
                        fieldconvert, xml_path, mode_chk, mode_pts_path, mode_out
                    )
                    arr = _read_interppoints_file(mode_out)
                    uc, vc = _guess_velocity_columns(arr.shape[1], dim=2)
                    mode_u_grid[m, :, :, k] = arr[:, uc].reshape(args.ny, args.nx)
                    mode_v_grid[m, :, :, k] = arr[:, vc].reshape(args.ny, args.nx)
                print(f"[MODE GRID {k + 1:4d}/{nd:4d}] interpolated all modes")
        finally:
            if args.keep_temp:
                print(f"Mode interpolation temporary files kept in: {mode_interp_root}")
            else:
                shutil.rmtree(mode_interp_root, ignore_errors=True)

    meta = {
        "case_dir_inferred": str(case_dir),
        "chk_dir": str(chk_dir),
        "xml": str(xml_path),
        "xml_sha256": _file_sha256(xml_path),
        "fieldconvert": str(fieldconvert),
        "fieldconvert_version": (
            _try_run_text([str(fieldconvert), "--version"])
            or _try_run_text([str(fieldconvert), "-V"])
        ),
        "chk_pattern": args.chk_pattern,
        "do_archive_file": str(do_archive_file),
        "do_archive_found": do_archive_file.is_file(),
        "do_archive_parts_found": len(archive_part_files),
        "do_source": do_source,
        "do_dat_dir": str(do_dat_dir),
        "do_dat_pattern": args.do_dat_pattern,
        "nx": args.nx,
        "ny": args.ny,
        "xmin": args.xmin,
        "xmax": args.xmax,
        "ymin": args.ymin,
        "ymax": args.ymax,
        "u_col": inferred_u_col,
        "v_col": inferred_v_col,
        "p_col": inferred_p_col,
        "dt_override": args.dt,
        "chk_stride_override": args.chk_stride,
        "viz_safe_arrays": [
            "u",
            "v",
            "p",
            "x",
            "y",
            "t",
            "mode_u",
            "mode_v",
            "yi",
            "t_do",
            "common_idx_chk",
            "common_idx_do",
            "archive_diag_cov",
            "archive_mode_norms",
            "archive_ortho_err",
        ],
        "quantitative_safe_arrays": [
            "yi",
            "ou_state",
            "t_do",
            "do_rng_state",
            "archive_mean_coeffs_u",
            "archive_mean_coeffs_v",
            "archive_mean_coeffs_p",
            "archive_mode_coeffs_u",
            "archive_mode_coeffs_v",
            "archive_diag_cov",
            "archive_mode_norms",
            "archive_ortho_err",
        ],
        "native_mode_note": (
            "mode_u_native/mode_v_native use Nektar internal physical-point ordering "
            "and are not self-describing coordinates."
        ),
    }
    meta.update(_git_provenance(Path(__file__).resolve().parents[1]))

    # Save time-last arrays (Ny, Nx, T) as requested, plus native DO state arrays.
    save_kwargs = dict(
        u=np.transpose(u_stack, (1, 2, 0)),
        v=np.transpose(v_stack, (1, 2, 0)),
        p=np.transpose(p_stack, (1, 2, 0)),
        x=xg,
        y=yg,
        t=t,
        chk_files=np.asarray([p.name for p in chk_files], dtype=object),
        metadata_json=json.dumps(meta, indent=2),
    )
    if do_source != "none" and do_dims is not None:
        save_kwargs.update(
            mode_u_native=mode_u_native_t,
            mode_v_native=mode_v_native_t,
            yi=yi_t,
            ou_state=ou_state_t,
            do_rng_state=rng_state_t,
            t_do=t_do,
            do_dims=do_dims,
        )
        if do_source == "archive":
            save_kwargs["do_archive_file"] = np.asarray(do_archive_file.name, dtype=object)
        elif do_source == "archive_parts":
            save_kwargs["do_archive_part_files"] = np.asarray(
                [p.name for p in archive_part_files], dtype=object
            )
        elif do_source == "legacy_dat":
            save_kwargs["do_dat_files"] = np.asarray(
                [p.name for p in do_dat_files], dtype=object
            )

        if archive_mean_coeffs_u_t is not None:
            save_kwargs["archive_mean_coeffs_u"] = archive_mean_coeffs_u_t
        if archive_mean_coeffs_v_t is not None:
            save_kwargs["archive_mean_coeffs_v"] = archive_mean_coeffs_v_t
        if archive_mean_coeffs_p_t is not None:
            save_kwargs["archive_mean_coeffs_p"] = archive_mean_coeffs_p_t
        if archive_mode_coeffs_u_t is not None:
            save_kwargs["archive_mode_coeffs_u"] = archive_mode_coeffs_u_t
        if archive_mode_coeffs_v_t is not None:
            save_kwargs["archive_mode_coeffs_v"] = archive_mode_coeffs_v_t
        if archive_diag_cov_t is not None:
            save_kwargs["archive_diag_cov"] = archive_diag_cov_t
        if archive_mode_norms_t is not None:
            save_kwargs["archive_mode_norms"] = archive_mode_norms_t
        if archive_ortho_err_t is not None:
            save_kwargs["archive_ortho_err"] = archive_ortho_err_t
        if mode_u_grid is not None:
            save_kwargs["mode_u"] = mode_u_grid
        if mode_v_grid is not None:
            save_kwargs["mode_v"] = mode_v_grid

        # Optional alignment by time value if both time arrays are available.
        # Tolerance: half the median step in t (or t_do, whichever is smaller),
        # which absorbs accumulated floating-point drift between the two filters
        # over O(1e5) steps without admitting genuinely different timestamps.
        if t.size and t_do.size:
            def _half_step(arr: np.ndarray) -> float:
                if arr.size < 2:
                    return 1e-9
                d = np.diff(arr)
                d = d[d > 0]
                return float(np.median(d) / 2.0) if d.size else 1e-9
            tol = max(min(_half_step(t), _half_step(t_do)), 1e-12)
            common_idx_chk = []
            common_idx_do = []
            for i, tc in enumerate(t):
                j = np.where(np.abs(t_do - tc) <= tol)[0]
                if j.size:
                    common_idx_chk.append(i)
                    common_idx_do.append(int(j[0]))
            save_kwargs["common_idx_chk"] = np.asarray(common_idx_chk, dtype=np.int64)
            save_kwargs["common_idx_do"] = np.asarray(common_idx_do, dtype=np.int64)

    _save_arrays(out, save_kwargs)

    print(f"\nSaved: {out}")
    print(f"u/v/p shapes (Ny,Nx,T): {save_kwargs['u'].shape}, {save_kwargs['v'].shape}, {save_kwargs['p'].shape}")
    print(f"x range: [{args.xmin}, {args.xmax}], y range: [{args.ymin}, {args.ymax}]")
    print(f"time range: [{t.min():.6g}, {t.max():.6g}]")
    if do_source != "none" and do_dims is not None:
        print(
            f"mode_u_native shape (S,nPts,Tdo): {mode_u_native_t.shape}; "
            f"yi shape (Np,S,Tdo): {yi_t.shape}"
        )
        if t_do.size:
            print(f"DO time range: [{t_do.min():.6g}, {t_do.max():.6g}]")
        print(f"DO source: {do_source}")
        if mode_u_grid is not None:
            print(
                f"mode_u/mode_v grid shapes (S,Ny,Nx,Tdo): "
                f"{mode_u_grid.shape}, {mode_v_grid.shape}"
            )
        if mode_u_grid is None:
            print(
                "Note: DO modes are saved on native Nektar physical-point ordering "
                "(not Cartesian grid-interpolated)."
            )
        else:
            print(
                "Note: native-order mode arrays are also kept "
                "(mode_u_native/mode_v_native) alongside interpolated mode_u/mode_v."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
