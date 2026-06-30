#!/usr/bin/env python3
"""Compute cylinder Cd/Cl directly from Nektar++ .chk files.

Method:
- For each checkpoint, build a temporary session XML that loads that .chk as
  InitialConditions and adds an AeroForces filter on the cylinder boundary.
- Run IncNavierStokesSolver in zero-step mode (Time == FinTime), which still
  evaluates filters at initialise.
- Read Fx_total/Fy_total from the generated .fce and convert to Cd/Cl.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from xml.etree import ElementTree as ET


def _first_float(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def _parse_parameter_map(root: ET.Element) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in root.findall(".//CONDITIONS/PARAMETERS/P"):
        txt = " ".join((p.text or "").split())
        if "=" not in txt:
            continue
        name, val = txt.split("=", 1)
        out[name.strip()] = val.strip()
    return out


def _set_or_add_parameter(root: ET.Element, name: str, value_expr: str) -> None:
    params = root.find(".//CONDITIONS/PARAMETERS")
    if params is None:
        cond = root.find(".//CONDITIONS")
        if cond is None:
            raise ValueError("Session XML has no <CONDITIONS> block.")
        params = ET.SubElement(cond, "PARAMETERS")

    for p in params.findall("P"):
        txt = " ".join((p.text or "").split())
        if "=" in txt and txt.split("=", 1)[0].strip() == name:
            p.text = f" {name} = {value_expr} "
            return

    p = ET.SubElement(params, "P")
    p.text = f" {name} = {value_expr} "


def _guess_uinf(root: ET.Element) -> float | None:
    for reg in root.findall(".//BOUNDARYCONDITIONS/REGION"):
        if reg.attrib.get("REF") == "0":
            for d in reg.findall("D"):
                if d.attrib.get("VAR") == "u":
                    val = _first_float(d.attrib.get("VALUE", ""))
                    if val is not None:
                        return val
    for d in root.findall(".//BOUNDARYCONDITIONS/REGION/D"):
        if d.attrib.get("VAR") == "u":
            val = _first_float(d.attrib.get("VALUE", ""))
            if val is not None:
                return val
    return None


def _guess_rho(root: ET.Element) -> float | None:
    pmap = _parse_parameter_map(root)
    if "rho" in pmap:
        return _first_float(pmap["rho"])
    return None


def _guess_diameter_from_geo(case_dir: Path) -> float | None:
    geo = case_dir / "mesh" / "mesh.geo"
    if not geo.is_file():
        return None
    for line in geo.read_text().splitlines():
        m = re.search(
            r"Disk\([^)]*\)\s*=\s*\{[^}]*,\s*([0-9eE+\-.]+)\s*,\s*([0-9eE+\-.]+)\s*\}\s*;",
            line,
        )
        if not m:
            continue
        rx = float(m.group(1))
        ry = float(m.group(2))
        if abs(rx - ry) > 1.0e-12:
            raise ValueError("Detected non-circular Disk in mesh.geo; pass --diameter explicitly.")
        return 2.0 * rx
    return None


def _find_chk_files(output_dirs: list[Path], pattern: str) -> list[Path]:
    files: list[Path] = []
    for out_dir in output_dirs:
        files.extend(sorted(out_dir.glob(pattern)))
    if not files:
        return []

    def key(p: Path) -> tuple[int, str]:
        m = re.search(r"_(\d+)\.chk$", p.name)
        return (int(m.group(1)) if m else -1, str(p))

    return sorted(files, key=key)


def _infer_checkpoint_stride(root: ET.Element, chk_prefix: str) -> int | None:
    filters = root.findall(".//FILTERS/FILTER")
    for flt in filters:
        if flt.attrib.get("TYPE") != "Checkpoint":
            continue
        ofile = None
        ofreq = None
        for prm in flt.findall("PARAM"):
            name = prm.attrib.get("NAME", "")
            txt = " ".join((prm.text or "").split())
            if name == "OutputFile":
                ofile = Path(txt).name
            elif name == "OutputFrequency":
                try:
                    ofreq = int(float(txt))
                except ValueError:
                    ofreq = None
        if ofile == chk_prefix and ofreq is not None:
            return ofreq
    return None


def _replace_initial_conditions(root: ET.Element, chk_abs: Path) -> None:
    cond = root.find(".//CONDITIONS")
    if cond is None:
        raise ValueError("Session XML has no <CONDITIONS> block.")

    for fn in cond.findall("FUNCTION"):
        if fn.attrib.get("NAME") == "InitialConditions":
            cond.remove(fn)

    fn = ET.SubElement(cond, "FUNCTION", {"NAME": "InitialConditions"})
    ET.SubElement(fn, "F", {"VAR": "u,v,p", "FILE": str(chk_abs)})


def _replace_filters(root: ET.Element, fce_prefix: str, boundary_id: int) -> None:
    existing = root.find(".//FILTERS")
    if existing is not None:
        parent = root
        for node in root.iter():
            if existing in list(node):
                parent = node
                break
        parent.remove(existing)

    filters = ET.SubElement(root, "FILTERS")
    af = ET.SubElement(filters, "FILTER", {"TYPE": "AeroForces"})

    p1 = ET.SubElement(af, "PARAM", {"NAME": "OutputFile"})
    p1.text = f" {fce_prefix} "
    p2 = ET.SubElement(af, "PARAM", {"NAME": "OutputFrequency"})
    p2.text = " 1 "
    p3 = ET.SubElement(af, "PARAM", {"NAME": "Boundary"})
    p3.text = f" B[{boundary_id}] "


def _parse_one_fce(fce_path: Path) -> tuple[float, float, float]:
    last: list[float] | None = None
    for raw in fce_path.read_text().splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 7:
            continue
        try:
            vals = [float(x) for x in parts]
        except ValueError:
            continue
        last = vals

    if last is None:
        raise ValueError(f"No numeric data in force file: {fce_path}")

    t = last[0]
    fx_tot = last[3]
    fy_tot = last[6]
    return t, fx_tot, fy_tot


def _build_solver_cmd(solver: Path, geom: Path | None, session_xml: Path) -> list[str]:
    cmd = [str(solver)]
    if geom is not None:
        cmd.append(str(geom))
    cmd.append(str(session_xml))
    return cmd


def _resolve_output_dirs(case_dir: Path, output_dir_args: list[Path] | None) -> list[Path]:
    if not output_dir_args:
        default_dir = (case_dir / "output").resolve()
        if not default_dir.is_dir():
            raise FileNotFoundError(
                "No output directory specified and default directory not found: "
                f"{default_dir}. Pass one or more --output-dir values."
            )
        return [default_dir]

    resolved: list[Path] = []
    for raw in output_dir_args:
        candidate = raw.resolve() if raw.is_absolute() else (case_dir / raw).resolve()
        if not candidate.is_dir():
            raise FileNotFoundError(f"Output directory not found: {candidate}")
        if candidate not in resolved:
            resolved.append(candidate)
    return resolved


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case-dir", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--session", type=Path, default=None, help="Base session XML")
    ap.add_argument("--geometry", type=Path, default=None, help="Geometry XML (if separate)")
    ap.add_argument(
        "--output-dir",
        type=Path,
        action="append",
        default=None,
        help=(
            "Directory containing checkpoint outputs. "
            "Pass multiple times to combine checkpoints from several folders."
        ),
    )
    ap.add_argument("--chk-pattern", default="*.chk", help="Glob matched in each --output-dir")
    ap.add_argument("--boundary", type=int, default=3, help="Cylinder boundary ID (B[id])")
    ap.add_argument("--solver", type=Path, default=Path("/home/isma/nektar++/build/dist/bin/IncNavierStokesSolver"))
    ap.add_argument("--rho", type=float, default=None)
    ap.add_argument("--uinf", type=float, default=None)
    ap.add_argument("--diameter", type=float, default=None)
    ap.add_argument("--ref-area", type=float, default=None)
    ap.add_argument("--chk-stride", type=int, default=None, help="Steps between checkpoints")
    ap.add_argument("--csv-out", type=Path, default=None)
    ap.add_argument("--keep-temp", action="store_true")
    args = ap.parse_args()

    case_dir = args.case_dir.resolve()
    session = (args.session or (case_dir / "casefile.xml")).resolve()
    if not session.is_file():
        raise FileNotFoundError(f"Session XML not found: {session}")

    if not args.solver.is_file():
        raise FileNotFoundError(f"IncNavierStokesSolver not found: {args.solver}")

    tree = ET.parse(session)
    root = tree.getroot()

    geom = args.geometry.resolve() if args.geometry else None
    if geom is None and root.find("GEOMETRY") is None:
        candidate = case_dir / "mesh" / "geometry.xml"
        if candidate.is_file():
            geom = candidate.resolve()

    output_dirs = _resolve_output_dirs(case_dir, args.output_dir)
    chk_files = _find_chk_files(output_dirs, args.chk_pattern)
    if not chk_files:
        joined = ", ".join(str(p) for p in output_dirs)
        print(f"No checkpoint files found in: {joined}", file=sys.stderr)
        return 2

    pmap = _parse_parameter_map(root)
    dt = _first_float(pmap.get("TimeStep", ""))

    prefix = chk_files[0].name.split("_")[0]
    chk_stride = args.chk_stride
    if chk_stride is None:
        chk_stride = _infer_checkpoint_stride(root, prefix)
    if chk_stride is None:
        chk_stride = 1

    uinf = args.uinf if args.uinf is not None else _guess_uinf(root)
    if uinf is None:
        uinf = 1.0

    rho = args.rho if args.rho is not None else _guess_rho(root)
    if rho is None:
        rho = 1.0

    diameter = args.diameter if args.diameter is not None else _guess_diameter_from_geo(case_dir)
    if diameter is None:
        diameter = 1.0

    ref_area = args.ref_area if args.ref_area is not None else diameter
    denom = 0.5 * rho * uinf * uinf * ref_area
    if abs(denom) < 1.0e-20:
        raise ValueError("Coefficient denominator is zero; check rho/uinf/ref-area.")

    rows: list[tuple[Path, float, float, float, float, float]] = []
    tmp_root = Path(tempfile.mkdtemp(prefix="nektar_chk_force_", dir="/tmp"))

    try:
        for chk in chk_files:
            m = re.search(r"_(\d+)\.chk$", chk.name)
            chk_idx = int(m.group(1)) if m else 0
            t_guess = float(chk_idx * chk_stride) * (dt if dt is not None else 1.0)

            local_tree = ET.parse(session)
            local_root = local_tree.getroot()

            _replace_initial_conditions(local_root, chk.resolve())
            _set_or_add_parameter(local_root, "Time", f"{t_guess:.16g}")
            _set_or_add_parameter(local_root, "FinTime", f"{t_guess:.16g}")

            fce_prefix = str((tmp_root / chk.stem).resolve())
            _replace_filters(local_root, fce_prefix=fce_prefix, boundary_id=args.boundary)

            tmp_xml = tmp_root / f"post_{chk.stem}.xml"
            local_tree.write(tmp_xml, encoding="utf-8", xml_declaration=True)

            cmd = _build_solver_cmd(args.solver.resolve(), geom, tmp_xml)
            proc = subprocess.run(
                cmd,
                cwd=case_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"Solver failed for {chk.name}\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"Output:\n{proc.stdout}"
                )

            fce_path = Path(f"{fce_prefix}.fce")
            if not fce_path.is_file():
                raise FileNotFoundError(f"Expected force file not found: {fce_path}")

            t_fce, fx, fy = _parse_one_fce(fce_path)
            cd = fx / denom
            cl = fy / denom
            rows.append((chk, t_fce, fx, fy, cd, cl))

    finally:
        if args.keep_temp:
            print(f"Temporary files kept at: {tmp_root}")
        else:
            shutil.rmtree(tmp_root, ignore_errors=True)

    if not rows:
        print("No Cl/Cd values computed.", file=sys.stderr)
        return 3

    if args.csv_out is not None:
        csv_out = args.csv_out
    elif len(output_dirs) == 1:
        csv_out = output_dirs[0] / "cl_cd_from_chk.csv"
    else:
        csv_out = case_dir / "cl_cd_from_chk.csv"
    csv_out = csv_out.resolve()
    csv_out.parent.mkdir(parents=True, exist_ok=True)

    with csv_out.open("w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["chk_file", "time", "Fx_total", "Fy_total", "Cd", "Cl"])
        for chk, t, fx, fy, cd, cl in rows:
            chk_label = str(chk.relative_to(case_dir)) if chk.is_relative_to(case_dir) else str(chk)
            w.writerow([chk_label, f"{t:.12g}", f"{fx:.12g}", f"{fy:.12g}", f"{cd:.12g}", f"{cl:.12g}"])

    cds = [r[4] for r in rows]
    cls = [r[5] for r in rows]
    print(f"Processed checkpoints: {len(rows)}")
    print(f"Output CSV: {csv_out}")
    print(f"Reference values: rho={rho:.6g}, Uinf={uinf:.6g}, D={diameter:.6g}, Aref={ref_area:.6g}")
    print(f"Cd_mean={sum(cds)/len(cds):.8f}, Cd_min={min(cds):.8f}, Cd_max={max(cds):.8f}")
    print(f"Cl_mean={sum(cls)/len(cls):.8f}, Cl_min={min(cls):.8f}, Cl_max={max(cls):.8f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
