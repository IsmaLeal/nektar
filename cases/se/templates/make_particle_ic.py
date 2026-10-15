#!/usr/bin/env python3
"""Write the velocity/pressure field of one DO particle as a Nektar field file,
to start a DNS from exactly the state that particle starts from.

The particle field is  mean + sum_k Y[p, k] mode_k  for u, v and p, built in
coefficient space from one DO archive (Format = Field, serial file or the
partitioned directory written by several ranks). The output is a single
serial .chk with fields u, v, p and the element blocks of the archive.

    python3 make_particle_ic.py ARCHIVE OUT.chk [--particle 0]

ARCHIVE is e.g. RUN/output/do/casefile.do_000000.fld (the step-0 archive).
"""

import argparse
import base64
import re
import sys
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

import numpy as np


def id_count(id_str):
    n = 0
    for part in id_str.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            n += int(b) - int(a) + 1
        elif part:
            n += 1
    return n


def coeffs_per_elem(shape, nmd):
    nm = [int(x) for x in nmd.replace("UNIORDER:", "").split(",")]
    if shape.lower().startswith("quad"):
        return nm[0] * nm[1]
    if shape.lower().startswith("tri"):
        p, q = nm
        m, M = min(p, q), max(p, q)
        return m * M - m * (m - 1) // 2
    raise ValueError(f"unsupported shape {shape}")


def decode(el):
    raw = base64.b64decode("".join((el.text or "").split()))
    if "B64Z" in el.attrib.get("COMPRESSED", ""):
        raw = zlib.decompress(raw)
    assert el.attrib.get("BITSIZE", "64") == "64"
    return np.frombuffer(raw, dtype="<f8")


def encode(arr, el):
    raw = np.ascontiguousarray(arr, dtype="<f8").tobytes()
    if "B64Z" in el.attrib.get("COMPRESSED", ""):
        raw = zlib.compress(raw)
    return base64.b64encode(raw).decode("ascii")


def meta_value(root, key):
    el = root.find(f"Metadata/DOVelocityCorrectionScheme_{key}")
    if el is None:
        sys.exit(f"archive metadata has no {key}")
    return el.text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--particle", type=int, default=0)
    a = ap.parse_args()

    if a.archive.is_dir():
        parts = sorted(a.archive.glob("P*.fld"))
    else:
        parts = [a.archive]
    roots = [ET.parse(p).getroot() for p in parts]

    S = int(meta_value(roots[0], "n_modes"))
    Np = int(meta_value(roots[0], "n_particles"))
    step = int(meta_value(roots[0], "step"))
    yi = np.frombuffer(bytes.fromhex(meta_value(roots[0], "Yi_hex")), dtype="<f8")
    assert yi.size == Np * S, (yi.size, Np, S)
    y = yi.reshape(Np, S)[a.particle]          # Yi is stored particle-major
    print(f"{a.archive}: step {step}, {S} modes, {Np} particles, "
          f"{len(parts)} file(s); particle {a.particle}: Y = {y}")

    out = ET.Element("NEKTAR")
    md = ET.SubElement(out, "Metadata")
    prov = roots[0].find("Metadata/Provenance")
    if prov is not None:
        md.append(prov)
    ET.SubElement(md, "Time").text = "0"
    nelem = 0
    for root in roots:
        for el in root.iter("ELEMENTS"):
            fields = [f.strip() for f in el.attrib["FIELDS"].split(",")]
            n = id_count(el.attrib["ID"]) * coeffs_per_elem(
                el.attrib["SHAPE"], el.attrib["NUMMODESPERDIR"])
            data = decode(el)
            assert data.size == n * len(fields), (data.size, n, len(fields))
            blk = {f: data[i * n:(i + 1) * n] for i, f in enumerate(fields)}
            new = []
            for var in ("u", "v", "p"):
                x = blk[var].copy()
                for k in range(S):
                    x += y[k] * blk[f"mode_{k}_{var}"]
                new.append(x)
            dst = ET.SubElement(out, "ELEMENTS", dict(el.attrib))
            dst.set("FIELDS", "u,v,p")
            dst.text = encode(np.concatenate(new), el)
            nelem += id_count(el.attrib["ID"])
    ET.ElementTree(out).write(a.out, encoding="utf-8", xml_declaration=True)
    print(f"wrote {a.out}: {nelem} elements, fields u,v,p")


if __name__ == "__main__":
    main()
