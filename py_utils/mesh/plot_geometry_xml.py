#!/usr/bin/env python3
"""Plot a 2D Nektar++ geometry mesh from a geometry XML file."""

from __future__ import annotations

import argparse
import base64
import math
import re
import struct
import sys
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


def _decode_b64z_text(text: str) -> bytes:
    compact = "".join(text.split())
    return zlib.decompress(base64.b64decode(compact))


def _read_vertices(vertex_node: ET.Element) -> dict[int, tuple[float, float]]:
    compressed = vertex_node.attrib.get("COMPRESSED", "").startswith("B64Z")
    if compressed:
        bitsize = int(vertex_node.attrib.get("BITSIZE", "64"))
        if bitsize != 64:
            raise ValueError(f"Unsupported VERTEX BITSIZE={bitsize}; expected 64.")
        data = _decode_b64z_text(vertex_node.text or "")
        rec = struct.Struct("<Qddd")
        if len(data) % rec.size != 0:
            raise ValueError("Invalid compressed VERTEX payload length.")
        out: dict[int, tuple[float, float]] = {}
        for i in range(0, len(data), rec.size):
            vid, x, y, _z = rec.unpack_from(data, i)
            out[int(vid)] = (x, y)
        return out

    out: dict[int, tuple[float, float]] = {}
    for v in vertex_node.findall("V"):
        vid = int(v.attrib["ID"])
        xyz = (v.text or "").split()
        out[vid] = (float(xyz[0]), float(xyz[1]))
    return out


def _read_edges(edge_node: ET.Element) -> dict[int, tuple[int, int]]:
    compressed = edge_node.attrib.get("COMPRESSED", "").startswith("B64Z")
    if compressed:
        bitsize = int(edge_node.attrib.get("BITSIZE", "64"))
        if bitsize != 64:
            raise ValueError(f"Unsupported EDGE BITSIZE={bitsize}; expected 64.")
        data = _decode_b64z_text(edge_node.text or "")
        rec = struct.Struct("<QQQ")
        if len(data) % rec.size != 0:
            raise ValueError("Invalid compressed EDGE payload length.")
        out: dict[int, tuple[int, int]] = {}
        for i in range(0, len(data), rec.size):
            eid, v0, v1 = rec.unpack_from(data, i)
            out[int(eid)] = (int(v0), int(v1))
        return out

    out: dict[int, tuple[int, int]] = {}
    for e in edge_node.findall("E"):
        eid = int(e.attrib["ID"])
        vv = [int(x) for x in re.split(r"\s+", (e.text or "").strip()) if x]
        if len(vv) == 2:
            out[eid] = (vv[0], vv[1])
    return out


def _read_element_edges(elem_node: ET.Element, tag: str, n_edges: int) -> dict[int, tuple[int, ...]]:
    out: dict[int, tuple[int, ...]] = {}

    blob = elem_node.find(tag)
    if blob is not None and "COMPRESSED" in blob.attrib:
        bitsize = int(blob.attrib.get("BITSIZE", "64"))
        if bitsize != 64:
            raise ValueError(f"Unsupported ELEMENT/{tag} BITSIZE={bitsize}; expected 64.")
        data = _decode_b64z_text(blob.text or "")
        rec = struct.Struct("<" + "Q" * (n_edges + 1))
        if len(data) % rec.size != 0:
            raise ValueError(f"Invalid compressed ELEMENT/{tag} payload length.")
        for i in range(0, len(data), rec.size):
            vals = rec.unpack_from(data, i)
            out[int(vals[0])] = tuple(int(v) for v in vals[1:])
        return out

    for t in elem_node.findall(tag):
        if "ID" not in t.attrib:
            continue
        eid = int(t.attrib["ID"])
        edge_ids = [int(x) for x in re.split(r"\s+", (t.text or "").strip()) if x]
        if len(edge_ids) == n_edges:
            out[eid] = tuple(edge_ids)
    return out


def _order_polygon(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    return sorted(points, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))


def _polygon_from_edges(edge_ids: tuple[int, ...], edges: dict[int, tuple[int, int]]) -> list[int] | None:
    vids: list[int] = []
    for e in edge_ids:
        if e not in edges:
            return None
        v0, v1 = edges[e]
        vids.extend([v0, v1])
    unique = sorted(set(vids))
    if len(unique) != len(edge_ids):
        return None
    return unique


def plot_geometry(xml_path: Path, line_width: float, alpha: float, save: Path | None) -> None:
    root = ET.parse(xml_path).getroot()
    geom = root.find("GEOMETRY")
    if geom is None:
        raise ValueError("Could not find <GEOMETRY> node.")

    vertices_node = geom.find("VERTEX")
    edge_node = geom.find("EDGE")
    elem_node = geom.find("ELEMENT")
    if vertices_node is None or edge_node is None or elem_node is None:
        raise ValueError("Missing <VERTEX>, <EDGE> or <ELEMENT> section.")

    vertices = _read_vertices(vertices_node)
    edges = _read_edges(edge_node)
    tris = _read_element_edges(elem_node, "T", 3)
    quads = _read_element_edges(elem_node, "Q", 4)

    polys: list[list[tuple[float, float]]] = []
    skipped = 0

    for edge_ids in list(tris.values()) + list(quads.values()):
        vids = _polygon_from_edges(edge_ids, edges)
        if vids is None:
            skipped += 1
            continue
        try:
            poly = [vertices[v] for v in vids]
        except KeyError:
            skipped += 1
            continue
        polys.append(_order_polygon(poly))

    fig, ax = plt.subplots(figsize=(9, 6))
    mesh = PolyCollection(
        polys,
        facecolors="none",
        edgecolors="0.3",
        linewidths=line_width,
        alpha=alpha,
    )
    ax.add_collection(mesh)
    ax.set_aspect("equal", adjustable="box")
    ax.autoscale_view()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        f"Nektar++ mesh: {xml_path.name} | elements: {len(polys)}"
        + (f" | skipped: {skipped}" if skipped else "")
    )

    if save is not None:
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=200, bbox_inches="tight")
        print(f"Saved plot to {save}")
    else:
        plt.show()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xml",
        type=Path,
        required=True,
        help="Path to Nektar++ geometry XML (e.g. mesh/geometry.xml).",
    )
    parser.add_argument("--line-width", type=float, default=0.45)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--save", type=Path, default=None, help="Optional output image path.")
    args = parser.parse_args()

    if not args.xml.is_file():
        raise FileNotFoundError(f"XML file not found: {args.xml}")
    if args.line_width <= 0:
        raise ValueError("--line-width must be > 0")
    if not (0.0 < args.alpha <= 1.0):
        raise ValueError("--alpha must be in (0, 1]")

    plot_geometry(args.xml, line_width=args.line_width, alpha=args.alpha, save=args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
