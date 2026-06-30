#!/usr/bin/env python3
"""Plot a 2D Nektar++ geometry mesh from geometry.xml and highlight elements.

Supports compressed geometry sections (B64Z-LittleEndian) used by Nektar++.
"""

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


def _parse_id_list(expr: str) -> set[int]:
    ids: set[int] = set()
    for token in expr.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a_str, b_str = token.split("-", 1)
            a, b = int(a_str), int(b_str)
            lo, hi = (a, b) if a <= b else (b, a)
            ids.update(range(lo, hi + 1))
        else:
            ids.add(int(token))
    return ids


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

    out = {}
    for v in vertex_node.findall("V"):
        vid = int(v.attrib["ID"])
        xyz = (v.text or "").split()
        out[vid] = (float(xyz[0]), float(xyz[1]))
    return out


def _read_triangles(elem_node: ET.Element) -> dict[int, tuple[int, int, int]]:
    # Preferred path for geometry.xml from NekMesh: compressed <T> blob.
    t_blob = elem_node.find("T")
    if t_blob is not None and "COMPRESSED" in t_blob.attrib:
        bitsize = int(t_blob.attrib.get("BITSIZE", "64"))
        if bitsize != 64:
            raise ValueError(f"Unsupported ELEMENT/T BITSIZE={bitsize}; expected 64.")
        data = _decode_b64z_text(t_blob.text or "")
        rec = struct.Struct("<QQQQ")
        if len(data) % rec.size != 0:
            raise ValueError("Invalid compressed ELEMENT/T payload length.")
        out: dict[int, tuple[int, int, int]] = {}
        for i in range(0, len(data), rec.size):
            eid, e0, e1, e2 = rec.unpack_from(data, i)
            out[int(eid)] = (int(e0), int(e1), int(e2))
        return out

    # Fallback for uncompressed XML with many <T ID="...">v0 v1 v2</T> lines.
    out = {}
    for t in elem_node.findall("T"):
        if "ID" not in t.attrib:
            continue
        eid = int(t.attrib["ID"])
        edge_ids = [int(x) for x in re.split(r"\s+", (t.text or "").strip()) if x]
        if len(edge_ids) != 3:
            continue
        out[eid] = (edge_ids[0], edge_ids[1], edge_ids[2])
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

    out = {}
    for e in edge_node.findall("E"):
        eid = int(e.attrib["ID"])
        vv = [int(x) for x in re.split(r"\s+", (e.text or "").strip()) if x]
        if len(vv) != 2:
            continue
        out[eid] = (vv[0], vv[1])
    return out


def _triangle_vertex_ids_from_edges(
    edge_ids: tuple[int, int, int], edges: dict[int, tuple[int, int]]
) -> tuple[int, int, int] | None:
    verts: list[int] = []
    for edge_id in edge_ids:
        if edge_id not in edges:
            return None
        v0, v1 = edges[edge_id]
        verts.append(v0)
        verts.append(v1)
    unique = sorted(set(verts))
    if len(unique) != 3:
        return None
    return (unique[0], unique[1], unique[2])


def _order_triangle(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    cx = sum(p[0] for p in points) / 3.0
    cy = sum(p[1] for p in points) / 3.0
    return sorted(points, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))


def plot_geometry(
    xml_path: Path,
    highlight_ids: set[int],
    annotate: bool,
    line_width: float,
    alpha: float,
    save: Path | None,
) -> None:
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
    triangles = _read_triangles(elem_node)

    base_polys = []
    hi_polys = []
    hi_centroids = []

    skipped = 0
    for eid, edge_ids in triangles.items():
        tri_vids = _triangle_vertex_ids_from_edges(edge_ids, edges)
        if tri_vids is None:
            skipped += 1
            continue

        try:
            poly = [vertices[tri_vids[0]], vertices[tri_vids[1]], vertices[tri_vids[2]]]
        except KeyError:
            skipped += 1
            continue
        poly = _order_triangle(poly)

        if eid in highlight_ids:
            hi_polys.append(poly)
            cx = (poly[0][0] + poly[1][0] + poly[2][0]) / 3.0
            cy = (poly[0][1] + poly[1][1] + poly[2][1]) / 3.0
            hi_centroids.append((eid, cx, cy))
        else:
            base_polys.append(poly)

    fig, ax = plt.subplots(figsize=(9, 6))

    base = PolyCollection(
        base_polys,
        facecolors="none",
        edgecolors="0.45",
        linewidths=line_width,
        alpha=alpha,
    )
    ax.add_collection(base)

    if hi_polys:
        hi = PolyCollection(
            hi_polys,
            facecolors="tab:orange",
            edgecolors="tab:red",
            linewidths=max(0.8, line_width * 1.8),
            alpha=0.9,
        )
        ax.add_collection(hi)

        if annotate:
            for eid, cx, cy in hi_centroids:
                ax.text(cx, cy, str(eid), ha="center", va="center", fontsize=8, color="black")

    ax.set_aspect("equal", adjustable="box")
    ax.autoscale_view()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        f"Nektar++ mesh from {xml_path.name} | elements: {len(triangles)}"
        + (f" | highlighted: {len(hi_polys)}" if highlight_ids else "")
    )

    if skipped:
        print(
            f"Warning: skipped {skipped} triangle(s) due to invalid edge/vertex references.",
            file=sys.stderr,
        )

    if save is not None:
        fig.savefig(save, dpi=200, bbox_inches="tight")
        print(f"Saved plot to {save}")
    else:
        plt.show()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xml",
        type=Path,
        default=Path("mesh/geometry.xml"),
        help="Path to Nektar++ geometry XML (default: mesh/geometry.xml)",
    )
    parser.add_argument(
        "--highlight",
        type=str,
        default="",
        help="Element IDs/ranges to highlight, e.g. '3,7,10-14'.",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Write highlighted element IDs at element centroids.",
    )
    parser.add_argument(
        "--line-width",
        type=float,
        default=0.35,
        help="Base mesh edge line width.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.97,
        help="Base mesh edge alpha.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="If set, save figure to this path instead of opening a window.",
    )

    args = parser.parse_args()
    highlight_ids = _parse_id_list(args.highlight) if args.highlight else set()

    try:
        plot_geometry(
            xml_path=args.xml,
            highlight_ids=highlight_ids,
            annotate=args.annotate,
            line_width=args.line_width,
            alpha=args.alpha,
            save=args.save,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
