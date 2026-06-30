"""Read/write a DOVelocityCorrectionScheme NEKTAR XML, preserving structure and comments."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from lxml import etree


_PARAM_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*(.+?)\s*$", re.DOTALL)


@dataclass
class ForcingChannel:
    var: str
    value: str


@dataclass
class CheckpointFilter:
    output_file: str = "output/casefile"
    output_frequency: int = 500


@dataclass
class DOVelocityCorrectionSchemeCase:
    """In-memory view of editable DOVelocityCorrectionScheme fields."""

    params: dict[str, str] = field(default_factory=dict)
    solver_type: str = "DOVelocityCorrectionScheme"
    forcing_channels: list[ForcingChannel] = field(default_factory=list)
    checkpoint: CheckpointFilter | None = None
    do_archive: CheckpointFilter | None = None


def _find_parameters_block(root: etree._Element) -> etree._Element | None:
    return root.find(".//CONDITIONS/PARAMETERS")


def _iter_param_elements(parameters: etree._Element):
    """Yield <P> elements that are not commented-out and have a NAME=VALUE body."""
    for el in parameters.findall("P"):
        if el.text is None:
            continue
        m = _PARAM_RE.match(el.text)
        if not m:
            continue
        yield el, m.group(1), m.group(2)


def _set_param(parameters: etree._Element, name: str, value: str) -> None:
    for el, key, _ in _iter_param_elements(parameters):
        if key == name:
            el.text = f" {name} = {value} "
            return
    new = etree.SubElement(parameters, "P")
    new.text = f" {name} = {value} "
    new.tail = "\n      "


def _get_param(parameters: etree._Element, name: str) -> str | None:
    for _el, key, val in _iter_param_elements(parameters):
        if key == name:
            return val
    return None


def _filter_to_obj(filt: etree._Element) -> CheckpointFilter:
    obj = CheckpointFilter()
    for p in filt.findall("PARAM"):
        nm = p.get("NAME") or ""
        if nm == "OutputFile" and p.text:
            obj.output_file = p.text.strip()
        elif nm == "OutputFrequency" and p.text:
            try:
                obj.output_frequency = int(p.text.strip())
            except ValueError:
                pass
    return obj


def _write_filter(parent: etree._Element, ftype: str, obj: CheckpointFilter) -> None:
    existing = None
    for f in parent.findall("FILTER"):
        if f.get("TYPE") == ftype:
            existing = f
            break
    if existing is None:
        existing = etree.SubElement(parent, "FILTER", attrib={"TYPE": ftype})
    for child in list(existing):
        if isinstance(child, etree._Element) and child.tag == "PARAM":
            existing.remove(child)
    pf = etree.SubElement(existing, "PARAM", attrib={"NAME": "OutputFile"})
    pf.text = obj.output_file
    pf.tail = "\n      "
    pq = etree.SubElement(existing, "PARAM", attrib={"NAME": "OutputFrequency"})
    pq.text = str(obj.output_frequency)
    pq.tail = "\n    "


def parse_xml(path: Path) -> tuple[etree._ElementTree, DOVelocityCorrectionSchemeCase]:
    parser = etree.XMLParser(remove_blank_text=False, remove_comments=False)
    tree = etree.parse(str(path), parser)
    root = tree.getroot()
    case = DOVelocityCorrectionSchemeCase()

    params = _find_parameters_block(root)
    if params is not None:
        for _el, key, val in _iter_param_elements(params):
            case.params[key] = val

    si = root.find(".//CONDITIONS/SOLVERINFO")
    if si is not None:
        for i in si.findall("I"):
            if i.get("PROPERTY") == "SolverType":
                case.solver_type = i.get("VALUE") or case.solver_type

    fc = root.find(".//CONDITIONS/FUNCTION[@NAME='ForcingChannels']")
    if fc is not None:
        for e in fc.findall("E"):
            v = e.get("VAR") or ""
            val = e.get("VALUE") or ""
            case.forcing_channels.append(ForcingChannel(v, val))

    filters = root.find("FILTERS")
    if filters is not None:
        for f in filters.findall("FILTER"):
            t = f.get("TYPE")
            if t == "Checkpoint":
                case.checkpoint = _filter_to_obj(f)
            elif t == "DOArchive":
                case.do_archive = _filter_to_obj(f)

    return tree, case


def write_xml(tree: etree._ElementTree, case: DOVelocityCorrectionSchemeCase, out_path: Path) -> None:
    root = tree.getroot()

    params = _find_parameters_block(root)
    if params is None:
        cond = root.find("CONDITIONS")
        if cond is None:
            cond = etree.SubElement(root, "CONDITIONS")
        params = etree.SubElement(cond, "PARAMETERS")
    for k, v in case.params.items():
        _set_param(params, k, v)

    si = root.find(".//CONDITIONS/SOLVERINFO")
    if si is not None:
        found = False
        for i in si.findall("I"):
            if i.get("PROPERTY") == "SolverType":
                i.set("VALUE", case.solver_type)
                found = True
        if not found:
            etree.SubElement(si, "I", attrib={"PROPERTY": "SolverType", "VALUE": case.solver_type})

    if case.forcing_channels:
        fc = root.find(".//CONDITIONS/FUNCTION[@NAME='ForcingChannels']")
        cond = root.find("CONDITIONS")
        if fc is None and cond is not None:
            fc = etree.SubElement(cond, "FUNCTION", attrib={"NAME": "ForcingChannels"})
        if fc is not None:
            for e in list(fc.findall("E")):
                fc.remove(e)
            for ch in case.forcing_channels:
                etree.SubElement(fc, "E", attrib={"VAR": ch.var, "VALUE": ch.value})

    filters = root.find("FILTERS")
    if filters is None:
        filters = etree.SubElement(root, "FILTERS")
    if case.checkpoint is not None:
        _write_filter(filters, "Checkpoint", case.checkpoint)
    if case.do_archive is not None:
        _write_filter(filters, "DOArchive", case.do_archive)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(
        str(out_path),
        encoding="utf-8",
        xml_declaration=True,
        pretty_print=False,
    )


def load_or_default(path: Path | None) -> tuple[etree._ElementTree | None, DOVelocityCorrectionSchemeCase]:
    if path and path.exists():
        return parse_xml(path)
    return None, DOVelocityCorrectionSchemeCase()
