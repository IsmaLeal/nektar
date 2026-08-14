#!/usr/bin/env python3
"""Concatenate a DO run and its restart continuation into one output tree.

Generalized from the first use (sig=0.0144: stage1 2026_07_15_do_clean_
dnspod_sig0p0144_t250, t=0-250, stitched with stage2 2026_07_16_do_cont_
sig0p0144_to1000, restarted exactly from stage1's final DOArchive state --
verified by md5 match of restart_archive.fld against stage1's own final
archive) into a single continuous output tree, so downstream postprocessing
(transition counting, phase_space_video, do_panels rendering, etc.) can
treat it as one run.

For each output stream:
  - chks (Checkpoint filter, no embedded time): stage1's indices kept
    as-is; stage2's chk0 (duplicate of stage1's final chk, same restart
    instant) is dropped, remaining indices renamed by the chk-stride
    offset.
  - do archive (DOArchive filter, step/time embedded in Info.xml and each
    P*.fld): stage1's archives kept as-is; stage2's step-0 archive
    (duplicate, byte-identical to stage1's final archive) is dropped;
    the rest are copied with directory name, Info.xml and all P*.fld
    step/time metadata shifted by stage1's final step/time.
  - axis_v.his / energy.mdl (plain text, one row/block per output time):
    stage1 kept as-is; stage2's first block/row (t=0, duplicate of
    stage1's final time) is dropped, remaining rows time-shifted.

Verifies continuity (md5 of restart file vs stage1's final archive) before
touching anything, and asserts row/file counts at the end as a sanity
check.

Usage:
    python3 assemble_do_total_run.py STAGE1_DIR STAGE2_DIR OUT_DIR \
        --stage1-fintime 150 --chk-freq 2500 --archive-freq 100 \
        --timestep 0.01 --npts 401 --label "sig=0.0072, Re=100, S=6, Np=1000"
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


def reset(out: Path):
    if out.exists():
        shutil.rmtree(out)
    (out / "output" / "chks").mkdir(parents=True)
    (out / "output" / "do").mkdir(parents=True)
    (out / "output" / "asymmetry").mkdir(parents=True)
    (out / "output" / "energy").mkdir(parents=True)


def verify_seam(stage1: Path, stage2: Path, stage1_steps: int):
    final_archive = stage1 / f"output/do/casefile.do_{stage1_steps:06d}.fld"
    restart = stage2 / "restart_archive.fld"
    assert final_archive.exists(), f"missing {final_archive}"
    assert restart.exists(), f"missing {restart}"
    import hashlib
    h1 = hashlib.md5((final_archive / "P0000000.fld").read_bytes()).hexdigest()
    h2 = hashlib.md5((restart / "P0000000.fld").read_bytes()).hexdigest()
    assert h1 == h2, (
        f"seam mismatch: {final_archive} vs {restart} "
        f"(md5 {h1} != {h2}) -- stage2 does NOT restart from stage1's end"
    )
    print(f"seam verified: {final_archive.name} == restart_archive.fld (md5 {h1})")


def copy_static(stage1: Path, stage2: Path, out: Path, stage1_fintime: float,
                total_fintime: float, label: str):
    for f in ("geometry.xml", "geometry.opt", "ic.chk", "qdag_channel.fld"):
        src = stage1 / f
        if src.exists():
            shutil.copy2(src, out / f)
    s1_tag = f"stage1_t0-{stage1_fintime:g}"
    s2_tag = f"stage2_t{stage1_fintime:g}-{total_fintime:g}"
    shutil.copy2(stage1 / "casefile.xml", out / f"casefile_{s1_tag}.xml")
    shutil.copy2(stage2 / "casefile.xml", out / f"casefile_{s2_tag}.xml")
    (out / "README.md").write_text(
        f"# Assembled DO run: {label}, t = 0-{total_fintime:g}\n\n"
        f"Stage 1: {stage1.name} (t=0-{stage1_fintime:g})\n"
        f"Stage 2: {stage2.name} (restart from stage1's t={stage1_fintime:g} "
        f"DOArchive state, global t={stage1_fintime:g}-{total_fintime:g})\n\n"
        "Assembled by py_utils/sudden_exp_postproc/assemble_do_total_run.py. "
        "Continuity verified by md5 match of restart_archive.fld against "
        "stage1's final archive.\n"
    )


def assemble_chks(stage1: Path, stage2: Path, out: Path, chk_stride: int):
    for i in range(0, chk_stride + 1):
        shutil.copytree(stage1 / f"output/chks/casefile_{i}.chk",
                        out / f"output/chks/casefile_{i}.chk")
    src_dir = stage2 / "output/chks"
    idxs = sorted(int(p.name.split("_")[1].split(".")[0])
                  for p in src_dir.glob("casefile_*.chk"))
    for i in idxs:
        if i == 0:
            continue  # duplicate of stage1's final chk
        shutil.copytree(src_dir / f"casefile_{i}.chk",
                        out / f"output/chks/casefile_{i + chk_stride}.chk")
    n = len(list((out / "output/chks").glob("casefile_*.chk")))
    expect = chk_stride + 1 + len(idxs) - 1
    assert n == expect, (n, expect)
    print(f"chks: {n} files (0..{chk_stride + max(idxs) - 1 + 1})")


_STEP_RE = re.compile(
    r"(<DOVelocityCorrectionScheme_step>)(\d+)(</DOVelocityCorrectionScheme_step>)")
_TIME_RE = re.compile(
    r"(<DOVelocityCorrectionScheme_time>)([0-9.eE+-]+)"
    r"(</DOVelocityCorrectionScheme_time>)")


def _shift_metadata_file(path: Path, step_off: int, time_off: float):
    text = path.read_text()

    def sub_step(m):
        return f"{m.group(1)}{int(m.group(2)) + step_off}{m.group(3)}"

    def sub_time(m):
        return f"{m.group(1)}{float(m.group(2)) + time_off:.16f}{m.group(3)}"

    text = _STEP_RE.sub(sub_step, text)
    text = _TIME_RE.sub(sub_time, text)
    path.write_text(text)


def assemble_do_archive(stage1: Path, stage2: Path, out: Path,
                        stage1_steps: int, stage1_fintime: float,
                        total_steps: int):
    for d in sorted(stage1.glob("output/do/casefile.do_*.fld")):
        shutil.copytree(d, out / "output/do" / d.name)

    for d in sorted(stage2.glob("output/do/casefile.do_*.fld")):
        step = int(d.name.split("do_")[1].split(".")[0])
        if step == 0:
            continue  # duplicate of stage1's final archive
        new_step = step + stage1_steps
        new_name = f"casefile.do_{new_step:06d}.fld"
        dst = out / "output/do" / new_name
        shutil.copytree(d, dst)
        _shift_metadata_file(dst / "Info.xml", stage1_steps, stage1_fintime)
        for p in dst.glob("P*.fld"):
            _shift_metadata_file(p, stage1_steps, stage1_fintime)

    n = len(list((out / "output/do").glob("casefile.do_*.fld")))
    print(f"do archive: {n} snapshots")
    return n


def assemble_his(stage1: Path, stage2: Path, out: Path, stage1_fintime: float,
                 npts: int):
    lines1 = (stage1 / "output/asymmetry/axis_v.his").read_text().splitlines()
    header = [l for l in lines1 if l.startswith("#")]
    data1 = [l for l in lines1 if not l.startswith("#")]

    lines2 = (stage2 / "output/asymmetry/axis_v.his").read_text().splitlines()
    data2 = [l for l in lines2 if not l.startswith("#")]
    data2 = data2[npts:]  # drop first (t=0, duplicate) block

    out_lines = []
    for l in data2:
        parts = l.split()
        t_new = float(parts[0]) + stage1_fintime
        out_lines.append(f"{t_new:.19e} " + " ".join(parts[1:]))

    out_file = out / "output/asymmetry/axis_v.his"
    out_file.write_text("\n".join(header + data1 + out_lines) + "\n")

    n_rows = len(data1) + len(out_lines)
    print(f"axis_v.his: {n_rows} rows ({n_rows // npts} time blocks)")
    return n_rows


def assemble_energy(stage1: Path, stage2: Path, out: Path,
                    stage1_fintime: float):
    lines1 = (stage1 / "output/energy/energy.mdl").read_text().splitlines()
    header, data1 = lines1[0], lines1[1:]

    lines2 = (stage2 / "output/energy/energy.mdl").read_text().splitlines()
    data2 = lines2[1:][1:]  # drop header, then drop duplicate t=0 row

    out_lines = []
    for l in data2:
        parts = l.split()
        t_new = float(parts[0]) + stage1_fintime
        out_lines.append(f"{t_new:g}\t{parts[1]}")

    out_file = out / "output/energy/energy.mdl"
    out_file.write_text("\n".join([header] + data1 + out_lines) + "\n")

    n = len(data1) + len(out_lines)
    print(f"energy.mdl: {n} rows")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage1", type=Path)
    ap.add_argument("stage2", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--stage1-fintime", type=float, required=True)
    ap.add_argument("--timestep", type=float, default=0.01)
    ap.add_argument("--chk-freq", type=int, default=2500,
                    help="Checkpoint filter OutputFrequency in steps")
    ap.add_argument("--archive-freq", type=int, default=100,
                    help="DOArchive/HistoryPoints/ModalEnergy OutputFrequency in steps")
    ap.add_argument("--npts", type=int, default=401,
                    help="HistoryPoints line point count")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    stage1_steps = round(args.stage1_fintime / args.timestep)
    chk_stride = stage1_steps // args.chk_freq

    stage2_fintime = float(
        re.search(r"FinTime\s*=\s*([0-9.]+)",
                 (args.stage2 / "casefile.xml").read_text()).group(1))
    total_fintime = args.stage1_fintime + stage2_fintime
    total_steps = round(total_fintime / args.timestep)

    verify_seam(args.stage1, args.stage2, stage1_steps)
    reset(args.out)
    copy_static(args.stage1, args.stage2, args.out, args.stage1_fintime,
               total_fintime, args.label)
    assemble_chks(args.stage1, args.stage2, args.out, chk_stride)
    n_do = assemble_do_archive(args.stage1, args.stage2, args.out,
                               stage1_steps, args.stage1_fintime, total_steps)
    expect_do = total_steps // args.archive_freq + 1
    assert n_do == expect_do, (n_do, expect_do)

    n_his = assemble_his(args.stage1, args.stage2, args.out,
                         args.stage1_fintime, args.npts)
    expect_his = args.npts * (total_steps // args.archive_freq + 1)
    assert n_his == expect_his, (n_his, expect_his)

    n_en = assemble_energy(args.stage1, args.stage2, args.out,
                           args.stage1_fintime)
    expect_en = total_steps // args.archive_freq + 1
    assert n_en == expect_en, (n_en, expect_en)

    print(f"\nassembled run at {args.out} (t=0-{total_fintime:g})")


if __name__ == "__main__":
    main()
