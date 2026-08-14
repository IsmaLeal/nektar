#!/usr/bin/env python3
"""Single entrypoint for reproducible py_utils workflows.

Two usage styles are supported:

1) Direct command passthrough
   python3 py_utils/workflow.py extract -- --xml <case.xml> --chk-dir <output> --out <out.h5>
   python3 py_utils/workflow.py suite -- --data <out.h5>

2) Case pipeline runner (recommended)
   python3 py_utils/workflow.py run-case --case-dir <case_dir>
   python3 py_utils/workflow.py run-case --case-dir <case_dir> --stages extract,derive,suite,check
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


SCRIPT_MAP = {
    "extract": "load_chk.py",
    "ic-profile": "ic_sampler/create_profile.py",
    "ic-generate": "ic_sampler/generate_ic_samples.py",
    "derive": "reconstruction/reconstruct_fields.py",
    "check": "check_convergence.py",
    "plot-mesh": "mesh/plot_geometry_xml.py",
    "video": "vorticity_video.py",
    "panels": "reconstruction/animation/animate_do_panels.py",
    "suite": "tests/run_do_test_suite.py",
    "reg-derive-stats": "tests/regression/do_derive_stats_csv_from_h5.py",
    "reg-compare-sem": "tests/regression/do_compare_sem_ns2d.py",
    "reg-compare-trace": "tests/regression/do_compare_trace_history.py",
    "reg-plot-variance": "tests/regression/do_plot_mode_variance.py",
}

PIPELINE_STAGES = {"extract", "derive", "suite", "check", "video", "panels", "reg-derive-stats"}
STAGE_HELP = {
    "extract": "read chk/archive and write canonical out.h5",
    "derive": "reconstruct realization fields from mean+modes+yi",
    "suite": "run intrinsic DO validation tests (raw archive hard-gate + field diagnostics)",
    "check": "run generic steady/periodic settling diagnostics",
    "plot-mesh": "plot 2D mesh from geometry XML",
    "video": "render vorticity video from u,v",
    "panels": "render DO multi-panel animation (mean/modes/realization/yi variance)",
    "reg-derive-stats": "derive authoritative raw DO stats CSV from out.h5",
}


def _run_script(root: Path, name: str, args: list[str]) -> int:
    script = root / SCRIPT_MAP[name]
    cmd = [sys.executable, str(script), *args]
    print(f"[run] {name}: {' '.join(shlex.quote(x) for x in cmd)}")
    return int(subprocess.run(cmd).returncode)


def _split_stage_list(text: str) -> list[str]:
    items = [x.strip() for x in text.split(",") if x.strip()]
    bad = [x for x in items if x not in PIPELINE_STAGES]
    if bad:
        raise ValueError(f"Unknown stage(s): {bad}. Allowed: {sorted(PIPELINE_STAGES)}")
    return items


def _default_xml(case_dir: Path) -> Path | None:
    c1 = case_dir / "casefile.xml"
    if c1.exists():
        return c1
    c2 = case_dir / "mesh" / "geometry.xml"
    if c2.exists():
        return c2
    return None


def _parse_pipeline_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run a reproducible case pipeline.")
    ap.add_argument("--case-dir", type=Path, required=True, help="Case directory root.")
    ap.add_argument(
        "--stages",
        default="extract,derive,panels",
        help=(
            "Comma-separated stages. Allowed: "
            "extract,derive,suite,check,video,panels,reg-derive-stats"
        ),
    )
    ap.add_argument(
        "--fieldconvert",
        type=Path,
        default=None,
        help="Path to FieldConvert binary (forwarded to extract).",
    )
    ap.add_argument("--xml", type=Path, default=None, help="Override case XML path.")
    ap.add_argument("--chk-dir", type=Path, default=None, help="Override checkpoint directory.")
    ap.add_argument("--out-h5", type=Path, default=None, help="Canonical extracted output file.")
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Result directory for reports/plots/derived tables.",
    )
    ap.add_argument(
        "--reconstructed-out",
        type=Path,
        default=None,
        help="Output file for reconstructed realizations (derive stage).",
    )
    ap.add_argument("--nx", type=int, default=None, help="Extraction grid Nx override.")
    ap.add_argument("--ny", type=int, default=None, help="Extraction grid Ny override.")
    ap.add_argument("--limit", type=int, default=None, help="Limit checkpoints for extraction.")
    ap.add_argument("--dt", type=float, default=None, help="Override dt in extraction.")
    ap.add_argument("--chk-stride", type=int, default=None, help="Override checkpoint stride.")
    ap.add_argument("--check-mode", choices=["auto", "steady", "periodic"], default="auto")
    ap.add_argument("--video-fps", type=int, default=30)
    ap.add_argument("--video-duration-sec", type=float, default=12.0)
    ap.add_argument("--keep-going", action="store_true", help="Continue remaining stages after failures.")
    return ap.parse_args(argv)


def _run_case(root: Path, argv: list[str]) -> int:
    args = _parse_pipeline_args(argv)
    case_dir = args.case_dir.resolve()
    if not case_dir.exists():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")

    stages = _split_stage_list(args.stages)
    xml = args.xml.resolve() if args.xml is not None else _default_xml(case_dir)
    chk_dir = args.chk_dir.resolve() if args.chk_dir is not None else (case_dir / "output")
    out_h5 = args.out_h5.resolve() if args.out_h5 is not None else (case_dir / "output" / "out.h5")
    results_dir = (
        args.results_dir.resolve()
        if args.results_dir is not None
        else (case_dir / "output" / "py_utils_results")
    )
    reconstructed_out = (
        args.reconstructed_out.resolve()
        if args.reconstructed_out is not None
        else (case_dir / "output" / "out_reconstructed.h5")
    )

    out_h5.parent.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[str, int]] = []

    for stg in stages:
        print(f"[stage] {stg}: {STAGE_HELP[stg]}")

        if stg == "extract":
            cmd = ["--out", str(out_h5)]
            if xml is not None:
                cmd.extend(["--xml", str(xml)])
            if chk_dir.exists():
                cmd.extend(["--chk-dir", str(chk_dir)])
            if args.nx is not None:
                cmd.extend(["--nx", str(args.nx)])
            if args.ny is not None:
                cmd.extend(["--ny", str(args.ny)])
            if args.limit is not None:
                cmd.extend(["--limit", str(args.limit)])
            if args.dt is not None:
                cmd.extend(["--dt", str(args.dt)])
            if args.chk_stride is not None:
                cmd.extend(["--chk-stride", str(args.chk_stride)])
            if args.fieldconvert is not None:
                cmd.extend(["--fieldconvert", str(args.fieldconvert)])
            rc = _run_script(root, "extract", cmd)
        elif stg == "derive":
            rc = _run_script(root, "derive", [str(out_h5), "--out", str(reconstructed_out)])
        elif stg == "suite":
            rc = _run_script(root, "suite", ["--data", str(out_h5), "--out-dir", str(results_dir / "suite")])
        elif stg == "check":
            rc = _run_script(
                root,
                "check",
                [
                    "--data",
                    str(out_h5),
                    "--mode",
                    args.check_mode,
                    "--csv-out",
                    str(results_dir / "check_convergence_metrics.csv"),
                    "--plot-out",
                    str(results_dir / "check_convergence_plot.png"),
                ],
            )
        elif stg == "video":
            rc = _run_script(
                root,
                "video",
                [
                    "--data",
                    str(out_h5),
                    "--out",
                    str(results_dir / "vorticity.mp4"),
                    "--fps",
                    str(args.video_fps),
                    "--duration-sec",
                    str(args.video_duration_sec),
                ],
            )
        elif stg == "panels":
            rc = _run_script(
                root,
                "panels",
                [
                    "--data",
                    str(out_h5),
                    "--field",
                    "velocity",
                    "--out",
                    str(results_dir / "do_panels.mp4"),
                    "--fps",
                    str(max(1, min(args.video_fps, 12))),
                ],
            )
        elif stg == "reg-derive-stats":
            rc = _run_script(
                root,
                "reg-derive-stats",
                ["--sem-h5", str(out_h5), "--out-csv", str(results_dir / "do_stats_derived.csv")],
            )
        else:
            raise AssertionError(f"Unhandled stage: {stg}")

        if rc != 0:
            failures.append((stg, rc))
            if not args.keep_going:
                break

    if failures:
        print(f"[summary] failures: {failures}")
        return int(failures[0][1])

    print("[summary] pipeline completed successfully")
    print(f"[summary] case_dir={case_dir}")
    print(f"[summary] out_h5={out_h5}")
    print(f"[summary] results_dir={results_dir}")
    return 0


def _print_help() -> None:
    print(__doc__)
    print("\nDirect passthrough commands:")
    for k in sorted(SCRIPT_MAP):
        print(f"  - {k}")
    print("\nCase pipeline:")
    print("  python3 py_utils/workflow.py run-case --case-dir <case_dir>")
    print("  python3 py_utils/workflow.py run-case --case-dir <case_dir> --stages extract,derive,suite,check")
    print("\nGUI:")
    print("  python3 py_utils/workflow.py gui")
    print("\nFor passthrough usage:")
    print("  python3 py_utils/workflow.py <command> -- --help")


def _run_gui() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from py_utils.dovelocitycorrectionscheme_gui.app import main as gui_main

    return int(gui_main())


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help", "help"}:
        _print_help()
        return 0

    cmd = sys.argv[1]
    root = Path(__file__).resolve().parent

    if cmd == "run-case":
        return _run_case(root, sys.argv[2:])

    if cmd == "gui":
        return _run_gui()

    if cmd not in SCRIPT_MAP:
        print(f"Unknown command '{cmd}'.\n")
        _print_help()
        return 2

    args = sys.argv[2:]
    if args and args[0] == "--":
        args = args[1:]
    return _run_script(root, cmd, args)


if __name__ == "__main__":
    raise SystemExit(main())
