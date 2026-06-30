from __future__ import annotations

import os
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


CANDIDATE_BINARIES = [
    REPO_ROOT / "build" / "solvers" / "IncNavierStokesSolver" / "IncNavierStokesSolver",
    REPO_ROOT / "build" / "dist" / "bin" / "IncNavierStokesSolver",
    REPO_ROOT / "build-debug" / "solvers" / "IncNavierStokesSolver" / "IncNavierStokesSolver-g",
]


def find_solver_binary() -> Path | None:
    env = os.environ.get("DOMINE_SOLVER")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p
    for cand in CANDIDATE_BINARIES:
        if cand.exists() and os.access(cand, os.X_OK):
            return cand
    onpath = shutil.which("IncNavierStokesSolver")
    return Path(onpath) if onpath else None


def workflow_script() -> Path:
    return REPO_ROOT / "py_utils" / "workflow.py"


# DO parameter names that the GUI edits in <PARAMETERS>.
DO_PARAM_KEYS = [
    "TimeStep",
    "FinTime",
    "IO_InfoSteps",
    "IO_CFLSteps",
    "Re",
    "DOModes",
    "DOParticles",
    "DOYiSeed",
    "DOYiSigma",
    "DOForcingNumChannels",
    "DOForcingSigma",
    "DOForcingTau",
    "DOForcingSeed",
]
