"""Solver process runner with checkpoint watching."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtCore import QObject, QProcess, QTimer, pyqtSignal


_CHK_RE = re.compile(r"^(.*)_(\d+)\.chk$")


@dataclass
class RunSpec:
    binary: Path
    xml_path: Path
    work_dir: Path
    chk_dir: Path
    chk_basename: str
    final_time: float
    time_step: float
    chk_frequency: int


class SolverRunner(QObject):
    started = pyqtSignal()
    finished = pyqtSignal(int)
    stdout_line = pyqtSignal(str)
    stderr_line = pyqtSignal(str)
    chk_dropped = pyqtSignal(int, float)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._proc = QProcess(self)
        self._proc.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        self._proc.readyReadStandardOutput.connect(self._on_stdout)
        self._proc.readyReadStandardError.connect(self._on_stderr)
        self._proc.finished.connect(self._on_finished)
        self._proc.started.connect(lambda: self.started.emit())

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._poll_chk)

        self._spec: RunSpec | None = None
        self._known_chk: set[int] = set()

    def is_running(self) -> bool:
        return self._proc.state() != QProcess.ProcessState.NotRunning

    def start(self, spec: RunSpec) -> None:
        if self.is_running():
            raise RuntimeError("solver already running")
        self._spec = spec
        self._known_chk.clear()
        spec.chk_dir.mkdir(parents=True, exist_ok=True)
        self._proc.setWorkingDirectory(str(spec.work_dir))
        self._proc.setProgram(str(spec.binary))
        self._proc.setArguments([str(spec.xml_path)])
        self._proc.start()
        self._timer.start()

    def stop(self) -> None:
        if self._proc.state() != QProcess.ProcessState.NotRunning:
            self._proc.terminate()
            if not self._proc.waitForFinished(3000):
                self._proc.kill()

    def _on_stdout(self) -> None:
        data = bytes(self._proc.readAllStandardOutput()).decode(errors="replace")
        for line in data.splitlines():
            self.stdout_line.emit(line)

    def _on_stderr(self) -> None:
        data = bytes(self._proc.readAllStandardError()).decode(errors="replace")
        for line in data.splitlines():
            self.stderr_line.emit(line)

    def _on_finished(self, code: int, _status) -> None:
        self._timer.stop()
        self._poll_chk()
        self.finished.emit(int(code))

    def _poll_chk(self) -> None:
        spec = self._spec
        if spec is None:
            return
        try:
            entries = list(spec.chk_dir.iterdir())
        except FileNotFoundError:
            return
        new_idxs: list[int] = []
        for e in entries:
            if not e.is_file() or e.suffix != ".chk":
                continue
            m = _CHK_RE.match(e.name)
            if not m:
                continue
            base, idx = m.group(1), int(m.group(2))
            if base != spec.chk_basename:
                continue
            if idx in self._known_chk:
                continue
            new_idxs.append(idx)
        for idx in sorted(new_idxs):
            self._known_chk.add(idx)
            t_sim = idx * spec.chk_frequency * spec.time_step
            self.chk_dropped.emit(idx, t_sim)
