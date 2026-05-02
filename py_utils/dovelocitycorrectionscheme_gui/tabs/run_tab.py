"""Run tab: edit DOVelocityCorrectionScheme XML, launch the solver, watch ETA."""
from __future__ import annotations

import shutil
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from py_utils.dovelocitycorrectionscheme_gui import xml_io
from py_utils.dovelocitycorrectionscheme_gui.config import DO_PARAM_KEYS, find_solver_binary
from py_utils.dovelocitycorrectionscheme_gui.eta import ETAEstimator, fmt_seconds
from py_utils.dovelocitycorrectionscheme_gui.runner import RunSpec, SolverRunner
from py_utils.dovelocitycorrectionscheme_gui.widgets import (
    CardGroup,
    ConsoleView,
    PathPicker,
    big_button,
    hsep,
)


PARAM_LABELS = {
    "TimeStep": "Δt (TimeStep)",
    "FinTime": "Final time",
    "IO_InfoSteps": "IO_InfoSteps",
    "IO_CFLSteps": "IO_CFLSteps",
    "Re": "Reynolds",
    "DOModes": "DO modes",
    "DOParticles": "DO particles",
    "DOYiSeed": "Yᵢ seed",
    "DOYiSigma": "Yᵢ σ",
    "DOForcingNumChannels": "Forcing channels",
    "DOForcingSigma": "Forcing σ",
    "DOForcingTau": "Forcing τ",
    "DOForcingSeed": "Forcing seed",
}


class RunTab(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._tree = None
        self._case = xml_io.DOVelocityCorrectionSchemeCase()
        self._eta = ETAEstimator()
        self._runner = SolverRunner(self)
        self._runner.started.connect(self._on_started)
        self._runner.finished.connect(self._on_finished)
        self._runner.stdout_line.connect(lambda s: self.console.append_line(s))
        self._runner.stderr_line.connect(lambda s: self.console.append_line(s, color="#FF9090"))
        self._runner.chk_dropped.connect(self._on_chk)
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 12, 16, 12)
        outer.setSpacing(10)

        hero = QLabel("DOVelocityCorrectionScheme — Run a simulation")
        hero.setProperty("kind", "hero")
        outer.addWidget(hero)

        sub = QLabel("Load a DOVelocityCorrectionScheme casefile, tweak the parameters, and launch the solver.")
        sub.setProperty("kind", "muted")
        outer.addWidget(sub)

        self.template_picker = PathPicker(
            mode="open-file",
            filter_="NEKTAR XML (*.xml);;All files (*)",
            placeholder="Path to a casefile.xml template",
        )
        self.template_picker.changed.connect(self._maybe_autoload)
        load_btn = QPushButton("Load")
        load_btn.clicked.connect(self._load_template)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Template:"))
        row1.addWidget(self.template_picker, 1)
        row1.addWidget(load_btn)
        outer.addLayout(row1)

        self.run_dir_picker = PathPicker(mode="open-dir", placeholder="Run directory (where edited XML & output go)")
        self.run_dir_picker.changed.connect(self._update_status)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Run dir:"))
        row2.addWidget(self.run_dir_picker, 1)
        outer.addLayout(row2)

        self.binary_picker = PathPicker(
            mode="open-file",
            placeholder="IncNavierStokesSolver binary (auto-detected)",
        )
        bin_default = find_solver_binary()
        if bin_default is not None:
            self.binary_picker.set_value(str(bin_default))
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Solver:"))
        row3.addWidget(self.binary_picker, 1)
        outer.addLayout(row3)

        outer.addWidget(hsep())

        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        outer.addWidget(split, 1)

        params_widget = QWidget()
        params_layout = QVBoxLayout(params_widget)
        params_layout.setContentsMargins(0, 0, 0, 0)
        params_layout.setSpacing(10)

        self._param_edits: dict[str, QLineEdit] = {}

        time_card = CardGroup("Time")
        for k in ("TimeStep", "FinTime", "IO_InfoSteps", "IO_CFLSteps"):
            le = QLineEdit()
            self._param_edits[k] = le
            time_card.add(PARAM_LABELS[k], le)
        params_layout.addWidget(time_card)

        physics_card = CardGroup("Physics")
        for k in ("Re",):
            le = QLineEdit()
            self._param_edits[k] = le
            physics_card.add(PARAM_LABELS[k], le)
        self.solver_type_edit = QLineEdit("DOVelocityCorrectionScheme")
        physics_card.add("SolverType", self.solver_type_edit)
        params_layout.addWidget(physics_card)

        do_card = CardGroup("DO")
        for k in ("DOModes", "DOParticles", "DOYiSeed", "DOYiSigma"):
            le = QLineEdit()
            self._param_edits[k] = le
            do_card.add(PARAM_LABELS[k], le)
        params_layout.addWidget(do_card)

        forcing_card = CardGroup("Stochastic forcing")
        for k in ("DOForcingNumChannels", "DOForcingSigma", "DOForcingTau", "DOForcingSeed"):
            le = QLineEdit()
            self._param_edits[k] = le
            forcing_card.add(PARAM_LABELS[k], le)
        params_layout.addWidget(forcing_card)

        io_card = CardGroup("Checkpoints")
        self.chk_freq_spin = QSpinBox()
        self.chk_freq_spin.setRange(1, 10_000_000)
        self.chk_freq_spin.setValue(500)
        io_card.add("Output every N steps", self.chk_freq_spin)
        self.chk_basename_edit = QLineEdit("output/casefile")
        io_card.add("Output basename", self.chk_basename_edit)
        self.do_archive_chk = QCheckBox("Also write DOArchive at the same cadence")
        self.do_archive_chk.setChecked(True)
        io_card.add("DOArchive", self.do_archive_chk)
        params_layout.addWidget(io_card)
        params_layout.addStretch(1)

        split.addWidget(params_widget)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        status_card = CardGroup("Status")
        self.status_label = QLabel("Idle.")
        status_card.add("State:", self.status_label)
        self.eta_label = QLabel("—")
        status_card.add("ETA:", self.eta_label)
        self.elapsed_label = QLabel("0s")
        status_card.add("Elapsed:", self.elapsed_label)
        self.last_chk_label = QLabel("—")
        status_card.add("Last chk:", self.last_chk_label)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        status_card.add("Progress:", self.progress)
        right_layout.addWidget(status_card)

        actions = QHBoxLayout()
        self.save_btn = big_button("Save XML")
        self.save_btn.clicked.connect(self._save_xml)
        self.run_btn = big_button("Run", primary=True)
        self.run_btn.clicked.connect(self._run)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setProperty("kind", "danger")
        self.stop_btn.setMinimumHeight(36)
        self.stop_btn.clicked.connect(self._runner.stop)
        self.stop_btn.setEnabled(False)
        actions.addWidget(self.save_btn)
        actions.addWidget(self.run_btn)
        actions.addWidget(self.stop_btn)
        right_layout.addLayout(actions)

        self.console = ConsoleView()
        right_layout.addWidget(self.console, 1)

        split.addWidget(right)
        split.setSizes([500, 700])

        self._fill_form_from_case()
        self._update_status()

    # --- form <-> case sync ---

    def _fill_form_from_case(self) -> None:
        for k, edit in self._param_edits.items():
            edit.setText(self._case.params.get(k, ""))
        self.solver_type_edit.setText(self._case.solver_type or "DOVelocityCorrectionScheme")
        cp = self._case.checkpoint
        if cp is not None:
            self.chk_freq_spin.setValue(int(cp.output_frequency))
            self.chk_basename_edit.setText(cp.output_file)
        self.do_archive_chk.setChecked(self._case.do_archive is not None)

    def _commit_form_to_case(self) -> None:
        for k, edit in self._param_edits.items():
            v = edit.text().strip()
            if v:
                self._case.params[k] = v
            elif k in self._case.params:
                del self._case.params[k]
        self._case.solver_type = self.solver_type_edit.text().strip() or "DOVelocityCorrectionScheme"
        cp = self._case.checkpoint or xml_io.CheckpointFilter()
        cp.output_frequency = int(self.chk_freq_spin.value())
        cp.output_file = self.chk_basename_edit.text().strip() or "output/casefile"
        self._case.checkpoint = cp
        if self.do_archive_chk.isChecked():
            arc = self._case.do_archive or xml_io.CheckpointFilter()
            arc.output_frequency = cp.output_frequency
            arc.output_file = cp.output_file
            self._case.do_archive = arc
        else:
            self._case.do_archive = None

    # --- handlers ---

    def _maybe_autoload(self, _txt: str) -> None:
        p = self.template_picker.path()
        if p and p.exists() and self._tree is None:
            self._load_template()

    def _load_template(self) -> None:
        p = self.template_picker.path()
        if p is None or not p.exists():
            QMessageBox.warning(self, "Template", "Template XML not found.")
            return
        try:
            tree, case = xml_io.parse_xml(p)
        except Exception as exc:
            QMessageBox.critical(self, "Template", f"Failed to parse XML:\n{exc}")
            return
        self._tree = tree
        self._case = case
        self._fill_form_from_case()
        self.console.append_line(f"[gui] loaded template {p}")

    def _resolve_paths(self) -> tuple[Path, Path] | None:
        run_dir = self.run_dir_picker.path()
        if run_dir is None:
            QMessageBox.warning(self, "Run dir", "Choose a run directory first.")
            return None
        run_dir = run_dir.expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        xml_path = run_dir / "casefile.xml"
        return run_dir, xml_path

    def _save_xml(self) -> Path | None:
        if self._tree is None:
            QMessageBox.warning(self, "Save", "Load a template XML first.")
            return None
        paths = self._resolve_paths()
        if paths is None:
            return None
        run_dir, xml_path = paths
        self._commit_form_to_case()
        try:
            xml_io.write_xml(self._tree, self._case, xml_path)
        except Exception as exc:
            QMessageBox.critical(self, "Save", f"Failed to write XML:\n{exc}")
            return None
        self.console.append_line(f"[gui] wrote {xml_path}")
        return xml_path

    def _run(self) -> None:
        if self._runner.is_running():
            return
        binary = self.binary_picker.path()
        if binary is None or not binary.exists():
            QMessageBox.warning(self, "Solver", "Solver binary not found.")
            return

        xml_path = self._save_xml()
        if xml_path is None:
            return
        run_dir = xml_path.parent
        self._commit_form_to_case()

        try:
            time_step = float(self._case.params["TimeStep"])
            final_time = float(self._case.params["FinTime"])
        except Exception:
            QMessageBox.warning(self, "Run", "TimeStep and FinTime must be numeric.")
            return
        chk_freq = int(self.chk_freq_spin.value())
        chk_basename_full = self.chk_basename_edit.text().strip() or "output/casefile"
        chk_dir_rel = Path(chk_basename_full).parent
        chk_basename = Path(chk_basename_full).name
        chk_dir = (run_dir / chk_dir_rel).resolve()

        if shutil.which(str(binary)) is None and not binary.exists():
            QMessageBox.critical(self, "Solver", f"Cannot execute {binary}")
            return

        self._eta.reset()
        self._eta.start()
        self._final_time = final_time
        self._chk_count = 0
        self.progress.setValue(0)
        self.eta_label.setText("—")
        self.last_chk_label.setText("—")
        self.console.append_line(f"[gui] launching {binary} {xml_path}")

        spec = RunSpec(
            binary=binary,
            xml_path=xml_path,
            work_dir=run_dir,
            chk_dir=chk_dir,
            chk_basename=chk_basename,
            final_time=final_time,
            time_step=time_step,
            chk_frequency=chk_freq,
        )
        try:
            self._runner.start(spec)
        except Exception as exc:
            QMessageBox.critical(self, "Run", str(exc))

    def _on_started(self) -> None:
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Running")
        self.status_label.setProperty("kind", "warn")
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def _on_finished(self, code: int) -> None:
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if code == 0:
            self.status_label.setText("Done")
            self.status_label.setProperty("kind", "ok")
            self.progress.setValue(1000)
        else:
            self.status_label.setText(f"Failed (code {code})")
            self.status_label.setProperty("kind", "error")
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def _on_chk(self, idx: int, t_sim: float) -> None:
        self._chk_count += 1
        self._eta.record(t_sim)
        self.last_chk_label.setText(f"#{idx} (t = {t_sim:.4g})")
        eta = self._eta.eta_seconds(self._final_time)
        self.eta_label.setText(fmt_seconds(eta))
        self.progress.setValue(int(round(self._eta.progress(self._final_time) * 1000)))
        self.elapsed_label.setText(fmt_seconds(self._eta._samples[-1].wall if self._eta.n else 0))
        self.console.append_line(f"[chk] #{idx} t_sim={t_sim:.4g}  ETA {fmt_seconds(eta)}")

    def _update_status(self) -> None:
        run_dir = self.run_dir_picker.path()
        if run_dir and run_dir.exists():
            self.status_label.setText("Ready")
        else:
            self.status_label.setText("Idle")
