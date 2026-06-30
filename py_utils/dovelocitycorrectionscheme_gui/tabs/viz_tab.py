"""Visualization tab: pick a run, choose a plot/animation, render and view it."""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from PyQt6.QtCore import Qt, QProcess, QUrl
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from py_utils.dovelocitycorrectionscheme_gui.config import workflow_script
from py_utils.dovelocitycorrectionscheme_gui.widgets import CardGroup, ConsoleView, PathPicker, big_button, hsep


# -- multimedia (optional) --
try:
    from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
    from PyQt6.QtMultimediaWidgets import QVideoWidget
    _HAS_MULTIMEDIA = True
except Exception:
    _HAS_MULTIMEDIA = False


@dataclass
class StageDef:
    key: str
    label: str
    output: str  # "image", "video", or "h5"
    out_filename: str
    needs_h5: bool = True
    build_args: Callable[["VizTab", Path, Path], list[str]] = field(default=lambda *_: [])


def _extract_args(self: "VizTab", out: Path, _data: Path) -> list[str]:
    args: list[str] = []
    chosen = self.xml_picker.path()
    xml = _resolve_geometry_xml(chosen)
    if chosen is not None and xml is not None and xml != chosen:
        self.console.append_line(
            f"[gui] {chosen.name} has no <GEOMETRY> — using {xml} for extract"
        )
    if xml is not None:
        args += ["--xml", str(xml)]
    chk_dir = self._chk_dir()
    if chk_dir is not None:
        args += ["--chk-dir", str(chk_dir)]
    args += ["--out", str(out)]
    if self.extract_nx.value() > 0:
        args += ["--nx", str(self.extract_nx.value())]
    if self.extract_ny.value() > 0:
        args += ["--ny", str(self.extract_ny.value())]
    if self.extract_limit.value() > 0:
        args += ["--limit", str(self.extract_limit.value())]
    if self.extract_step.value() > 1:
        args += ["--chk-step", str(self.extract_step.value())]
    args.append("--interp-modes-from-archive")
    return args


def _video_args(self: "VizTab", out: Path, _data: Path) -> list[str]:
    return [
        "--out", str(out),
        "--fps", str(self.video_fps.value()),
        "--duration-sec", f"{self.video_duration.value():.2f}",
    ]


def _panels_args(self: "VizTab", out: Path, _data: Path) -> list[str]:
    args = [
        "--out", str(out),
        "--fps", str(self.panels_fps.value()),
        "--field", self.panels_field.currentText(),
    ]
    if self.panels_max_frames.value() > 0:
        args += ["--max-frames", str(self.panels_max_frames.value())]
    return args


def _check_args(self: "VizTab", out: Path, _data: Path) -> list[str]:
    return [
        "--mode", self.check_mode.currentText(),
        "--csv-out", str(out.with_suffix(".csv")),
        "--plot-out", str(out),
    ]


def _has_geometry(p: Path) -> bool:
    try:
        with p.open("r", errors="replace") as fh:
            for _ in range(2000):
                line = fh.readline()
                if not line:
                    break
                if "<GEOMETRY" in line:
                    return True
    except OSError:
        pass
    return False


def _resolve_geometry_xml(xml: Path | None) -> Path | None:
    """For split casefile/geometry layouts, fall back to sibling files that
    actually contain a <GEOMETRY> block."""
    if xml is None:
        return None
    if xml.exists() and _has_geometry(xml):
        return xml
    parent = xml.parent if xml.exists() else None
    candidates: list[Path] = []
    if parent is not None:
        candidates += [
            parent / "combined.xml",
            parent / "mesh" / "geometry.xml",
            parent.parent / "mesh" / "geometry.xml",
        ]
    for cand in candidates:
        if cand.exists() and _has_geometry(cand):
            return cand
    return xml  # let the script raise a clear error


def _mesh_args(self: "VizTab", out: Path, _data: Path) -> list[str]:
    # plot_geometry_xml uses --xml as input and writes to --save
    chosen = self.xml_picker.path()
    geom = _resolve_geometry_xml(chosen)
    if geom is not None and chosen is not None and geom != chosen:
        self.console.append_line(
            f"[gui] {chosen.name} has no <GEOMETRY> — using {geom} instead"
        )
    args: list[str] = []
    if geom is not None:
        args += ["--xml", str(geom)]
    args += ["--save", str(out)]
    return args


STAGES = [
    StageDef("extract",         "Extract DO outputs (.h5)","h5",    "out.h5",               False, _extract_args),
    StageDef("video",           "Vorticity video",         "video", "vorticity.mp4",        True,  _video_args),
    StageDef("panels",          "DO multi-panel animation","video", "do_panels.mp4",        True,  _panels_args),
    StageDef("check",           "Convergence diagnostics", "image", "check_convergence.png",True,  _check_args),
    StageDef("plot-mesh",       "Mesh geometry",           "image", "mesh.png",             False, _mesh_args),
]


_FRAME_RE = re.compile(r"frame\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)


class VizTab(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._proc: QProcess | None = None
        self._current_stage: StageDef | None = None
        self._current_output: Path | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 12, 16, 12)
        outer.setSpacing(10)

        hero = QLabel("Visualize a DOVelocityCorrectionScheme run")
        hero.setProperty("kind", "hero")
        outer.addWidget(hero)

        sub = QLabel("Pick a run, choose what to plot, render — preview is shown inline.")
        sub.setProperty("kind", "muted")
        outer.addWidget(sub)

        self.run_dir_picker = PathPicker(mode="open-dir", placeholder="Run directory containing output/")
        self.run_dir_picker.changed.connect(self._auto_detect)
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Run dir:"))
        row1.addWidget(self.run_dir_picker, 1)
        outer.addLayout(row1)

        self.xml_picker = PathPicker(mode="open-file", filter_="XML (*.xml)", placeholder="casefile.xml (auto-detected)")
        self.h5_picker = PathPicker(mode="open-file", filter_="HDF5 (*.h5);;All files (*)", placeholder="out.h5 (auto-detected)")
        self.out_dir_picker = PathPicker(mode="open-dir", placeholder="Where to save plots/videos")

        row_xml = QHBoxLayout()
        row_xml.addWidget(QLabel("XML:"))
        row_xml.addWidget(self.xml_picker, 1)
        outer.addLayout(row_xml)

        row_h5 = QHBoxLayout()
        row_h5.addWidget(QLabel("out.h5:"))
        row_h5.addWidget(self.h5_picker, 1)
        outer.addLayout(row_h5)

        row_out = QHBoxLayout()
        row_out.addWidget(QLabel("Save to:"))
        row_out.addWidget(self.out_dir_picker, 1)
        outer.addLayout(row_out)

        outer.addWidget(hsep())

        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        outer.addWidget(split, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        stage_card = CardGroup("What to plot")
        self.stage_combo = QComboBox()
        for s in STAGES:
            self.stage_combo.addItem(s.label, s.key)
        self.stage_combo.currentIndexChanged.connect(self._on_stage_changed)
        stage_card.add("Stage:", self.stage_combo)
        left_layout.addWidget(stage_card)

        # Per-stage option cards
        self.options_stack = QStackedWidget()

        # Extract (.h5)
        extract_card = CardGroup("Extract options")
        self.extract_nx = QSpinBox()
        self.extract_nx.setRange(0, 10_000); self.extract_nx.setValue(0)
        self.extract_nx.setSpecialValueText("(default: 100)")
        extract_card.add("Grid Nx", self.extract_nx)
        self.extract_ny = QSpinBox()
        self.extract_ny.setRange(0, 10_000); self.extract_ny.setValue(0)
        self.extract_ny.setSpecialValueText("(default: 100)")
        extract_card.add("Grid Ny", self.extract_ny)
        self.extract_limit = QSpinBox()
        self.extract_limit.setRange(0, 1_000_000); self.extract_limit.setValue(0)
        self.extract_limit.setSpecialValueText("(all)")
        extract_card.add("Limit chk", self.extract_limit)
        self.extract_step = QSpinBox()
        self.extract_step.setRange(1, 10_000); self.extract_step.setValue(1)
        self.extract_step.setToolTip("Take every Nth checkpoint (1 = no skipping).")
        extract_card.add("Chk step", self.extract_step)
        self.options_stack.addWidget(extract_card)

        # Vorticity video
        vid_card = CardGroup("Vorticity video options")
        self.video_fps = QSpinBox(); self.video_fps.setRange(1, 120); self.video_fps.setValue(30)
        vid_card.add("FPS", self.video_fps)
        self.video_duration = QDoubleSpinBox(); self.video_duration.setRange(0.5, 600.0); self.video_duration.setValue(12.0); self.video_duration.setSuffix(" s")
        vid_card.add("Duration", self.video_duration)
        self.options_stack.addWidget(vid_card)

        # Panels
        panel_card = CardGroup("DO panels options")
        self.panels_fps = QSpinBox(); self.panels_fps.setRange(1, 60); self.panels_fps.setValue(6)
        panel_card.add("FPS", self.panels_fps)
        self.panels_field = QComboBox(); self.panels_field.addItems(["vorticity", "velocity"])
        panel_card.add("Field", self.panels_field)
        self.panels_max_frames = QSpinBox(); self.panels_max_frames.setRange(0, 100_000); self.panels_max_frames.setValue(0); self.panels_max_frames.setSpecialValueText("(all)")
        panel_card.add("Max frames", self.panels_max_frames)
        self.options_stack.addWidget(panel_card)

        # Check convergence
        check_card = CardGroup("Convergence options")
        self.check_mode = QComboBox(); self.check_mode.addItems(["auto", "steady", "periodic"])
        check_card.add("Mode", self.check_mode)
        self.options_stack.addWidget(check_card)

        # Mesh
        mesh_card = CardGroup("Mesh options")
        mesh_card.add("(no extra options)", QLabel(""))
        self.options_stack.addWidget(mesh_card)

        left_layout.addWidget(self.options_stack)

        actions = QHBoxLayout()
        self.render_btn = big_button("Render", primary=True)
        self.render_btn.clicked.connect(self._render)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setProperty("kind", "danger")
        self.cancel_btn.setMinimumHeight(36)
        self.cancel_btn.clicked.connect(self._cancel)
        self.cancel_btn.setEnabled(False)
        actions.addWidget(self.render_btn)
        actions.addWidget(self.cancel_btn)
        left_layout.addLayout(actions)

        progress_card = CardGroup("Progress")
        self.progress = QProgressBar(); self.progress.setRange(0, 0)  # indeterminate by default
        progress_card.add("State:", self.progress)
        self.frame_label = QLabel("—")
        progress_card.add("Frames:", self.frame_label)
        left_layout.addWidget(progress_card)

        self.console = ConsoleView()
        left_layout.addWidget(self.console, 1)

        split.addWidget(left)

        # Right: preview area
        preview = QWidget()
        pv_layout = QVBoxLayout(preview)
        pv_layout.setContentsMargins(0, 0, 0, 0)
        pv_layout.setSpacing(8)
        pv_layout.addWidget(QLabel("Preview"))
        self.preview_stack = QStackedWidget()

        # image preview
        self.image_label = QLabel("No preview yet — render something.")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setProperty("kind", "muted")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.image_label)
        self.preview_stack.addWidget(scroll)

        # video preview
        if _HAS_MULTIMEDIA:
            self.video_widget = QVideoWidget()
            self.media_player = QMediaPlayer()
            self.audio_out = QAudioOutput()
            self.media_player.setAudioOutput(self.audio_out)
            self.media_player.setVideoOutput(self.video_widget)

            video_box = QWidget()
            vb = QVBoxLayout(video_box)
            vb.setContentsMargins(0, 0, 0, 0)
            vb.addWidget(self.video_widget, 1)
            vb_ctl = QHBoxLayout()
            self.play_btn = QPushButton("▶ Play")
            self.play_btn.clicked.connect(self._toggle_play)
            vb_ctl.addWidget(self.play_btn)
            vb_ctl.addStretch(1)
            vb.addLayout(vb_ctl)
            self.preview_stack.addWidget(video_box)
        else:
            stub = QLabel("PyQt6.QtMultimedia not available — videos will be saved but not previewed inline.")
            stub.setAlignment(Qt.AlignmentFlag.AlignCenter)
            stub.setProperty("kind", "muted")
            self.preview_stack.addWidget(stub)

        pv_layout.addWidget(self.preview_stack, 1)

        self.output_path_label = QLabel("")
        self.output_path_label.setProperty("kind", "muted")
        self.output_path_label.setWordWrap(True)
        pv_layout.addWidget(self.output_path_label)

        split.addWidget(preview)
        split.setSizes([460, 760])

        self._on_stage_changed(0)

    # --- helpers ---

    def _auto_detect(self, _txt: str) -> None:
        run_dir = self.run_dir_picker.path()
        if run_dir is None or not run_dir.exists():
            return
        # If user picked the output/ dir directly, treat its parent as the run.
        case_root = run_dir.parent if run_dir.name == "output" else run_dir
        output_dir = run_dir if run_dir.name == "output" else run_dir / "output"

        # Prefer combined.xml when it exists: split-layout casefile.xml has no
        # <GEOMETRY> block, which makes FieldConvert dump core during extract.
        for cand in (
            case_root / "combined.xml",
            case_root / "casefile.xml",
            case_root / "mesh" / "geometry.xml",
        ):
            if cand.exists() and not self.xml_picker.value():
                self.xml_picker.set_value(str(cand))
                break
        h5 = output_dir / "out.h5"
        if h5.exists() and not self.h5_picker.value():
            self.h5_picker.set_value(str(h5))
        out = output_dir / "py_utils_results"
        if not self.out_dir_picker.value():
            self.out_dir_picker.set_value(str(out))

    def _on_stage_changed(self, idx: int) -> None:
        # one option card per stage, in same order
        self.options_stack.setCurrentIndex(min(idx, self.options_stack.count() - 1))

    def _chk_dir(self) -> Path | None:
        rd = self.run_dir_picker.path()
        if rd is None:
            return None
        return rd if rd.name == "output" else rd / "output"

    # --- run a stage ---

    def _stage_def(self) -> StageDef:
        key = self.stage_combo.currentData()
        for s in STAGES:
            if s.key == key:
                return s
        raise RuntimeError("no stage selected")

    def _render(self) -> None:
        if self._proc is not None and self._proc.state() != QProcess.ProcessState.NotRunning:
            return
        stage = self._stage_def()

        if stage.key == "extract":
            h5 = self.h5_picker.path()
            if h5 is not None:
                out_path = h5.expanduser()
            else:
                chk = self._chk_dir()
                if chk is None:
                    QMessageBox.warning(self, "Output", "Pick a run directory or set the out.h5 path.")
                    return
                out_path = chk / "out.h5"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            data = Path()
        else:
            out_dir = self.out_dir_picker.path()
            if out_dir is None:
                QMessageBox.warning(self, "Output", "Choose an output directory.")
                return
            out_dir = out_dir.expanduser()
            out_dir.mkdir(parents=True, exist_ok=True)

            if stage.needs_h5:
                data = self.h5_picker.path()
                if data is None or not data.exists():
                    QMessageBox.warning(self, "Data", "out.h5 not found. Run the Extract stage first.")
                    return
            else:
                data = self.h5_picker.path() or Path()

            out_path = out_dir / stage.out_filename

        args = [str(workflow_script()), stage.key, "--"]
        if stage.needs_h5:
            args += ["--data", str(data)]
        args += stage.build_args(self, out_path, data)

        self._current_stage = stage
        self._current_output = out_path
        self._spawn(sys.executable, args)
        self.console.append_line(f"[gui] {sys.executable} {' '.join(args)}")

    def _spawn(self, prog: str, args: list[str]) -> None:
        self._proc = QProcess(self)
        self._proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._proc.readyReadStandardOutput.connect(self._on_output)
        self._proc.finished.connect(self._on_finished)
        self.progress.setRange(0, 0)
        self.frame_label.setText("—")
        self.render_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self._proc.start(prog, args)

    def _on_output(self) -> None:
        data = bytes(self._proc.readAllStandardOutput()).decode(errors="replace") if self._proc else ""
        for line in data.splitlines():
            self.console.append_line(line)
            m = _FRAME_RE.search(line)
            if m:
                cur, tot = int(m.group(1)), int(m.group(2))
                self.progress.setRange(0, max(1, tot))
                self.progress.setValue(cur)
                self.frame_label.setText(f"{cur}/{tot}")

    def _on_finished(self, code: int, _status) -> None:
        self.render_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        if self.progress.maximum() == 0:
            self.progress.setRange(0, 1)
            self.progress.setValue(1)
        self._proc = None

        stage = self._current_stage
        out = self._current_output
        produced = out is not None and out.exists()

        # check_convergence.py uses exit code 2 to signal "not_converged" — that
        # is informational, not a failure. The PNG/CSV are still produced.
        if stage is not None and stage.key == "check" and code == 2:
            self.console.append_line(
                "[gui] convergence check: not_converged (exit 2) — outputs produced",
                color="#FFB454",
            )
            code = 0

        if code != 0 and not produced:
            self.console.append_line(f"[gui] render failed (code {code})", color="#FF9090")
            return
        if code != 0:
            self.console.append_line(
                f"[gui] script exited with code {code} but output exists — previewing",
                color="#FFB454",
            )
        else:
            self.console.append_line("[gui] render complete")

        if produced and stage is not None:
            self._show_preview(stage, out)
            self.output_path_label.setText(f"Saved to: {out}")

    def _cancel(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            if not self._proc.waitForFinished(1500):
                self._proc.kill()

    def _show_preview(self, stage: StageDef, path: Path) -> None:
        if not path.exists():
            self.console.append_line(f"[gui] expected output not found: {path}", color="#FF9090")
            return
        if stage.output == "h5":
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText(
                f"Extract complete — out.h5 written to:\n{path}\n\n"
                "You can now run video / panels / convergence stages."
            )
            self.preview_stack.setCurrentIndex(0)
            self.h5_picker.set_value(str(path))
            return
        if stage.output == "image":
            pix = QPixmap(str(path))
            if pix.isNull():
                self.image_label.setText(f"Cannot display {path}")
            else:
                self.image_label.setPixmap(pix.scaled(
                    self.image_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                ))
            self.preview_stack.setCurrentIndex(0)
        else:
            if _HAS_MULTIMEDIA:
                self.media_player.setSource(QUrl.fromLocalFile(str(path)))
                self.media_player.play()
                self.play_btn.setText("⏸ Pause")
                self.preview_stack.setCurrentIndex(1)
            else:
                self.preview_stack.setCurrentIndex(1)

    def _toggle_play(self) -> None:
        if not _HAS_MULTIMEDIA:
            return
        from PyQt6.QtMultimedia import QMediaPlayer  # noqa
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
            self.play_btn.setText("▶ Play")
        else:
            self.media_player.play()
            self.play_btn.setText("⏸ Pause")
