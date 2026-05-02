"""Reusable widgets for the DOVelocityCorrectionScheme GUI."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QTextCursor
from PyQt6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QStyle,
    QStyleOption,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class CardGroup(QGroupBox):
    """Section card with a title."""

    def __init__(self, title: str, parent: QWidget | None = None):
        super().__init__(title, parent)
        self.setObjectName("Card")
        self.setProperty("class", "Card")
        self._form = QFormLayout(self)
        self._form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self._form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        self._form.setHorizontalSpacing(14)
        self._form.setVerticalSpacing(8)

    def add(self, label: str, widget: QWidget) -> QWidget:
        self._form.addRow(label, widget)
        return widget


class PathPicker(QWidget):
    changed = pyqtSignal(str)

    def __init__(
        self,
        mode: str = "open-file",
        filter_: str = "All files (*)",
        placeholder: str = "",
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.mode = mode
        self.filter_ = filter_
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.edit = QLineEdit(self)
        self.edit.setPlaceholderText(placeholder)
        self.edit.textChanged.connect(self.changed.emit)
        btn = QToolButton(self)
        btn.setText("…")
        btn.setToolTip("Browse")
        btn.clicked.connect(self._browse)
        layout.addWidget(self.edit, 1)
        layout.addWidget(btn, 0)

    def value(self) -> str:
        return self.edit.text().strip()

    def path(self) -> Path | None:
        v = self.value()
        return Path(v).expanduser() if v else None

    def set_value(self, v: str | Path) -> None:
        self.edit.setText(str(v))

    def _browse(self) -> None:
        if self.mode == "open-dir":
            d = QFileDialog.getExistingDirectory(self, "Select directory", self.edit.text() or "")
            if d:
                self.set_value(d)
        elif self.mode == "save-file":
            f, _ = QFileDialog.getSaveFileName(self, "Save", self.edit.text() or "", self.filter_)
            if f:
                self.set_value(f)
        else:
            f, _ = QFileDialog.getOpenFileName(self, "Open", self.edit.text() or "", self.filter_)
            if f:
                self.set_value(f)


class ConsoleView(QPlainTextEdit):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(5000)
        font = QFont("Monospace")
        font.setStyleHint(QFont.StyleHint.TypeWriter)
        self.setFont(font)
        self.setObjectName("Console")

    def append_line(self, text: str, color: str | None = None) -> None:
        if color:
            html = f'<span style="color:{color};">{text.replace("<", "&lt;").replace(">", "&gt;")}</span>'
            self.appendHtml(html)
        else:
            self.appendPlainText(text)
        self.moveCursor(QTextCursor.MoveOperation.End)


def hsep() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setFrameShadow(QFrame.Shadow.Sunken)
    return f


def big_button(label: str, primary: bool = False) -> QPushButton:
    b = QPushButton(label)
    b.setMinimumHeight(36)
    b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    if primary:
        b.setProperty("kind", "primary")
    return b
