"""DOVelocityCorrectionScheme GUI entry point."""
from __future__ import annotations

import sys

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import QApplication, QMainWindow, QStatusBar, QTabWidget

from py_utils.dovelocitycorrectionscheme_gui import theme
from py_utils.dovelocitycorrectionscheme_gui.tabs.run_tab import RunTab
from py_utils.dovelocitycorrectionscheme_gui.tabs.viz_tab import VizTab


class DomineMain(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DOVelocityCorrectionScheme — DO VCS control")
        self.resize(1320, 860)
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.setTabPosition(QTabWidget.TabPosition.North)
        tabs.addTab(RunTab(), "Run")
        tabs.addTab(VizTab(), "Visualize")
        self.setCentralWidget(tabs)
        sb = QStatusBar()
        sb.showMessage("Ready")
        self.setStatusBar(sb)


def main() -> int:
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName("DOVelocityCorrectionScheme")
    theme.apply(app)
    win = DomineMain()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
