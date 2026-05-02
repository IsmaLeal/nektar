"""DOVelocityCorrectionScheme GUI theme: qdarkstyle base plus custom accents."""
from __future__ import annotations

ACCENT = "#7CC4FF"
ACCENT_HI = "#A6D8FF"
SUCCESS = "#5BD49A"
WARN = "#FFB454"
ERROR = "#FF6F6F"
BG_CARD = "#22272E"
BG_CARD_BORDER = "#2F3641"


CUSTOM_QSS = f"""
QWidget {{
    font-size: 13px;
}}

QGroupBox#Card {{
    border: 1px solid {BG_CARD_BORDER};
    border-radius: 10px;
    margin-top: 14px;
    background-color: {BG_CARD};
    padding: 10px 12px 12px 12px;
}}
QGroupBox#Card::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    color: {ACCENT_HI};
    font-weight: 600;
}}

QPlainTextEdit#Console {{
    background-color: #14181E;
    color: #DDE4EC;
    border: 1px solid {BG_CARD_BORDER};
    border-radius: 8px;
}}

QPushButton[kind="primary"] {{
    background-color: {ACCENT};
    color: #0B1117;
    border: none;
    border-radius: 8px;
    padding: 8px 14px;
    font-weight: 600;
}}
QPushButton[kind="primary"]:hover {{
    background-color: {ACCENT_HI};
}}
QPushButton[kind="primary"]:disabled {{
    background-color: #3a4250;
    color: #8a93a0;
}}

QPushButton[kind="danger"] {{
    background-color: {ERROR};
    color: #0B1117;
    border: none;
    border-radius: 8px;
    padding: 8px 14px;
    font-weight: 600;
}}

QProgressBar {{
    border: 1px solid {BG_CARD_BORDER};
    border-radius: 6px;
    text-align: center;
    background-color: #14181E;
    color: #DDE4EC;
    height: 18px;
}}
QProgressBar::chunk {{
    background-color: {ACCENT};
    border-radius: 6px;
}}

QTabWidget::pane {{
    border: none;
    top: -1px;
}}
QTabBar::tab {{
    padding: 8px 16px;
    margin-right: 2px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    background: #1B2027;
    color: #B3BCC9;
}}
QTabBar::tab:selected {{
    background: {BG_CARD};
    color: {ACCENT_HI};
    border-bottom: 2px solid {ACCENT};
}}

QLabel[kind="hero"] {{
    font-size: 20px;
    font-weight: 700;
    color: {ACCENT_HI};
}}
QLabel[kind="muted"] {{
    color: #8b95a3;
}}
QLabel[kind="ok"]    {{ color: {SUCCESS}; font-weight: 600; }}
QLabel[kind="warn"]  {{ color: {WARN};    font-weight: 600; }}
QLabel[kind="error"] {{ color: {ERROR};   font-weight: 600; }}
"""


def apply(app) -> None:
    try:
        import qdarkstyle  # type: ignore

        app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api="pyqt6") + CUSTOM_QSS)
    except Exception:
        app.setStyleSheet(CUSTOM_QSS)
