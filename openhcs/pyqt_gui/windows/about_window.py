"""Native About window for the OpenHCS desktop application."""

from __future__ import annotations

import platform

from PyQt6.QtCore import PYQT_VERSION_STR, QT_VERSION_STR, QSize, Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QLabel,
    QVBoxLayout,
)

from openhcs import __version__ as OPENHCS_VERSION
from openhcs.agent.ui_bridge_identities import AboutOpenHCSWindowIdentity
from openhcs.pyqt_gui.branding import openhcs_brand_pixmap
from openhcs.resources.brand import BrandAsset


class AboutOpenHCSWindow(QDialog):
    """Present package-owned identity, links, and runtime version details."""

    def __init__(self, main_window=None, service_adapter=None) -> None:
        super().__init__(main_window)
        self.main_window = main_window
        self.service_adapter = service_adapter
        self.setObjectName("about_openhcs_window")
        self.setWindowTitle(AboutOpenHCSWindowIdentity.require_title())
        self.setModal(False)
        self.setMinimumWidth(440)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 20)
        layout.setSpacing(10)

        logo = QLabel(self)
        logo.setObjectName("about_openhcs_logo")
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo.setPixmap(
            openhcs_brand_pixmap(
                BrandAsset.MARK,
                QSize(132, 112),
            )
        )
        layout.addWidget(logo)

        title = QLabel("OpenHCS", self)
        title.setObjectName("about_openhcs_title")
        title_font = QFont(title.font())
        title_font.setPointSize(title_font.pointSize() + 8)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        version = QLabel(f"Version {OPENHCS_VERSION}", self)
        version.setObjectName("about_openhcs_version")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        version.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(version)

        description = QLabel(
            "Open High-Content Screening\n"
            "Composable image-analysis pipelines for microscopy experiments.",
            self,
        )
        description.setObjectName("about_openhcs_description")
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        description.setWordWrap(True)
        layout.addWidget(description)

        links = QLabel(
            '<a href="https://openhcs.readthedocs.io/">Documentation</a>'
            " &nbsp;·&nbsp; "
            '<a href="https://github.com/OpenHCSDev/OpenHCS">Source code</a>'
            " &nbsp;·&nbsp; "
            '<a href="https://github.com/OpenHCSDev/OpenHCS/issues">Report an issue</a>',
            self,
        )
        links.setObjectName("about_openhcs_links")
        links.setAlignment(Qt.AlignmentFlag.AlignCenter)
        links.setOpenExternalLinks(True)
        layout.addWidget(links)

        separator = QFrame(self)
        separator.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(separator)

        runtime = QLabel(
            "\n".join(
                (
                    f"Python {platform.python_version()}",
                    f"Qt {QT_VERSION_STR} · PyQt {PYQT_VERSION_STR}",
                    f"{platform.system()} {platform.release()} · {platform.machine()}",
                    "MIT License · Copyright © 2026 Tristan Simas",
                )
            ),
            self,
        )
        runtime.setObjectName("about_openhcs_runtime")
        runtime.setAlignment(Qt.AlignmentFlag.AlignCenter)
        runtime.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(runtime)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        buttons.setObjectName("about_openhcs_buttons")
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)
