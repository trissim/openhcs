"""Browsable OpenHCS help backed by the canonical agent knowledge service."""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import QSignalBlocker, Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from openhcs.agent.dto.knowledge import (
    KnowledgeBaseCatalog,
    KnowledgeBaseDocument,
    KnowledgeBaseDocumentRequest,
    KnowledgeBaseDocumentSummary,
    KnowledgeBaseDocumentTarget,
    KnowledgeBaseSearchHit,
    KnowledgeBaseSearchRequest,
    KnowledgeBaseSearchResult,
)
from openhcs.agent.services.knowledge_base_service import KnowledgeBaseService
from pyqt_reactive.theming import ColorScheme, StyleSheetGenerator


KNOWLEDGE_ITEM_ROLE = Qt.ItemDataRole.UserRole
KNOWLEDGE_DOCUMENT_MAX_CHARS = 50_000


@dataclass(frozen=True, slots=True)
class KnowledgeDocumentSelection:
    """Exact knowledge document/section identity carried by one UI row."""

    document_id: str
    section_id: str | None = None

    @classmethod
    def from_summary(
        cls,
        summary: KnowledgeBaseDocumentSummary,
    ) -> "KnowledgeDocumentSelection":
        return cls(document_id=summary.document_id)

    @classmethod
    def from_search_hit(
        cls,
        hit: KnowledgeBaseSearchHit,
    ) -> "KnowledgeDocumentSelection":
        return cls(
            document_id=hit.document.document_id,
            section_id=None if hit.section is None else hit.section.section_id,
        )


class HelpWindow(QDialog):
    """Browse and search the same source-backed knowledge available to MCP agents."""

    def __init__(
        self,
        main_window=None,
        service_adapter=None,
        *,
        knowledge_service: KnowledgeBaseService | None = None,
        color_scheme: ColorScheme | None = None,
        parent=None,
    ) -> None:
        parent_widget = parent or main_window
        super().__init__(parent_widget)
        self.main_window = main_window
        self.service_adapter = service_adapter
        self.knowledge_service = knowledge_service or KnowledgeBaseService()
        self.color_scheme = color_scheme or self._resolved_color_scheme()
        self.style_generator = StyleSheetGenerator(self.color_scheme)
        self.catalog: KnowledgeBaseCatalog | None = None
        self.search_result: KnowledgeBaseSearchResult | None = None
        self.current_document: KnowledgeBaseDocument | None = None
        self._active_document_id: str | None = None
        self._updating_sections = False

        self.setWindowTitle("OpenHCS Knowledge Base")
        self.setModal(False)
        self.setMinimumSize(760, 520)
        self.resize(980, 680)
        self._setup_ui()
        self._load_catalog()

    def _resolved_color_scheme(self) -> ColorScheme:
        if self.service_adapter is None:
            return ColorScheme()
        return self.service_adapter.get_current_color_scheme()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        title = QLabel("OpenHCS Knowledge Base", self)
        title.setObjectName("knowledge_title")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        search_row = QHBoxLayout()
        self.search_input = QLineEdit(self)
        self.search_input.setObjectName("knowledge_search_input")
        self.search_input.setPlaceholderText("Search documentation")
        self.search_input.setClearButtonEnabled(True)
        self.search_input.returnPressed.connect(self._search)
        search_row.addWidget(self.search_input, 1)

        self.search_button = QPushButton("Search", self)
        self.search_button.setObjectName("knowledge_search_button")
        self.search_button.setToolTip("Search the OpenHCS knowledge base")
        self.search_button.clicked.connect(self._search)
        search_row.addWidget(self.search_button)

        self.browse_button = QToolButton(self)
        self.browse_button.setObjectName("knowledge_browse_button")
        self.browse_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogResetButton)
        )
        self.browse_button.setToolTip("Clear search and browse all documents")
        self.browse_button.clicked.connect(self._show_catalog)
        search_row.addWidget(self.browse_button)
        layout.addLayout(search_row)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.knowledge_index = QListWidget(splitter)
        self.knowledge_index.setObjectName("knowledge_index")
        self.knowledge_index.setMinimumWidth(260)
        self.knowledge_index.currentItemChanged.connect(
            self._open_selected_item
        )

        content_panel = QWidget(splitter)
        content_layout = QVBoxLayout(content_panel)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(4)

        self.document_title = QLabel(content_panel)
        self.document_title.setObjectName("knowledge_document_title")
        self.document_title.setWordWrap(True)
        self.document_title.setStyleSheet("font-size: 14px; font-weight: bold;")
        content_layout.addWidget(self.document_title)

        self.section_selector = QComboBox(content_panel)
        self.section_selector.setObjectName("knowledge_section_selector")
        self.section_selector.currentIndexChanged.connect(self._section_changed)
        content_layout.addWidget(self.section_selector)

        self.document_content = QPlainTextEdit(content_panel)
        self.document_content.setObjectName("knowledge_document_content")
        self.document_content.setReadOnly(True)
        self.document_content.setLineWrapMode(
            QPlainTextEdit.LineWrapMode.WidgetWidth
        )
        content_layout.addWidget(self.document_content, 1)

        splitter.addWidget(self.knowledge_index)
        splitter.addWidget(content_panel)
        splitter.setSizes((300, 680))
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)

        footer = QHBoxLayout()
        self.status_label = QLabel(self)
        self.status_label.setObjectName("knowledge_status")
        footer.addWidget(self.status_label, 1)
        close_button = QPushButton("Close", self)
        close_button.setObjectName("knowledge_close_button")
        close_button.clicked.connect(self.close)
        footer.addWidget(close_button)
        layout.addLayout(footer)

        self.setStyleSheet(self.style_generator.generate_dialog_style())

    def _load_catalog(self) -> None:
        self.catalog = self.knowledge_service.list_documents()
        self._show_catalog()

    def open_target(self, target: KnowledgeBaseDocumentTarget) -> None:
        """Show one exact canonical document/section and select its catalog row."""
        self._show_catalog()
        matching_row = None
        for row in range(self.knowledge_index.count()):
            item = self.knowledge_index.item(row)
            selection = item.data(KNOWLEDGE_ITEM_ROLE)
            if (
                isinstance(selection, KnowledgeDocumentSelection)
                and selection.document_id == target.document_id
            ):
                matching_row = row
                break
        if matching_row is None:
            raise ValueError(
                f"Unknown knowledge-base document target {target.document_id!r}"
            )

        blocker = QSignalBlocker(self.knowledge_index)
        self.knowledge_index.setCurrentRow(matching_row)
        del blocker
        self._load_document(
            KnowledgeDocumentSelection(
                document_id=target.document_id,
                section_id=target.section_id,
            )
        )

    def _show_catalog(self) -> None:
        self.search_input.clear()
        self.search_result = None
        if self.catalog is None:
            self.catalog = self.knowledge_service.list_documents()
        self.knowledge_index.clear()
        for document in self.catalog.documents:
            item = QListWidgetItem(document.title)
            item.setToolTip(document.summary)
            item.setData(
                KNOWLEDGE_ITEM_ROLE,
                KnowledgeDocumentSelection.from_summary(document),
            )
            self.knowledge_index.addItem(item)
        self.status_label.setText(self._catalog_status(self.catalog))
        if self.knowledge_index.count():
            self.knowledge_index.setCurrentRow(0)
        else:
            self._show_message("No knowledge documents are available.")

    def _search(self) -> None:
        query = self.search_input.text().strip()
        if not query:
            self._show_catalog()
            return
        self.search_result = self.knowledge_service.search(
            KnowledgeBaseSearchRequest.from_fields(query=query)
        )
        self.knowledge_index.clear()
        for hit in self.search_result.hits:
            item = QListWidgetItem(self._search_hit_title(hit))
            item.setToolTip(hit.snippet)
            item.setData(
                KNOWLEDGE_ITEM_ROLE,
                KnowledgeDocumentSelection.from_search_hit(hit),
            )
            self.knowledge_index.addItem(item)
        self.status_label.setText(self._search_status(self.search_result))
        if self.knowledge_index.count():
            self.knowledge_index.setCurrentRow(0)
        else:
            self._show_message(f"No results for {query!r}.")

    def _open_selected_item(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        if current is None:
            return
        selection = current.data(KNOWLEDGE_ITEM_ROLE)
        if not isinstance(selection, KnowledgeDocumentSelection):
            raise TypeError("Knowledge index item has no typed document selection")
        self._load_document(selection)

    def _load_document(self, selection: KnowledgeDocumentSelection) -> None:
        document = self.knowledge_service.get_document(
            KnowledgeBaseDocumentRequest.from_fields(
                document_id=selection.document_id,
                section_id=selection.section_id,
                max_chars=KNOWLEDGE_DOCUMENT_MAX_CHARS,
            )
        )
        self.current_document = document
        self._active_document_id = selection.document_id
        self._render_document(document)
        self._populate_sections(document, selection.section_id)

    def _section_changed(self, _index: int) -> None:
        if self._updating_sections or self._active_document_id is None:
            return
        section_id = self.section_selector.currentData()
        if section_id is not None and not isinstance(section_id, str):
            raise TypeError("Knowledge section selector contains a non-string id")
        self._load_document(
            KnowledgeDocumentSelection(
                document_id=self._active_document_id,
                section_id=section_id,
            )
        )

    def _populate_sections(
        self,
        document: KnowledgeBaseDocument,
        selected_section_id: str | None,
    ) -> None:
        self._updating_sections = True
        try:
            self.section_selector.clear()
            self.section_selector.addItem("All sections", None)
            selected_index = 0
            for section in document.sections:
                self.section_selector.addItem(section.title, section.section_id)
                if section.section_id == selected_section_id:
                    selected_index = self.section_selector.count() - 1
            self.section_selector.setCurrentIndex(selected_index)
        finally:
            self._updating_sections = False

    def _render_document(self, document: KnowledgeBaseDocument) -> None:
        if document.errors:
            self.document_title.setText("Knowledge request failed")
            self.document_content.setPlainText(
                "\n".join(error.message for error in document.errors)
            )
            return
        title = "Knowledge document"
        if document.document is not None:
            title = document.document.title
        if document.selected_section_id is not None:
            section = next(
                (
                    candidate
                    for candidate in document.sections
                    if candidate.section_id == document.selected_section_id
                ),
                None,
            )
            if section is not None:
                title = f"{title} / {section.title}"
        self.document_title.setText(title)
        self.document_content.setPlainText(document.content)
        if document.truncated:
            self.status_label.setText(
                f"Showing the first {document.max_chars:,} characters."
            )

    def _show_message(self, message: str) -> None:
        self.current_document = None
        self._active_document_id = None
        self.document_title.setText("OpenHCS Knowledge Base")
        self.document_content.setPlainText(message)
        self._updating_sections = True
        try:
            self.section_selector.clear()
        finally:
            self._updating_sections = False

    @staticmethod
    def _search_hit_title(hit: KnowledgeBaseSearchHit) -> str:
        if hit.section is None:
            return hit.document.title
        return f"{hit.document.title} / {hit.section.title}"

    @staticmethod
    def _catalog_status(catalog: KnowledgeBaseCatalog) -> str:
        suffix = ""
        if catalog.warnings:
            suffix = f"; {len(catalog.warnings)} unavailable"
        return f"{len(catalog.documents)} documents{suffix}"

    @staticmethod
    def _search_status(result: KnowledgeBaseSearchResult) -> str:
        if result.errors:
            return result.errors[0].message
        return f"{len(result.hits)} results for {result.query!r}"
