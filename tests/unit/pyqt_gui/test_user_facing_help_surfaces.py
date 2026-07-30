from __future__ import annotations

import ast
import inspect
import textwrap
from pathlib import Path

from PyQt6.QtWidgets import QApplication
from pyqt_reactive.services.help_document import HelpDocumentFormat
from pyqt_reactive.services.window_manager import WindowManager
from pyqt_reactive.widgets.shared.clickable_help_components import HelpButton
from pyqt_reactive.windows.help_window_manager import HelpWindowManager

from openhcs.agent.dto.functions import FunctionCatalogEntry, catalog_page
from openhcs.agent.dto.knowledge import (
    KnowledgeBaseDocumentRequest,
    KnowledgeBaseDocumentSummary,
    KnowledgeBaseDocumentTarget,
    KnowledgeBaseSearchRequest,
)
from openhcs.agent.services.knowledge_base_service import (
    MAX_DOCUMENT_CHARS,
    KnowledgeBaseDocumentSpec,
    KnowledgeBaseService,
)
from objectstate.lazy_factory import ensure_global_config_context
from objectstate.object_state import ObjectStateRegistry
from openhcs.core.config import GlobalPipelineConfig
from openhcs.pyqt_gui.config import get_default_ui_config
from openhcs.pyqt_gui.services.main_window_workflows import build_main_window_specs
from openhcs.pyqt_gui.services.ui_window_ids import OpenHCSUiWindowId
from openhcs.pyqt_gui.widgets.pipeline_editor import PipelineEditorWidget
from openhcs.pyqt_gui.widgets.plate_manager import PlateManagerWidget
from openhcs.pyqt_gui.widgets.shared.openhcs_manager_mixins import (
    OpenHCSSingleRowActionManagerMixin,
)
from openhcs.pyqt_gui.windows.help_window import (
    KNOWLEDGE_ITEM_ROLE,
    FunctionDocumentSelection,
    HelpWindow,
    KnowledgeDocumentSelection,
)
from tests.unit.pyqt_gui.test_pipeline_editor_widget import (
    PipelineEditorServiceStub,
    QtApplicationHarness,
)
from tests.unit.pyqt_gui.test_plate_manager_widget import (
    PlateManagerServiceStub,
    close_widget,
)


def _document_spec(
    document_id: str,
    title: str,
    source_path: str,
) -> KnowledgeBaseDocumentSpec:
    return KnowledgeBaseDocumentSpec(
        KnowledgeBaseDocumentSummary(
            document_id=document_id,
            title=title,
            summary=f"{title} summary.",
            source_path=source_path,
            tags=("test",),
            section_count=0,
        )
    )


def _knowledge_service(tmp_path: Path) -> KnowledgeBaseService:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "compilation.md").write_text(
        "# Pipeline Compilation\n\nCompilation validates ObjectState-backed steps.\n",
        encoding="utf-8",
    )
    (docs / "configuration.md").write_text(
        "# Configuration\n\nConfiguration fields retain declaration help.\n",
        encoding="utf-8",
    )
    (docs / "interface.md").write_text(
        "# Desktop Interface\n\n"
        "## Plate Manager\n\nManage datasets and lifecycle operations.\n\n"
        "## Pipeline Editor\n\nBuild and edit ordered processing steps.\n",
        encoding="utf-8",
    )
    return KnowledgeBaseService(
        repo_root=tmp_path,
        document_specs=(
            _document_spec(
                "pipeline_compilation",
                "Pipeline Compilation",
                "docs/compilation.md",
            ),
            _document_spec(
                "configuration",
                "Configuration",
                "docs/configuration.md",
            ),
            _document_spec(
                "openhcs_basic_interface",
                "Desktop Interface",
                "docs/interface.md",
            ),
        ),
    )


def _help_probe(image, threshold: float = 0.5):
    """Segment bright objects in an image.

    Args:
        image: Input image plane.
        threshold: Intensity threshold used for segmentation.

    Returns:
        Labeled objects.

    Examples:
        labels = help_probe(image, threshold=0.75)
    """

    return image > threshold


class _FunctionCatalogService:
    function_id = "openhcs:test_help_probe"

    def __init__(self) -> None:
        self.search_queries: list[str | None] = []
        self.catalog_calls = 0
        self.resolved_ids: list[str] = []
        self.entry = FunctionCatalogEntry(
            function_id=self.function_id,
            import_path=f"{__name__}._help_probe",
            name="help_probe",
            module=__name__,
            library="openhcs",
            signature="help_probe(threshold=0.5)",
            summary="Segment bright objects in an image.",
            backend_tags=("test",),
        )

    def search(
        self,
        *,
        query: str | None = None,
        library: str | None = None,
        limit: int = 50,
        compact_signatures: bool = False,
    ):
        del compact_signatures
        self.search_queries.append(query)
        matches = query is None or query.casefold() in (
            f"{self.entry.name} {self.entry.summary}".casefold()
        )
        items = (self.entry,) if matches and library in (None, "openhcs") else ()
        return catalog_page(
            items=items[:limit],
            total=len(items),
            limit=limit,
            query=query,
            library=library,
        )

    def catalog(self, *, compact_signatures: bool = False):
        del compact_signatures
        self.catalog_calls += 1
        return catalog_page(
            items=(self.entry,),
            total=1,
            limit=1,
            query=None,
            library=None,
        )

    def resolve(self, function_id: str):
        self.resolved_ids.append(function_id)
        if function_id != self.function_id:
            raise ValueError(function_id)
        return _help_probe


def test_help_window_projects_exact_canonical_catalog_search_and_document(
    tmp_path: Path,
) -> None:
    QtApplicationHarness.app()
    service = _knowledge_service(tmp_path)
    function_catalog = _FunctionCatalogService()
    window = HelpWindow(
        knowledge_service=service,
        function_catalog_service=function_catalog,
    )

    try:
        assert window.catalog == service.list_documents()
        assert window.knowledge_index.count() == len(window.catalog.documents)

        query = "ObjectState compilation"
        window.search_input.setText(query)
        window.search_button.click()
        QApplication.processEvents()

        request = KnowledgeBaseSearchRequest.from_fields(query=query)
        assert window.search_result == service.search(request)
        item = window.knowledge_index.currentItem()
        assert item is not None
        selection = item.data(KNOWLEDGE_ITEM_ROLE)
        assert isinstance(selection, KnowledgeDocumentSelection)
        expected_document = service.get_document(
            KnowledgeBaseDocumentRequest.from_fields(
                document_id=selection.document_id,
                section_id=selection.section_id,
                max_chars=MAX_DOCUMENT_CHARS,
            )
        )
        assert window.current_document == expected_document
        assert "Compilation validates ObjectState-backed steps." in (
            window.document_content.toPlainText()
        )
        assert window.document_content.current_document is not None
        assert function_catalog.search_queries == [query]
    finally:
        window.close()


def test_help_window_resolves_registered_function_through_shared_renderer(
    tmp_path: Path,
) -> None:
    QtApplicationHarness.app()
    function_catalog = _FunctionCatalogService()
    window = HelpWindow(
        knowledge_service=_knowledge_service(tmp_path),
        function_catalog_service=function_catalog,
    )

    try:
        window.help_indexes.setCurrentWidget(window.function_index)
        QApplication.processEvents()

        assert window.function_catalog is not None
        assert window.function_catalog.total == 1
        assert window.function_index.count() == 1
        assert function_catalog.catalog_calls == 1
        assert function_catalog.search_queries == []

        item = window.function_index.currentItem()
        assert item is not None
        selection = item.data(KNOWLEDGE_ITEM_ROLE)
        assert isinstance(selection, FunctionDocumentSelection)
        assert selection.function_id == function_catalog.function_id
        assert window.current_function_id == function_catalog.function_id
        assert function_catalog.resolved_ids == [function_catalog.function_id]

        rendered_text = window.document_content.toPlainText()
        assert "help_probe" in rendered_text
        assert "Parameters" in rendered_text
        assert "threshold" in rendered_text
        assert "Intensity threshold used for segmentation." in rendered_text
        assert "Examples" in rendered_text
        assert window.document_content.current_document is not None
        assert window.document_content.current_document.content
        assert not window.section_selector.isEnabled()
        assert window.section_selector.isHidden()

        window.help_indexes.setCurrentWidget(window.knowledge_index)
        QApplication.processEvents()

        assert window.current_document is not None
        assert window.current_function_id is None
        assert window.section_selector.isEnabled()
        assert not window.section_selector.isHidden()
        assert "Compilation validates" in window.document_content.toPlainText()
    finally:
        window.close()


def test_help_window_derives_rst_rendering_from_source_authority(
    tmp_path: Path,
) -> None:
    QtApplicationHarness.app()
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.rst").write_text(
        "Analysis Guide\n"
        "==============\n\n"
        "Use the `OpenHCS documentation <https://openhcs.org>`_.\n\n"
        "Example\n"
        "-------\n\n"
        ".. code-block:: python\n\n"
        "   result = analyze(image)\n",
        encoding="utf-8",
    )
    service = KnowledgeBaseService(
        repo_root=tmp_path,
        document_specs=(
            _document_spec("analysis_guide", "Analysis Guide", "docs/guide.rst"),
        ),
    )
    window = HelpWindow(
        knowledge_service=service,
        function_catalog_service=_FunctionCatalogService(),
    )

    try:
        rendered = window.document_content.current_document
        assert rendered is not None
        assert rendered.markup is HelpDocumentFormat.RESTRUCTURED_TEXT
        assert rendered.base_url == str((docs / "guide.rst").parent)
        assert "Analysis Guide" not in rendered.content
        assert "OpenHCS documentation" in window.document_content.toPlainText()
        assert "result = analyze(image)" in window.document_content.toPlainText()
    finally:
        window.close()


def test_knowledge_window_is_registered_under_canonical_ui_window_id() -> None:
    spec = build_main_window_specs()[OpenHCSUiWindowId.knowledge_base]

    assert spec.window_class is HelpWindow
    assert spec.title == "OpenHCS Knowledge Base"
    assert spec.initialize_on_startup is False


def test_knowledge_window_opens_exact_manager_document_sections(
    tmp_path: Path,
) -> None:
    QtApplicationHarness.app()
    service = _knowledge_service(tmp_path)
    window = HelpWindow(
        knowledge_service=service,
        function_catalog_service=_FunctionCatalogService(),
    )

    try:
        for section_id in ("plate-manager", "pipeline-editor"):
            target = KnowledgeBaseDocumentTarget(
                document_id="openhcs_basic_interface",
                section_id=section_id,
            )
            window.open_target(target)

            selected_item = window.knowledge_index.currentItem()
            assert selected_item is not None
            selection = selected_item.data(KNOWLEDGE_ITEM_ROLE)
            assert isinstance(selection, KnowledgeDocumentSelection)
            assert selection.document_id == target.document_id
            assert window.current_document == service.get_document(
                KnowledgeBaseDocumentRequest.from_fields(
                    document_id=target.document_id,
                    section_id=target.section_id,
                    max_chars=MAX_DOCUMENT_CHARS,
                )
            )
            assert window.section_selector.currentData() == section_id
    finally:
        window.close()


def test_pipeline_and_plate_manager_help_buttons_open_managed_knowledge_window(
    tmp_path: Path,
    monkeypatch,
) -> None:
    QtApplicationHarness.app()
    ObjectStateRegistry.clear()
    knowledge_service = _knowledge_service(tmp_path)
    function_catalog = _FunctionCatalogService()
    calls: list[tuple[str, bool]] = []

    class ManagedKnowledgeMainWindow:
        def show_window(
            self,
            window_id: str,
            hide_if_startup: bool = True,
        ) -> None:
            calls.append((window_id, hide_if_startup))
            spec = build_main_window_specs()[window_id]
            WindowManager.show_or_focus(
                window_id,
                lambda: spec.window_class(
                    knowledge_service=knowledge_service,
                    function_catalog_service=function_catalog,
                ),
            )

    main_window = ManagedKnowledgeMainWindow()
    monkeypatch.setattr(
        HelpWindowManager,
        "show_docstring_help",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Manager Help must not open class-docstring help")
        ),
    )
    pipeline_service = PipelineEditorServiceStub()
    pipeline_service.main_window = main_window
    pipeline = PipelineEditorWidget(pipeline_service)
    plate_service = PlateManagerServiceStub()
    plate_service.main_window = main_window
    ensure_global_config_context(GlobalPipelineConfig, plate_service.global_config)
    plate = PlateManagerWidget(
        plate_service,
        gui_config=get_default_ui_config(),
    )

    try:
        assert isinstance(pipeline.context_help_button, HelpButton)
        assert isinstance(plate.context_help_button, HelpButton)
        assert pipeline.context_help_button.objectName() == (
            "pipeline_editor_help_button"
        )
        assert plate.context_help_button.objectName() == "plate_manager_help_button"
        assert pipeline.button_panel.get_button("pipeline_editor_help_button") is None
        assert plate.button_panel.get_button("plate_manager_help_button") is None
        assert pipeline.title_layout._help_widget is pipeline.context_help_button
        assert plate.title_layout._help_widget is plate.context_help_button
        assert (
            pipeline.title_layout._title_layout.indexOf(
                pipeline.context_help_button
            )
            >= 0
        )
        assert (
            plate.title_layout._title_layout.indexOf(plate.context_help_button)
            >= 0
        )

        pipeline.context_help_button.click()
        QApplication.processEvents()
        pipeline_help = WindowManager.get_window(OpenHCSUiWindowId.knowledge_base)
        assert isinstance(pipeline_help, HelpWindow)
        assert pipeline_help.knowledge_service is knowledge_service
        assert pipeline_help.catalog == knowledge_service.list_documents()
        assert pipeline_help.current_document is not None
        assert pipeline_help.current_document.document is not None
        assert pipeline_help.current_document.document.document_id == (
            PipelineEditorWidget.HELP_KNOWLEDGE_TARGET.document_id
        )
        assert pipeline_help.current_document.selected_section_id == (
            PipelineEditorWidget.HELP_KNOWLEDGE_TARGET.section_id
        )
        pipeline_help.close()
        QApplication.processEvents()

        plate.context_help_button.click()
        QApplication.processEvents()
        plate_help = WindowManager.get_window(OpenHCSUiWindowId.knowledge_base)
        assert isinstance(plate_help, HelpWindow)
        assert plate_help.knowledge_service is knowledge_service
        assert plate_help.catalog == knowledge_service.list_documents()
        assert plate_help.current_document is not None
        assert plate_help.current_document.document is not None
        assert plate_help.current_document.document.document_id == (
            PlateManagerWidget.HELP_KNOWLEDGE_TARGET.document_id
        )
        assert plate_help.current_document.selected_section_id == (
            PlateManagerWidget.HELP_KNOWLEDGE_TARGET.section_id
        )

        assert calls == [
            (OpenHCSUiWindowId.knowledge_base, False),
            (OpenHCSUiWindowId.knowledge_base, False),
        ]
    finally:
        managed_help = WindowManager.get_window(OpenHCSUiWindowId.knowledge_base)
        if managed_help is not None:
            managed_help.close()
        WindowManager.unregister(OpenHCSUiWindowId.knowledge_base)
        pipeline.close()
        close_widget(plate)
        ObjectStateRegistry.clear()


def test_context_help_installation_delegates_to_title_composition() -> None:
    source = textwrap.dedent(
        inspect.getsource(
            OpenHCSSingleRowActionManagerMixin.install_context_help_button
        )
    )
    tree = ast.parse(source)
    attribute_names = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }

    assert "set_help_widget" in attribute_names
    assert {
        "add_button",
        "parentWidget",
        "layout",
        "columnCount",
    }.isdisjoint(attribute_names)


def test_context_help_uses_typed_managed_window_authority() -> None:
    source = textwrap.dedent(inspect.getsource(OpenHCSSingleRowActionManagerMixin))
    tree = ast.parse(source)
    attribute_names = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    function_names = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }

    assert "show_managed_help" in function_names
    assert {"show_window", "knowledge_base"}.issubset(attribute_names)
    assert "help_target" not in source
    assert "callback" not in source.casefold()
    assert {"parentWidget", "parent", "layout", "columnCount"}.isdisjoint(
        attribute_names
    )
