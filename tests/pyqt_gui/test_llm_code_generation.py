"""Tests for LLM code generation feature."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import os
from types import FunctionType

from openhcs.agent.dto.functions import FunctionCatalogEntry, catalog_page
from openhcs.core.config import GlobalPipelineConfig
from openhcs.core.pipeline_document import PipelineDocument
from openhcs.core.steps.function_step import FunctionStep
from openhcs.pyqt_gui.services.llm_pipeline_service import (
    CodeDeclarationStrategy,
    LLMPipelineService,
)
from openhcs.ui.shared.plate_manager_code_document import (
    PlateManagerOrchestratorCodePayload,
)

# Skip PyQt6 GUI tests in CPU-only mode
pytestmark = pytest.mark.skipif(
    os.getenv('OPENHCS_CPU_ONLY', 'false').lower() == 'true',
    reason="PyQt6 GUI tests skipped in CPU-only mode"
)

class _FunctionCatalog:
    def search(self, **request):
        entry = FunctionCatalogEntry(
            function_id="test:normalize",
            import_path="test_functions.normalize",
            name="normalize",
            module="test_functions",
            library="test",
            signature="normalize(image)",
            summary="Normalize an image.",
            backend_tags=("numpy",),
        )
        return catalog_page(
            items=(entry,),
            total=1,
            limit=request.get("limit", 50),
            query=request.get("query"),
            library=request.get("library"),
        )


def _service() -> LLMPipelineService:
    return LLMPipelineService(_FunctionCatalog())

# Conditionally import PyQt6 widgets for GUI tests
try:
    from pyqt_reactive.widgets.llm_chat_panel import LLMChatPanel
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False


def test_llm_service_builds_system_prompt():
    """Test that system prompt is built correctly."""
    service = _service()
    assert "OpenHCS Architecture Principles" in service.system_prompt
    assert "FunctionStep" in service.system_prompt
    assert "VariableComponents" in service.system_prompt

    function_prompt = service.get_system_prompt(FunctionType)
    assert "MeasurementsArtifactType" in function_prompt
    assert "ObjectLabelsArtifactType" in function_prompt
    assert "DataclassMeasurementColumnarRows" in function_prompt
    assert '@artifact_outputs(("segmentation_masks"' not in function_prompt


@patch('openhcs.pyqt_gui.services.llm_pipeline_service.requests.post')
def test_llm_service_generates_code(mock_post):
    """Test successful code generation."""
    # Mock LLM response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        'response': 'pipeline_steps = []\n# Generated code'
    }
    mock_response.raise_for_status = Mock()
    mock_post.return_value = mock_response

    service = _service()
    code = service.generate_code("normalize images", PipelineDocument)

    assert "pipeline_steps" in code
    assert mock_post.called


@patch('openhcs.pyqt_gui.services.llm_pipeline_service.requests.post')
def test_llm_service_handles_errors(mock_post):
    """Test error handling."""
    mock_post.side_effect = Exception("Connection failed")

    service = _service()
    with pytest.raises(Exception, match="Failed to connect"):
        service.generate_code("test", PipelineDocument)


def test_llm_service_dispatches_from_existing_declaration_types():
    """Prompt selection uses authored nominal declarations, not kind mirrors."""
    declaration_types = (
        PipelineDocument,
        FunctionStep,
        GlobalPipelineConfig,
        FunctionType,
        PlateManagerOrchestratorCodePayload,
    )
    strategies = tuple(
        CodeDeclarationStrategy.for_declaration_type(declaration_type)
        for declaration_type in declaration_types
    )

    assert len({type(strategy) for strategy in strategies}) == len(declaration_types)
    assert all(strategy.context_suffix() for strategy in strategies)


def test_llm_service_does_not_read_catalog_until_prompt_is_requested():
    class _UnreadCatalog:
        def search(self, **request):
            raise AssertionError(f"unexpected eager catalog read: {request!r}")

    service = LLMPipelineService(_UnreadCatalog())

    assert service._system_prompts == {}


def test_llm_service_cleans_markdown():
    """Test that markdown code blocks are cleaned."""
    service = _service()

    # Test cleaning markdown code blocks
    test_cases = [
        ("```python\ncode here\n```", "code here"),
        ("```\ncode here\n```", "code here"),
        ("code without markdown", "code without markdown"),
    ]

    for input_code, expected_output in test_cases:
        cleaned = service._clean_generated_code(input_code)
        assert cleaned == expected_output


@pytest.mark.skipif(not PYQT_AVAILABLE, reason="PyQt6 not available")
def test_chat_panel_creation(qtbot):
    """Test chat panel can be created."""
    panel = LLMChatPanel(
        declaration_type=PipelineDocument,
        llm_service=_service(),
    )
    qtbot.addWidget(panel)

    assert panel.declaration_type is PipelineDocument
    assert panel.llm_service is not None
    assert panel.generate_button is not None
    assert panel.user_input is not None


@pytest.mark.skipif(not PYQT_AVAILABLE, reason="PyQt6 not available")
def test_chat_panel_signal_emission(qtbot):
    """Test that panel emits code_generated signal."""
    panel = LLMChatPanel(
        declaration_type=FunctionStep,
        llm_service=_service(),
    )
    qtbot.addWidget(panel)

    # Connect signal to mock
    mock_handler = Mock()
    panel.code_generated.connect(mock_handler)

    # Simulate successful generation
    test_code = "step = FunctionStep(func=normalize, name='test')"
    panel._on_generation_success(test_code)
    panel._on_insert_clicked()

    # Verify signal was emitted with correct code
    mock_handler.assert_called_once_with(test_code)


@pytest.mark.skipif(not PYQT_AVAILABLE, reason="PyQt6 not available")
def test_chat_panel_empty_request_warning(qtbot):
    """Test that empty requests show a warning."""
    panel = LLMChatPanel(
        declaration_type=PipelineDocument,
        llm_service=_service(),
    )
    qtbot.addWidget(panel)

    # Mock QMessageBox to prevent actual dialog
    with patch('pyqt_reactive.widgets.llm_chat_panel.QMessageBox.warning') as mock_warning:
        # Clear input and try to generate
        panel.user_input.clear()
        panel._on_generate_clicked()

        # Verify warning was shown
        mock_warning.assert_called_once()


@pytest.mark.skipif(not PYQT_AVAILABLE, reason="PyQt6 not available")
def test_chat_panel_appends_to_history(qtbot):
    """Test that messages are appended to chat history."""
    panel = LLMChatPanel(
        declaration_type=PipelineDocument,
        llm_service=_service(),
    )
    qtbot.addWidget(panel)

    # Clear history and add message
    panel.chat_history.clear()
    panel._chat_appender.append_text("Test message")

    # Verify message was added
    history_text = panel.chat_history.toPlainText()
    assert "Test message" in history_text


@pytest.mark.skipif(not PYQT_AVAILABLE, reason="PyQt6 not available")
def test_chat_panel_clear_button(qtbot):
    """Test that clear button clears history."""
    panel = LLMChatPanel(
        declaration_type=PipelineDocument,
        llm_service=_service(),
    )
    qtbot.addWidget(panel)

    # Add message and then clear
    panel._chat_appender.append_text("Test message")
    panel._on_clear_clicked()

    # Verify history was cleared
    history_text = panel.chat_history.toPlainText()
    assert history_text == ""
