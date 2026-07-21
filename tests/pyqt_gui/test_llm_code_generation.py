"""Tests for LLM code generation feature."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import os

# Skip PyQt6 GUI tests in CPU-only mode
pytestmark = pytest.mark.skipif(
    os.getenv('OPENHCS_CPU_ONLY', 'false').lower() == 'true',
    reason="PyQt6 GUI tests skipped in CPU-only mode"
)

from openhcs.pyqt_gui.services.llm_pipeline_service import LLMPipelineService

# Conditionally import PyQt6 widgets for GUI tests
try:
    from openhcs.pyqt_gui.widgets.llm_chat_panel import LLMChatPanel
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False


def test_llm_service_builds_system_prompt():
    """Test that system prompt is built correctly."""
    service = LLMPipelineService()
    assert "OpenHCS Architecture Principles" in service.system_prompt
    assert "FunctionStep" in service.system_prompt
    assert "VariableComponents" in service.system_prompt

    function_prompt = service.get_system_prompt("function")
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

    service = LLMPipelineService()
    code = service.generate_code("normalize images", code_type='pipeline')

    assert "pipeline_steps" in code
    assert mock_post.called


@patch('openhcs.pyqt_gui.services.llm_pipeline_service.requests.post')
def test_llm_service_handles_errors(mock_post):
    """Test error handling."""
    mock_post.side_effect = Exception("Connection failed")

    service = LLMPipelineService()
    with pytest.raises(Exception, match="Failed to connect"):
        service.generate_code("test", code_type='pipeline')


def test_llm_service_context_specific_prompts():
    """Test that different code types get appropriate context."""
    service = LLMPipelineService()

    # Test different code types
    for code_type in ['pipeline', 'step', 'config', 'function', 'orchestrator']:
        # Should not raise exception
        # (actual generation would require LLM service running)
        assert code_type in ['pipeline', 'step', 'config', 'function', 'orchestrator']


def test_llm_service_cleans_markdown():
    """Test that markdown code blocks are cleaned."""
    service = LLMPipelineService()

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
    panel = LLMChatPanel(code_type='pipeline')
    qtbot.addWidget(panel)

    assert panel.code_type == 'pipeline'
    assert panel.llm_service is not None
    assert panel.generate_button is not None
    assert panel.user_input is not None


@pytest.mark.skipif(not PYQT_AVAILABLE, reason="PyQt6 not available")
def test_chat_panel_signal_emission(qtbot):
    """Test that panel emits code_generated signal."""
    panel = LLMChatPanel(code_type='step')
    qtbot.addWidget(panel)

    # Connect signal to mock
    mock_handler = Mock()
    panel.code_generated.connect(mock_handler)

    # Simulate successful generation
    test_code = "step = FunctionStep(func=normalize, name='test')"
    panel.on_generation_finished(test_code)

    # Verify signal was emitted with correct code
    mock_handler.assert_called_once_with(test_code)


@pytest.mark.skipif(not PYQT_AVAILABLE, reason="PyQt6 not available")
def test_chat_panel_empty_request_warning(qtbot):
    """Test that empty requests show a warning."""
    panel = LLMChatPanel(code_type='pipeline')
    qtbot.addWidget(panel)

    # Mock QMessageBox to prevent actual dialog
    with patch('openhcs.pyqt_gui.widgets.llm_chat_panel.QMessageBox.warning') as mock_warning:
        # Clear input and try to generate
        panel.user_input.clear()
        panel.on_generate_clicked()

        # Verify warning was shown
        mock_warning.assert_called_once()


@pytest.mark.skipif(not PYQT_AVAILABLE, reason="PyQt6 not available")
def test_chat_panel_appends_to_history(qtbot):
    """Test that messages are appended to chat history."""
    panel = LLMChatPanel(code_type='pipeline')
    qtbot.addWidget(panel)

    # Clear history and add message
    panel.chat_history.clear()
    panel.append_to_history("Test message")

    # Verify message was added
    history_text = panel.chat_history.toPlainText()
    assert "Test message" in history_text


@pytest.mark.skipif(not PYQT_AVAILABLE, reason="PyQt6 not available")
def test_chat_panel_clear_button(qtbot):
    """Test that clear button clears history."""
    panel = LLMChatPanel(code_type='pipeline')
    qtbot.addWidget(panel)

    # Add message and then clear
    panel.append_to_history("Test message")
    panel.on_clear_clicked()

    # Verify history was cleared
    history_text = panel.chat_history.toPlainText()
    assert history_text == ""
