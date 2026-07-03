import openhcs.serialization.pycodify_formatters  # noqa: F401
from pycodify import Assignment, generate_python_source

from openhcs.config_framework import ObjectStateRegistry
from openhcs.constants.constants import VariableComponents
from openhcs.core.config import LazyProcessingConfig
from openhcs.core.steps.function_step import FunctionStep
from openhcs.processing.backends.processors.numpy_processor import (
    stack_percentile_normalize,
)
from openhcs.pyqt_gui.widgets.pipeline_editor import PipelineEditorWidget


def test_pipeline_code_serializes_objectstate_materialized_step():
    ObjectStateRegistry.clear()
    try:
        step = FunctionStep(
            func=(
                stack_percentile_normalize,
                {"low_percentile": 0.5, "high_percentile": 99.5},
            ),
            name="Image Enhancement Processing",
        )

        widget = PipelineEditorWidget.__new__(PipelineEditorWidget)
        widget.current_plate = "/tmp/openhcs-test-plate"
        widget.pipeline_steps = [step]

        widget._update_pipeline_steps(widget.current_plate, widget.pipeline_steps)
        pipeline_state = widget._ensure_pipeline_state(widget.current_plate)
        step_scope_id = pipeline_state.parameters["step_scope_ids"][0]
        step_state = ObjectStateRegistry.get_by_scope(step_scope_id)

        step_state.update_parameter(
            "processing_config.variable_components", [VariableComponents.SITE]
        )

        assert step.processing_config == LazyProcessingConfig()
        assert step_state.to_object().processing_config == LazyProcessingConfig(
            variable_components=[VariableComponents.SITE]
        )

        python_code = generate_python_source(
            Assignment("pipeline_steps", widget._get_steps_for_serialization()),
            header="# Edit this pipeline and save to apply changes",
            clean_mode=True,
        )

        assert "processing_config=LazyProcessingConfig(" in python_code
        assert "variable_components=[" in python_code
        assert "VariableComponents.SITE" in python_code
    finally:
        ObjectStateRegistry.clear()
