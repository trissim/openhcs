import openhcs.serialization.pycodify_formatters  # noqa: F401
from pycodify import Assignment, generate_python_source
from pycodify import BlankLine
from pycodify import CodeBlock

from openhcs.config_framework import ObjectStateRegistry
from openhcs.config_framework.object_state import ObjectState
from openhcs.constants.constants import VariableComponents
from openhcs.core.config import GlobalPipelineConfig
from openhcs.core.config import LazyProcessingConfig
from openhcs.core.config import PipelineConfig
from openhcs.core.steps.function_step import FunctionStep
from openhcs.processing.backends.processors.numpy_processor import (
    stack_percentile_normalize,
)
from openhcs.pyqt_gui.widgets.plate_manager import PlateManagerWidget
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


def test_plate_code_serializes_objectstate_materialized_configs():
    ObjectStateRegistry.clear()
    try:
        widget = PlateManagerWidget.__new__(PlateManagerWidget)
        widget.global_config = GlobalPipelineConfig()
        widget.plate_configs = {}

        global_state = ObjectState(
            object_instance=GlobalPipelineConfig(),
            scope_id="",
        )
        ObjectStateRegistry.register(global_state, _skip_snapshot=True)
        global_state.update_parameter("num_workers", 7)

        plate_path = "/tmp/openhcs-test-plate"
        plate_state = ObjectState(
            object_instance=PipelineConfig(),
            scope_id=plate_path,
            parent_state=global_state,
        )
        ObjectStateRegistry.register(plate_state, _skip_snapshot=True)
        plate_state.update_parameter("num_workers", 9)

        assert widget.global_config.num_workers == 1
        assert widget._get_global_config_for_serialization().num_workers == 7
        assert (
            widget._get_pipeline_config_for_serialization(plate_path).num_workers == 9
        )

        python_code = generate_python_source(
            CodeBlock.from_items(
                [
                    Assignment(
                        "global_config",
                        widget._get_global_config_for_serialization(),
                    ),
                    BlankLine(),
                    Assignment(
                        "per_plate_configs",
                        {
                            plate_path: widget._get_pipeline_config_for_serialization(
                                plate_path
                            )
                        },
                    ),
                ]
            ),
            header="# Edit this orchestrator configuration and save to apply changes",
            clean_mode=True,
        )

        assert "global_config = GlobalPipelineConfig(" in python_code
        assert "num_workers=7" in python_code
        assert "per_plate_configs = {" in python_code
        assert "num_workers=9" in python_code
    finally:
        ObjectStateRegistry.clear()
