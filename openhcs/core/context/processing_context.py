"""
Processing Context for OpenHCS.

This module defines the ProcessingContext class, which maintains state during pipeline execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from polystore.filemanager import FileManager

from openhcs.core.config import (
    AnalysisConsolidationConfig,
    PlateMetadataConfig,
    StreamingConfig,
)
from openhcs.core.compiled_step_plan import CompiledStepPlan
from openhcs.core.debug import (
    DebugEventSink,
    DebugExecutionContext,
    NO_OP_DEBUG_EVENT_SINK,
)
from openhcs.core.runtime_stores import RuntimeValueStore
from openhcs.core.runtime_pattern_cache import RuntimePatternDiscoveryCache
from openhcs.core.runtime_stack_cache import RuntimeImageStackCache
from openhcs.core.runtime_source_binding_cache import RuntimeSourceBindingContextCache
from openhcs.core.source_workspace_projection import VirtualWorkspaceSourceProjectionCache
from openhcs.core.axis_filter import StepAxisFilterMap
from openhcs.core.steps.function_output_identity import FunctionOutputIdentityCache


@dataclass(frozen=True, slots=True)
class RequiredVisualizer:
    """Streaming visualizer required by a compiled processing context."""

    backend_name: str | None
    config: StreamingConfig

    @property
    def key(self) -> tuple[str, int]:
        return (self.config.viewer_type, self.config.port)

    @property
    def launch_message(self) -> str:
        return f"Launching {self.config.viewer_type} viewer on port {self.config.port}"


class ProcessingContext:
    """
    Maintains state during pipeline execution.

    The ProcessingContext is the canonical owner of all state during pipeline execution.
    After compilation and freezing, it should be treated as immutable by processing steps.

    OWNERSHIP: This class may ONLY be instantiated by PipelineOrchestrator.
    All other components must receive a context instance, never create one.

    Attributes:
        step_plans: Dictionary mapping step indices to compiled execution plans.
        axis_id: Identifier of the multiprocessing axis value being processed.
        filemanager: Instance of FileManager for VFS operations.
        analysis_consolidation_config: Runtime analysis consolidation settings.
        plate_metadata_config: Runtime plate metadata settings.
        auto_add_output_plate_to_plate_manager: Runtime output-plate registration flag.
        pipeline_sequential_mode: Flag indicating pipeline-wide vs step-wide sequential processing.
        pipeline_sequential_combinations: Pre-computed sequential combinations for pipeline-wide mode.
        current_sequential_combination: Active combination during pipeline-wide sequential execution.
        _is_frozen: Internal flag indicating if the context is immutable.
    """

    _is_frozen: bool = False

    def __init__(
        self,
        step_plans: dict[int, CompiledStepPlan] | None = None,
        axis_id: str | None = None,
        filemanager: FileManager | None = None,
        analysis_consolidation_config: AnalysisConsolidationConfig | None = None,
        plate_metadata_config: PlateMetadataConfig | None = None,
        auto_add_output_plate_to_plate_manager: bool = False,
    ):
        """
        Initialize the processing context.

        Args:
            step_plans: Dictionary mapping step indices to compiled execution plans.
            axis_id: Identifier of the multiprocessing axis value being processed.
            filemanager: FileManager instance for VFS operations.
            analysis_consolidation_config: Analysis consolidation runtime settings.
            plate_metadata_config: Plate metadata runtime settings.
            auto_add_output_plate_to_plate_manager: Output-plate registration flag.
        """
        self._is_frozen = False

        if step_plans is None:
            self.step_plans = {}
        else:
            self.step_plans = dict(step_plans)
        self.runtime_value_store = RuntimeValueStore()
        self.runtime_image_stack_cache = RuntimeImageStackCache()
        self.runtime_function_output_identity_cache = FunctionOutputIdentityCache()
        self.runtime_pattern_discovery_cache = RuntimePatternDiscoveryCache()
        self.runtime_source_binding_context_cache = RuntimeSourceBindingContextCache()
        self.runtime_source_workspace_projection_cache = (
            VirtualWorkspaceSourceProjectionCache()
        )
        self.axis_id = axis_id
        self.filemanager = filemanager
        self.microscope_handler = None
        self.input_dir = None
        self.workspace_path = None
        self.plate_path = None
        self.required_visualizers: list[RequiredVisualizer] = []
        self.step_axis_filters: StepAxisFilterMap = {}
        self.metadata_cache: dict[str, dict[str, str | None]] | None = None
        self.analysis_consolidation_config = (
            analysis_consolidation_config
            if analysis_consolidation_config is not None
            else AnalysisConsolidationConfig()
        )
        self.plate_metadata_config = (
            plate_metadata_config
            if plate_metadata_config is not None
            else PlateMetadataConfig()
        )
        self.auto_add_output_plate_to_plate_manager = (
            auto_add_output_plate_to_plate_manager
        )

        self.execution_id = None
        self.plate_id = None
        self.worker_slot = None
        self.owned_wells = None
        self.debug_event_sink = NO_OP_DEBUG_EVENT_SINK

        self.pipeline_sequential_mode = False
        self.pipeline_sequential_combinations = None
        self.current_sequential_combination = None

    def bind_execution_runtime(
        self,
        *,
        execution_id: str,
        plate_id: str,
        worker_slot: str,
        owned_wells: tuple[str, ...],
    ) -> None:
        """Bind worker-owned execution identity after compilation freeze."""

        self.execution_id = execution_id
        self.plate_id = plate_id
        self.worker_slot = worker_slot
        self.owned_wells = list(owned_wells)

    def install_debug_event_sink(self, debug_event_sink: DebugEventSink) -> None:
        """Install the debug sink selected for this execution context."""

        if not isinstance(debug_event_sink, DebugEventSink):
            raise TypeError(
                "debug_event_sink must be DebugEventSink, "
                f"got {type(debug_event_sink).__name__}."
            )
        self.debug_event_sink = debug_event_sink

    def inject_plan(self, step_id: int, plan: CompiledStepPlan) -> None:
        """
        Inject a step plan into the context.

        This method is the canonical way to add step plans to the context during compilation.
        All step configuration must be injected into the context using this method.

        Args:
            step_id: The step index used as the compiled-plan key.
            plan: The step execution plan

        Raises:
            AttributeError: If the context is frozen.
        """
        if self._is_frozen:
            raise AttributeError("Cannot inject plan into a frozen ProcessingContext.")
        self.step_plans[step_id] = plan

    def freeze(self) -> None:
        """
        Freezes the context, making its attributes immutable.

        This should be called after all compilation and plan injection is complete.
        Essential attributes like step_plans, filemanager, and axis_id must be set.

        Raises:
            RuntimeError: If essential attributes are not set before freezing.
        """
        if not self.axis_id:
            raise RuntimeError("Cannot freeze ProcessingContext: 'axis_id' is not set.")
        if self.filemanager is None:
            raise RuntimeError(
                "Cannot freeze ProcessingContext: 'filemanager' is not set."
            )
        if self.step_plans is None:
            raise RuntimeError(
                "Cannot freeze ProcessingContext: 'step_plans' is not set."
            )

        self._is_frozen = True

    def is_frozen(self) -> bool:
        """
        Check if the context is frozen.

        Returns:
            True if the context is frozen, False otherwise.
        """
        return self._is_frozen


DebugExecutionContext.register(ProcessingContext)
