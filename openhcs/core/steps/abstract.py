"""
Abstract Step Interface

This module defines the AbstractStep interface, which is the base class for all steps
in the OpenHCS pipeline. It provides the core functionality for step execution,
validation, and state management.
"""

import abc
import logging
from typing import TYPE_CHECKING, List, Optional

from openhcs.constants.constants import VariableComponents, GroupBy, get_default_variable_components, get_default_group_by
from openhcs.constants.input_source import InputSource

# Import LazyStepMaterializationConfig for type hints
from openhcs.core.config import LazyStepMaterializationConfig, LazyStreamingDefaults, LazyNapariStreamingConfig, LazyFijiStreamingConfig
from openhcs.core.config import LazyStepWellFilterConfig
from openhcs.core.config import LazyProcessingConfig, LazyDtypeConfig
from openhcs.core.source_bindings import (
    EMPTY_SOURCE_BINDINGS,
    StepSourceBindingsConfig,
)

# ProcessingContext is used in type hints
if TYPE_CHECKING:
    from openhcs.core.context.processing_context import ProcessingContext
# StepResult is no longer returned by process()


#def get_step_id(step: 'AbstractStep') -> str:
#    """
#    Generate a stable step ID from a step object reference.
#
#    This function provides a deterministic way to derive a step's ID
#    from its object reference, enabling stateless execution where
#    step objects don't need to store their own IDs as attributes.
#
#    Args:
#        step: The step object to generate an ID for
#
#    Returns:
#        A stable string ID based on the step object's identity
#
#    Note:
#        This uses the same algorithm as step.__init__() to ensure
#        consistency between compilation and execution phases.
#    """
#    return str(id(step))


class AbstractStep(abc.ABC):
    """
    Abstract base class for all steps in the OpenHCS pipeline.

    This class defines the interface that all steps must implement.
    Steps are stateful during pipeline definition and compilation (holding attributes
    like name, input/output memory types, etc.). After compilation, these attributes
    are stripped by the StepAttributeStripper, and the step instances become
    stateless shells. During execution, steps operate solely based on the
    ProcessingContext (which is frozen) and their specific plan within
    context.step_plans.

    Input Source Control:

    The input_source parameter controls where a step reads its input data:

    - InputSource.PREVIOUS_STEP (default): Standard pipeline chaining where the step
      reads from the output directory of the previous step. This maintains normal
      sequential data flow.

    - InputSource.PIPELINE_START: The step reads from the original pipeline input
      directory, bypassing all previous step outputs. This replaces the @chain_breaker
      decorator functionality and is used for position generation and quality control.

    Usage Examples:

    Standard processing step (default):
    ```python
    step = FunctionStep(
        func=my_processing_function,
        name="process_images"
        # input_source defaults to InputSource.PREVIOUS_STEP
    )
    ```

    Position generation accessing original images:
    ```python
    step = FunctionStep(
        func=ashlar_compute_tile_positions_gpu,
        name="compute_positions",
        input_source=InputSource.PIPELINE_START
    )
    ```

    """

    # Attributes like input_memory_type, output_memory_type, etc.,
    # are defined in concrete subclasses (e.g., FunctionStep) as needed.

    def __init__(
        self,
        *,  # Force keyword-only arguments
        name: str = None,
        description: str = None,
        enabled: bool = True,
        debug_pause: bool = False,
        dtype_config: 'LazyDtypeConfig' = LazyDtypeConfig(),
        processing_config: 'LazyProcessingConfig' = LazyProcessingConfig(),
        source_bindings: 'StepSourceBindingsConfig' = EMPTY_SOURCE_BINDINGS,
        step_well_filter_config: 'LazyStepWellFilterConfig' = LazyStepWellFilterConfig(),
        step_materialization_config: 'LazyStepMaterializationConfig' = LazyStepMaterializationConfig(),
        streaming_defaults: 'LazyStreamingDefaults' = LazyStreamingDefaults(),
        napari_streaming_config: 'LazyNapariStreamingConfig' = LazyNapariStreamingConfig(),
        fiji_streaming_config: 'LazyFijiStreamingConfig' = LazyFijiStreamingConfig(),
    ) -> None:
        """
        Initialize a step. These attributes are primarily used during the
        pipeline definition and compilation phase. After compilation, step
        instances are stripped of these attributes by StepAttributeStripper
        to enforce statelessness during execution.

        Args:
            name: Human-readable name for the step. Defaults to class name.
            description: Optional description of what this step does.
            enabled: Whether this step is enabled. Disabled steps are filtered out
                    during pipeline compilation. Defaults to True.
            debug_pause: Whether bounded debug "run to pause" should stop after
                    this step. This is definition-time debug metadata and does
                    not affect normal execution.
            dtype_config: LazyDtypeConfig for dtype conversion behavior in memory type decorators.
            processing_config: LazyProcessingConfig for variable_components, group_by, input_source, and sequential processing.
            source_bindings: StepSourceBindingsConfig for named semantic input bindings.
            step_well_filter_config: LazyStepWellFilterConfig for well filtering.
            step_materialization_config: Optional LazyStepMaterializationConfig for per-step materialized output.
                                   When provided, enables saving materialized copy of step output
                                   to custom location in addition to normal memory backend processing.
                                   Use LazyStepMaterializationConfig() for safe defaults that prevent path collisions.
            napari_streaming_config: Optional LazyNapariStreamingConfig for napari streaming.
                                   When provided, enables real-time streaming to napari viewer.
            fiji_streaming_config: Optional LazyFijiStreamingConfig for Fiji streaming.
                                 When provided, enables real-time streaming to Fiji viewer.
        """
        self.name = name or self.__class__.__name__
        self.description = description
        self.enabled = enabled
        self.debug_pause = debug_pause
        self.dtype_config = dtype_config
        self.processing_config = processing_config
        if not isinstance(source_bindings, StepSourceBindingsConfig):
            raise TypeError(
                "AbstractStep.source_bindings must be StepSourceBindingsConfig, "
                f"got {type(source_bindings).__name__}."
            )
        self.source_bindings = source_bindings
        self.step_well_filter_config = step_well_filter_config
        self.step_materialization_config = step_materialization_config
        self.streaming_defaults = streaming_defaults
        self.napari_streaming_config = napari_streaming_config
        self.fiji_streaming_config = fiji_streaming_config
        self._scope_token: str | None = None

        # Internal compiler hints - set by path planner during compilation
        self.__input_dir__ = None
        self.__output_dir__ = None

        # Generate a stable step_id based on object id at instantiation.
        # This ID is used to link the step object to its plan in the context.
#        self.step_id = str(id(self))

        logger_instance = logging.getLogger(__name__)
        #logger_instance.debug(f"Created step '{self.name}' (type: {self.__class__.__name__}) with ID {self.step_id}")

    @abc.abstractmethod
    def process(self, context: 'ProcessingContext', step_index: int) -> None:
        """
        Process the step with the given context and step index.

        This method must be implemented by all step subclasses.
        During execution, the step instance is stateless. All necessary
        configuration and paths are retrieved from context.step_plans[self.step_id].
        The context itself is frozen and must not be modified.
        Outputs are written to VFS via context.filemanager based on the steps plan.
        This method returns None.

        Args:
            context: The frozen ProcessingContext containing all required fields,
                     including step_plans and filemanager.
        """
        raise NotImplementedError("AbstractStep.process() must be implemented by subclasses")
