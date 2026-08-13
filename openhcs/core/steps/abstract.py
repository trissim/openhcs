"""
Abstract Step Interface

This module defines the AbstractStep interface, which is the base class for all steps
in the OpenHCS pipeline. It provides the core functionality for step execution,
validation, and state management.
"""

import abc
import inspect
from dataclasses import is_dataclass
from typing import TYPE_CHECKING, get_type_hints

from objectstate import get_base_type_for_lazy, semantic_values_equal

from openhcs.constants.input_source import InputSource as InputSource

# Import LazyStepMaterializationConfig for type hints
from openhcs.core.config import LazyStepMaterializationConfig, LazyStreamingDefaults, LazyNapariStreamingConfig, LazyFijiStreamingConfig
from openhcs.core.config import LazyStepWellFilterConfig
from openhcs.core.config import LazyProcessingConfig, LazyDtypeConfig
from openhcs.core.config import LazyStepSourceBindingsConfig
from openhcs.core.source_bindings import (
    StepSourceBindingsConfig,
)

# ProcessingContext is used in type hints
if TYPE_CHECKING:
    from openhcs.core.context.processing_context import ProcessingContext


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

    ``processing_config.input_source`` controls the ordinary main-flow source:

    - InputSource.PREVIOUS_STEP (default): Standard pipeline chaining where the step
      consumes the previous step's main-flow result. This maintains normal
      sequential data flow.

    - InputSource.PIPELINE_START: The step consumes pipeline-start main flow,
      bypassing previous step results. This replaces the @chain_breaker decorator
      functionality and is used for position generation and quality control.

    Separately named inputs are callable artifact declarations satisfied through
    source bindings or prior artifact producers; they are not another
    ``InputSource`` value.

    Usage Examples:

    Standard processing step (default):
    ```python
    step = FunctionStep(
        func=my_processing_function,
        name="process_images",
        processing_config=LazyProcessingConfig(
            input_source=InputSource.PREVIOUS_STEP,
        ),
    )
    ```

    Position generation accessing original images:
    ```python
    step = FunctionStep(
        func=ashlar_compute_tile_positions_gpu,
        name="compute_positions",
        processing_config=LazyProcessingConfig(
            input_source=InputSource.PIPELINE_START,
        ),
    )
    ```

    """

    @classmethod
    def config_classes_by_field_name(cls) -> dict[str, type]:
        """Return config kwargs declared by this root's constructor signature."""

        del cls
        type_hints = get_type_hints(AbstractStep.__init__)
        return {
            field_name: declared_type
            for field_name, parameter in inspect.signature(
                AbstractStep.__init__
            ).parameters.items()
            if field_name != "self"
            and parameter.kind is inspect.Parameter.KEYWORD_ONLY
            and isinstance((declared_type := type_hints.get(field_name)), type)
            and is_dataclass(get_base_type_for_lazy(declared_type) or declared_type)
        }

    def declaration_parameters(self) -> dict[str, object]:
        """Return the public values owned by this step declaration."""

        return {
            name: value
            for name, value in vars(self).items()
            if not name.startswith("_")
        }

    def same_declaration(self, other: object) -> bool:
        """Return whether another step is the same complete declaration."""

        return (
            type(self) is type(other)
            and isinstance(other, AbstractStep)
            and semantic_values_equal(
                self.declaration_parameters(),
                other.declaration_parameters(),
            )
        )

    def occurrence_authorities(self) -> tuple[object, ...]:
        """Return ordered authorities for reconciling declaration occurrences."""

        return (type(self),)

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
        source_bindings: 'LazyStepSourceBindingsConfig' = LazyStepSourceBindingsConfig(),
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
            dtype_config: Output dtype policy for this step's decorated callable.
                Unset fields inherit the pipeline dtype policy.
            processing_config: Stack-axis components, post-assembly grouping,
                and previous-step versus pipeline-start main-flow source for
                this step. Pipeline-level sequential processing is owned
                separately by SequentialProcessingConfig.
            source_bindings: Named semantic inputs selected for this step from
                pipeline sources or prior step artifacts.
            step_well_filter_config: Wells eligible to execute this step. The
                step policy can narrow but cannot restore wells removed by the
                pipeline execution-domain filter.
            step_materialization_config: Optional persistent checkpoint of this
                step's ordinary main-flow result. Named artifacts have an
                independent persistence plan.
            streaming_defaults: Shared viewer enablement, well selection,
                lifetime, host, and transport defaults inherited by this step's
                registered viewer configurations.
            napari_streaming_config: Napari-specific enablement, display,
                transport, and scope overrides for this step.
            fiji_streaming_config: Fiji-specific enablement, display,
                transport, and scope overrides for this step.
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
