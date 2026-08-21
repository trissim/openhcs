"""Typed compiled step plans used as compiler/runtime source of truth."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from openhcs.constants.constants import (
    GPU_MEMORY_TYPES,
    MemoryType,
    SequentialComponents,
    VariableComponents,
)
from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactOutputPlan,
    ArtifactSpecRef,
)
from openhcs.core.callable_contract import FunctionStepExecutionScope
from openhcs.core.component_group_scope import ComponentGroupScope
from openhcs.core.function_patterns import CompiledFunctionPattern
from openhcs.core.source_bindings import (
    CompiledSourceBindingPlan,
    CompiledSourceUniversePlan,
)
from openhcs.core.source_load_plan import SourceLoadPlan
from openhcs.core.step_dependencies import StepInputDependency

if TYPE_CHECKING:
    from openhcs.core.config import StreamingConfig
else:
    StreamingConfig = Any

ArtifactInputPlans = Mapping[ArtifactSpecRef, ArtifactInputPlan]
ArtifactOutputPlans = Mapping[ArtifactSpecRef, ArtifactOutputPlan]


@dataclass(frozen=True, slots=True)
class FrameworkDeviceBinding:
    """One framework-local GPU device selected for compiled execution."""

    memory_type: MemoryType
    device_id: int

    def __post_init__(self) -> None:
        if self.memory_type not in GPU_MEMORY_TYPES:
            raise ValueError(
                "FrameworkDeviceBinding requires a GPU memory declaration, got "
                f"{self.memory_type.value!r}."
            )
        if type(self.device_id) is not int or self.device_id < 0:
            raise ValueError(
                "FrameworkDeviceBinding.device_id must be a non-negative integer."
            )


@dataclass(frozen=True, slots=True)
class FrameworkDeviceAssignment:
    """Compiled framework-local device bindings for one step footprint."""

    bindings: tuple[FrameworkDeviceBinding, ...] = ()

    def __post_init__(self) -> None:
        memory_types = tuple(binding.memory_type for binding in self.bindings)
        if len(memory_types) != len(frozenset(memory_types)):
            raise ValueError(
                "FrameworkDeviceAssignment cannot bind one memory type twice."
            )

    @classmethod
    def from_mapping(
        cls,
        device_ids: Mapping[MemoryType, int],
    ) -> "FrameworkDeviceAssignment":
        return cls(
            tuple(
                FrameworkDeviceBinding(memory_type, device_id)
                for memory_type, device_id in sorted(
                    device_ids.items(),
                    key=lambda item: item[0].value,
                )
            )
        )

    @classmethod
    def merge(
        cls,
        assignments: Sequence["FrameworkDeviceAssignment"],
    ) -> "FrameworkDeviceAssignment":
        """Merge compatible compiled assignments into one cleanup footprint."""

        device_ids: dict[MemoryType, int] = {}
        for assignment in assignments:
            for binding in assignment.bindings:
                existing = device_ids.get(binding.memory_type)
                if existing is not None and existing != binding.device_id:
                    raise ValueError(
                        f"Conflicting {binding.memory_type.value!r} device "
                        f"assignments: {existing} and {binding.device_id}."
                    )
                device_ids[binding.memory_type] = binding.device_id
        return cls.from_mapping(device_ids)

    def select(
        self,
        memory_types: frozenset[MemoryType],
    ) -> "FrameworkDeviceAssignment":
        """Return the exact subset required by one compiled step."""

        selected = tuple(
            binding
            for binding in self.bindings
            if binding.memory_type in memory_types
        )
        if frozenset(binding.memory_type for binding in selected) != memory_types:
            missing = memory_types - frozenset(
                binding.memory_type for binding in selected
            )
            raise ValueError(
                "No framework-local device assignment exists for "
                f"{', '.join(sorted(item.value for item in missing))}."
            )
        return type(self)(selected)

    def device_id_for(self, memory_type: MemoryType) -> int | None:
        """Return the declared local device, or ``None`` for CPU memory."""

        if memory_type not in GPU_MEMORY_TYPES:
            return None
        for binding in self.bindings:
            if binding.memory_type is memory_type:
                return binding.device_id
        raise ValueError(
            f"No compiled device assignment exists for {memory_type.value!r}."
        )

    def scope_for(
        self,
        declaration: str | None,
    ) -> AbstractContextManager[None]:
        """Return the declaration-owned execution scope for one framework."""

        if declaration is None:
            return nullcontext()
        memory_type = MemoryType(declaration)
        device_id = self.device_id_for(memory_type)
        if device_id is None:
            return nullcontext()
        return memory_type.device_scope(device_id)

    def cleanup_loaded(self) -> None:
        """Release loaded allocators in this exact compiled footprint."""

        failures: list[Exception] = []
        for binding in self.bindings:
            try:
                binding.memory_type.cleanup_loaded(binding.device_id)
            except Exception as error:
                error.add_note(
                    f"Cleanup failed for {binding.memory_type.value!r} "
                    f"device {binding.device_id}."
                )
                failures.append(error)
        if failures:
            raise ExceptionGroup(
                "Compiled framework cleanup failed",
                failures,
            )


@dataclass(frozen=True, slots=True)
class InputConversionPlan:
    """Typed input-conversion section of a compiled step plan."""

    output_dir: Path
    backend: str
    uses_virtual_workspace: bool
    original_subdir: str


@dataclass(frozen=True, slots=True)
class MaterializedOutputPlan:
    """Typed materialized-output section of a compiled step plan."""

    output_dir: Path
    backend: str
    plate_root: str
    sub_dir: str
    analysis_results_dir: str | None


@dataclass(frozen=True, slots=True)
class SequentialRuntimeFilter:
    """One compile-resolved sequential component constraint."""

    component: SequentialComponents
    value: str

    @property
    def component_name(self) -> str:
        return self.component.value


@dataclass(frozen=True, slots=True)
class SequentialRuntimeFilterPlan:
    """Compile-resolved sequential filtering for one runtime context."""

    filters: tuple[SequentialRuntimeFilter, ...] = ()

    @classmethod
    def disabled(cls) -> "SequentialRuntimeFilterPlan":
        return cls()

    @property
    def enabled(self) -> bool:
        return bool(self.filters)


@dataclass(frozen=True, slots=True)
class RuntimeArtifactMaterializationPlan:
    """Compile-resolved runtime artifact materialization target."""

    persistent_enabled: bool = False
    persistent_backend: str | None = None

    @classmethod
    def disabled(cls) -> "RuntimeArtifactMaterializationPlan":
        return cls()

    @property
    def has_persistent_target(self) -> bool:
        return self.persistent_enabled

    def require_persistent_backend(self) -> str:
        if self.persistent_backend is None:
            raise RuntimeError(
                "Runtime artifact materialization plan has no persistent backend."
            )
        return self.persistent_backend


@dataclass(slots=True)
class CompiledStepPlan:
    """Mutable compile-time plan for one pipeline step.

    This object is intentionally the source of truth. Compiler phases should
    mutate fields on this dataclass rather than writing string-keyed dicts.
    """

    step_index: int
    step_name: str
    step_type: str
    axis_id: str
    step_scope_id: str | None = None
    func: Any = None
    input_dir: Path | None = None
    output_dir: Path | None = None
    output_plate_root: str | None = None
    sub_dir: str | None = None
    analysis_results_dir: str | None = None
    pipeline_position: int | None = None
    input_source: Any = None
    variable_components: Sequence[VariableComponents] | None = None
    group_by: Any = None
    sequential_processing: Any = None
    sequential_filter_plan: SequentialRuntimeFilterPlan = field(
        default_factory=SequentialRuntimeFilterPlan.disabled
    )
    main_input_dependency: StepInputDependency = field(
        default_factory=StepInputDependency.unresolved
    )
    source_binding_plan: CompiledSourceBindingPlan = field(
        default_factory=CompiledSourceBindingPlan.empty
    )
    source_universe_plan: CompiledSourceUniversePlan = field(
        default_factory=CompiledSourceUniversePlan.empty
    )
    source_load_plan: SourceLoadPlan = field(default_factory=SourceLoadPlan)
    runtime_artifact_materialization: RuntimeArtifactMaterializationPlan = field(
        default_factory=RuntimeArtifactMaterializationPlan.disabled
    )
    artifact_inputs: OrderedDict[ArtifactSpecRef, ArtifactInputPlan] = field(
        default_factory=OrderedDict
    )
    artifact_outputs: OrderedDict[ArtifactSpecRef, ArtifactOutputPlan] = field(
        default_factory=OrderedDict
    )
    execution_group_scope: ComponentGroupScope = field(
        default_factory=ComponentGroupScope.ungrouped
    )
    compiled_function_pattern: CompiledFunctionPattern | None = None
    input_conversion: InputConversionPlan | None = None
    input_conversion_config: Any = None
    materialized_output: MaterializedOutputPlan | None = None
    materialization_config: Any = None
    read_backend: str | None = None
    write_backend: str | None = None
    main_flow_axis_persistence_enabled: bool | None = None
    input_memory_type: str | None = None
    output_memory_type: str | None = None
    device_assignment: FrameworkDeviceAssignment = field(
        default_factory=FrameworkDeviceAssignment
    )
    zarr_config: Mapping[str, Any] | None = None
    streaming_configs: dict[str, StreamingConfig] = field(default_factory=dict)
    visualize: bool = False
    create_openhcs_metadata: bool = False
    chainbreaker: bool = False
    error: str | None = None

    def require_function_execution_ready(self) -> "CompiledStepPlan":
        """Validate the compiler-owned fields required by FunctionStep runtime."""
        if not self.axis_id:
            raise ValueError(
                f"Compiled plan for step {self.step_index} ({self.step_name}) "
                "has no axis_id."
            )
        if self.input_dir is None:
            raise ValueError(
                f"Compiled plan for step {self.step_index} ({self.step_name}) "
                "has no input_dir."
            )
        if self.output_dir is None:
            raise ValueError(
                f"Compiled plan for step {self.step_index} ({self.step_name}) "
                "has no output_dir."
            )
        self.require_variable_components()
        if self.read_backend is None:
            raise ValueError(
                f"Compiled plan for step {self.step_index} ({self.step_name}) "
                "has no read_backend."
            )
        if self.write_backend is None:
            raise ValueError(
                f"Compiled plan for step {self.step_index} ({self.step_name}) "
                "has no write_backend."
            )
        if self.pipeline_position is None:
            raise ValueError(
                f"Compiled plan for step {self.step_index} ({self.step_name}) "
                "has no pipeline_position."
            )
        if self.output_plate_root is None:
            raise ValueError(
                f"Compiled plan for step {self.step_index} ({self.step_name}) "
                "has no output_plate_root."
            )
        if self.sub_dir is None:
            raise ValueError(
                f"Compiled plan for step {self.step_index} ({self.step_name}) "
                "has no sub_dir."
            )
        if self.compiled_function_pattern is None:
            raise ValueError(
                f"Compiled plan for step {self.step_index} ({self.step_name}) "
                "has no compiled_function_pattern."
            )
        return self

    def require_variable_components(self) -> Sequence[VariableComponents]:
        variable_components = self.variable_components
        if variable_components is None:
            raise ValueError(
                f"Step {self.step_index} ({self.step_name}) is missing compiled "
                "variable_components. Stack-axis semantics must be resolved "
                "before runtime execution."
            )
        return variable_components

    @property
    def execution_scope(self) -> FunctionStepExecutionScope:
        """Return the scope owned by the compiled function pattern."""
        pattern = self.compiled_function_pattern
        if pattern is None:
            raise ValueError(
                f"Compiled plan for step {self.step_index} ({self.step_name}) "
                "has no compiled_function_pattern."
            )
        return pattern.execution_scope

    @property
    def owns_runtime_outputs(self) -> bool:
        """Return whether this context owns the step's runtime outputs."""
        return self.execution_scope.context_owns_outputs(
            metadata_writer=self.create_openhcs_metadata,
        )

    @property
    def gpu_memory_types(self) -> frozenset[MemoryType]:
        """Return the GPU-backed memory declarations used by this step."""

        memory_types = {
            MemoryType(declaration)
            for declaration in (
                self.input_memory_type,
                self.output_memory_type,
            )
            if declaration is not None
        }
        if self.compiled_function_pattern is not None:
            memory_types.update(
                memory_type
                for invocation in self.compiled_function_pattern.iter_invocations()
                for memory_type in invocation.contract.declared_memory_types
            )
        return frozenset(
            memory_type
            for memory_type in memory_types
            if memory_type in GPU_MEMORY_TYPES
        )

    @property
    def requires_gpu(self) -> bool:
        """Return whether the compiled execution footprint requires a GPU."""

        return bool(self.gpu_memory_types)

    def device_id_for(self, declaration: str | None) -> int | None:
        """Return the compiled framework-local device for one declaration."""

        if declaration is None:
            return None
        return self.device_assignment.device_id_for(MemoryType(declaration))

    def memory_device_scope(
        self,
        declaration: str | None,
    ) -> AbstractContextManager[None]:
        """Return the compiled declaration-owned framework device scope."""

        return self.device_assignment.scope_for(declaration)

    @property
    def variable_component_values(self) -> list[str]:
        return [component.value for component in self.require_variable_components()]

    @property
    def group_by_value(self) -> str | None:
        return self.group_by.value if self.group_by else None

    @property
    def execution_group_value(self) -> str | None:
        component = self.execution_group_scope.component
        return None if component is None else component.value

    @property
    def artifact_analysis_output_dir(self) -> Path:
        output_dir = self.analysis_results_dir
        if self.materialized_output is not None:
            output_dir = self.materialized_output.analysis_results_dir
        if output_dir is None:
            raise ValueError(
                f"Step {self.step_index} ({self.step_name}) has no analysis results directory."
            )
        return Path(output_dir)

    @property
    def artifact_images_dir(self) -> str:
        if self.materialized_output is not None:
            return str(self.materialized_output.output_dir)
        if self.output_dir is None:
            raise ValueError(
                f"Compiled plan for step {self.step_index} ({self.step_name}) "
                "has no output_dir."
            )
        return str(self.output_dir)
