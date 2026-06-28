"""Typed compiled step plans used as compiler/runtime source of truth."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from openhcs.constants.constants import SequentialComponents, VariableComponents
from openhcs.core.artifacts import ArtifactInputPlan, ArtifactOutputPlan
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

ArtifactInputPlans = Mapping[str, ArtifactInputPlan]
ArtifactOutputPlans = Mapping[str, ArtifactOutputPlan]


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
    artifact_inputs: OrderedDict[str, ArtifactInputPlan] = field(
        default_factory=OrderedDict
    )
    artifact_outputs: OrderedDict[str, ArtifactOutputPlan] = field(
        default_factory=OrderedDict
    )
    artifact_inputs_by_group: dict[
        Any, OrderedDict[str, ArtifactInputPlan]
    ] = field(default_factory=dict)
    artifact_outputs_by_group: dict[
        Any, OrderedDict[str, ArtifactOutputPlan]
    ] = field(default_factory=dict)
    execution_groups: list[str | None] = field(default_factory=lambda: [None])
    compiled_function_pattern: CompiledFunctionPattern | None = None
    input_conversion: InputConversionPlan | None = None
    input_conversion_config: Any = None
    materialized_output: MaterializedOutputPlan | None = None
    materialization_config: Any = None
    read_backend: str | None = None
    write_backend: str | None = None
    input_memory_type: str | None = None
    output_memory_type: str | None = None
    gpu_id: int | None = None
    zarr_config: Mapping[str, Any] | None = None
    streaming_configs: dict[str, StreamingConfig] = field(default_factory=dict)
    visualize: bool = False
    create_openhcs_metadata: bool = False
    chainbreaker: bool = False
    error: str | None = None
