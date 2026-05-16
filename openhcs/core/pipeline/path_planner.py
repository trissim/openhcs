"""
Pipeline path planning - actually reduced duplication.

This version ACTUALLY eliminates duplication instead of adding abstraction theater.
"""

import logging
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Set

from openhcs.constants.input_source import InputSource
from openhcs.core.artifacts import ArtifactInputPlan, ArtifactOutputPlan, ArtifactSpec
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.function_patterns import (
    CompiledFunctionPattern,
    compile_function_pattern,
    inject_artifact_input_values,
    inject_kwargs_into_pattern,
    strip_disabled_functions,
)
from openhcs.core.invocation_artifacts import (
    ArtifactDeclarationStepContext,
    InvocationArtifactDeclarationProviderLike,
    callable_contract_artifact_declarations,
)
from openhcs.core.compiled_step_plan import (
    CompiledStepPlan,
    InputConversionPlan,
    MaterializedOutputPlan,
)
from openhcs.core.pipeline.artifact_planning import (
    ArtifactGraph,
    extract_artifact_declarations,
)
from openhcs.core.pipeline.step_snapshot import (
    StepSnapshot,
    build_step_snapshots,
)
from openhcs.core.step_dependencies import (
    StepInputDependency,
    StepInputDependencyKind,
)
from openhcs.core.steps.abstract import AbstractStep

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArtifactPlanMaps:
    """Compiled artifact I/O maps for one step."""

    declarations: ArtifactGraph
    execution_groups: List[Optional[str]]
    inputs: dict[str, ArtifactInputPlan]
    outputs: dict[str, ArtifactOutputPlan]
    inputs_by_group: dict[Optional[str], OrderedDict]
    outputs_by_group: dict[Optional[str], OrderedDict]


@dataclass(frozen=True)
class PathPlanningContext:
    """Nominal construction context for one path-planning pass."""

    context: ProcessingContext
    pipeline_config: Any
    orchestrator: Any | None = None
    step_snapshots: tuple[StepSnapshot, ...] = ()
    declaration_provider: InvocationArtifactDeclarationProviderLike = (
        callable_contract_artifact_declarations
    )


@dataclass(frozen=True)
class PathPlannerExecutionGroups:
    """Execution-group discovery stage for path planning."""

    planner: Any

    @staticmethod
    def normalize_group_key(key: Optional[Any]) -> Optional[str]:
        if key is None:
            return None
        return str(key)

    def get_execution_groups(self, snapshot: StepSnapshot) -> List[Optional[str]]:
        """Determine which component groups this step will execute for."""
        if not snapshot.is_function_step:
            return [None]

        func_pattern = snapshot.func
        if isinstance(func_pattern, dict):
            result = [self.normalize_group_key(k) for k in func_pattern.keys()]
            logger.debug("Dict function pattern groups: %s", result)
            return result

        group_by = self.normalized_group_by(snapshot)
        logger.debug(
            "Resolved group_by for step %s via StepSnapshot: %s",
            snapshot.name,
            group_by,
        )

        if not self.group_by_requires_component_keys(group_by):
            logger.debug("No group_by configured; using a single ungrouped execution.")
            return [None]

        if self.planner.orchestrator is None:
            logger.warning(
                "PathPlanner: orchestrator not available; cannot resolve "
                "group_by component keys for artifact planning."
            )
            return [None]

        try:
            result = [
                self.normalize_group_key(k)
                for k in self.planner.orchestrator.get_component_keys(group_by)
            ]
            logger.debug("Resolved execution groups from orchestrator: %s", result)
            return result
        except Exception as e:
            logger.warning(
                f"PathPlanner: failed to resolve component keys for {group_by}: {e}"
            )
            return [None]

    @staticmethod
    def group_by_requires_component_keys(group_by: Any) -> bool:
        from openhcs.constants import GroupBy

        if group_by is None or group_by == GroupBy.NONE:
            return False
        return group_by.value is not None

    @staticmethod
    def normalized_group_by(snapshot: StepSnapshot) -> Any:
        """Use the same group_by normalization as compiled execution plans."""
        from openhcs.core.pipeline.funcstep_contract_validator import (
            FuncStepContractValidator,
        )

        return FuncStepContractValidator.normalized_group_by(
            snapshot.group_by,
            snapshot.variable_components,
            snapshot.name,
        )


@dataclass(frozen=True)
class PathPlannerArtifactStage:
    """Artifact declaration, I/O-plan, and FunctionStep injection stage."""

    planner: Any

    def prepare_step_declarations(
        self,
        snapshot: StepSnapshot,
    ) -> tuple[ArtifactGraph, List[Optional[str]], Any]:
        """Normalize a step's function pattern and collect artifact declarations."""
        if not snapshot.is_function_step:
            return ArtifactGraph.empty(), [None], None

        func_pattern = self.inject_injectable_params(snapshot.func, snapshot)
        func_pattern = strip_disabled_functions(func_pattern)

        declarations = extract_artifact_declarations(
            func_pattern if func_pattern else [],
            declaration_provider=self.planner.declaration_provider,
            step_context=self.artifact_declaration_context(snapshot),
        )
        execution_groups = self.planner.execution_groups.get_execution_groups(snapshot)
        declarations = self.namespace_grouped_outputs_for_runtime_consumers(
            func_pattern,
            declarations,
            execution_groups,
        )
        return declarations, execution_groups, func_pattern

    def namespace_grouped_outputs_for_runtime_consumers(
        self,
        func_pattern: Any,
        declarations: ArtifactGraph,
        execution_groups: List[Optional[str]],
    ) -> ArtifactGraph:
        """Namespace grouped artifact outputs unless a later step consumes them globally."""
        if (
            isinstance(func_pattern, dict)
            or execution_groups == [None]
            or not declarations.output_names
        ):
            return declarations

        output_groups = {
            output_key: tuple(
                self.planner.execution_groups.normalize_group_key(group)
                for group in execution_groups
            )
            for output_key in declarations.output_names
        }
        return declarations.with_output_groups(output_groups)

    def compile_plan_maps(
        self,
        snapshot: StepSnapshot,
        step_index: int,
        declarations: ArtifactGraph,
        execution_groups: List[Optional[str]],
    ) -> ArtifactPlanMaps:
        """Compile artifact declarations into runtime I/O maps."""
        step_name = snapshot.name
        artifact_outputs = self.process_artifact_outputs(
            declarations.outputs,
            step_index,
            declarations.output_groups,
            step_name=step_name,
        )
        artifact_inputs = self.process_artifact_inputs(
            declarations.inputs,
            declarations.outputs,
            step_index,
            consumer_groups=execution_groups,
            step_name=step_name,
        )
        normalized_groups = [
            self.planner.execution_groups.normalize_group_key(group)
            for group in execution_groups
        ]

        return ArtifactPlanMaps(
            declarations=declarations,
            execution_groups=execution_groups,
            inputs=artifact_inputs,
            outputs=artifact_outputs,
            inputs_by_group=self.planner.paths.artifact_inputs_by_group(
                artifact_inputs,
                normalized_groups,
            ),
            outputs_by_group=self.planner.paths.artifact_outputs_by_group(
                artifact_outputs
            ),
        )

    def build_step_compiled_function_pattern(
        self,
        snapshot: StepSnapshot,
        is_function_step: bool,
        func_pattern: Any,
        artifact_inputs: Mapping[str, ArtifactInputPlan],
        artifact_outputs: Mapping[str, ArtifactOutputPlan],
    ) -> CompiledFunctionPattern | None:
        """Build the executable function-pattern graph for a FunctionStep."""
        if not is_function_step or not func_pattern:
            return None

        return compile_function_pattern(
            func_pattern,
            artifact_inputs,
            artifact_outputs,
            declaration_provider=self.planner.declaration_provider,
            step_context=self.artifact_declaration_context(snapshot),
        )

    @staticmethod
    def artifact_declaration_context(
        snapshot: StepSnapshot,
    ) -> ArtifactDeclarationStepContext:
        """Return compile-time context for invocation artifact providers."""
        return ArtifactDeclarationStepContext(
            step_name=snapshot.name,
            step_index=snapshot.index,
            source_bindings=snapshot.source_bindings,
            processing_config=snapshot.processing_config,
        )

    def process_artifact_outputs(
        self,
        outputs: Mapping[str, ArtifactSpec],
        sid: int,
        output_groups: Optional[Mapping[str, Set[Optional[str]]]] = None,
        step_name: Optional[str] = None,
    ) -> dict[str, ArtifactOutputPlan]:
        """Compile storage plans for artifacts produced by this step."""
        result: dict[str, ArtifactOutputPlan] = {}
        if not outputs:
            return result

        results_path = self.planner.paths.results_path()
        for key, spec in sorted(outputs.items()):
            filename = PipelinePathPlanner._build_axis_filename(
                self.planner.ctx.axis_id,
                key,
                step_index=sid,
            )
            path = results_path / filename
            groups = output_groups.get(key, {None}) if output_groups else {None}
            normalized_groups = sorted(
                {
                    self.planner.execution_groups.normalize_group_key(g)
                    for g in groups
                }
            )
            paths_by_group = self.planner.paths.paths_by_group(
                str(path),
                normalized_groups,
            )
            result[key] = ArtifactOutputPlan(
                name=key,
                path=str(path),
                kind=spec.kind,
                materialization=spec.materialization,
                group_keys=tuple(normalized_groups),
                paths_by_group=paths_by_group,
                producer_step_index=sid,
                producer_step_scope_id=self.planner.plans[sid].step_scope_id,
                producer_step_name=step_name,
            )
            self.planner.declared[key] = result[key]

        return result

    def process_artifact_inputs(
        self,
        inputs: Mapping[str, ArtifactSpec],
        step_outputs: Mapping[str, ArtifactSpec],
        sid: int,
        consumer_groups: Optional[List[Optional[str]]] = None,
        step_name: Optional[str] = None,
    ) -> dict[str, ArtifactInputPlan]:
        """Compile storage plans for artifacts consumed by this step."""
        result: dict[str, ArtifactInputPlan] = {}
        if not inputs:
            return result

        consumer_groups = consumer_groups or [None]
        normalized_consumers = [
            self.planner.execution_groups.normalize_group_key(g)
            for g in consumer_groups
        ]

        for key, input_spec in sorted(inputs.items()):
            if key in self.planner.declared:
                result[key] = self._producer_artifact_input_plan(
                    key,
                    input_spec,
                    normalized_consumers,
                    sid,
                    step_name,
                )
            elif key in step_outputs:
                output_spec = step_outputs[key]
                if output_spec.kind != input_spec.kind:
                    raise ValueError(
                        f"Artifact '{key}' is produced as {output_spec.kind.value} "
                        f"but consumed as {input_spec.kind.value} in step '{step_name or sid}'."
                    )
                result[key] = ArtifactInputPlan(
                    name=key,
                    path="self",
                    kind=input_spec.kind,
                    source_step_id=sid,
                    source_step_scope_id=self.planner.plans[sid].step_scope_id,
                )
            elif key not in METADATA_RESOLVERS:
                raise ValueError(f"Step {sid} needs '{key}' but it's not available")

        return result

    def _producer_artifact_input_plan(
        self,
        key: str,
        input_spec: ArtifactSpec,
        normalized_consumers: list[Optional[str]],
        sid: int,
        step_name: Optional[str],
    ) -> ArtifactInputPlan:
        producer = self.planner.declared[key]
        if producer.kind != input_spec.kind:
            producer_name = (
                producer.producer_step_name
                or producer.producer_step_index
                or "unknown"
            )
            consumer_name = step_name or sid
            raise ValueError(
                f"Artifact input '{key}' in step '{consumer_name}' expects "
                f"{input_spec.kind.value}, but producer step '{producer_name}' "
                f"provides {producer.kind.value}."
            )
        producer_groups = list(producer.group_keys or (None,))

        if producer_groups != [None] and normalized_consumers == [None]:
            paths_by_group = dict(producer.paths_by_group or {})
        elif producer_groups != [None]:
            missing = [
                group
                for group in normalized_consumers
                if group not in producer_groups
            ]
            if missing:
                if len(producer_groups) == 1 and producer.paths_by_group:
                    producer_group = producer_groups[0]
                    producer_path = producer.paths_by_group.get(producer_group)
                    if producer_path is not None:
                        paths_by_group = {
                            group: producer_path
                            for group in normalized_consumers
                        }
                        return ArtifactInputPlan(
                            name=key,
                            path=producer_path,
                            kind=producer.kind,
                            paths_by_group=paths_by_group,
                            group_keys=tuple(normalized_consumers),
                            source_step_id=producer.producer_step_index,
                            source_step_scope_id=producer.producer_step_scope_id,
                        )
                producer_name = (
                    producer.producer_step_name
                    or producer.producer_step_index
                    or "unknown"
                )
                consumer_name = step_name or sid
                raise ValueError(
                    f"Artifact input '{key}' in step '{consumer_name}' cannot be resolved: "
                    f"producer step '{producer_name}' provides groups {producer_groups}, "
                    f"but consumer needs {missing}."
                )
            paths_by_group = {
                group: producer.paths_by_group[group]
                for group in normalized_consumers
                if producer.paths_by_group
                and group in producer.paths_by_group
            }
        else:
            paths_by_group = {
                group: producer.path for group in normalized_consumers
            }

        return ArtifactInputPlan(
            name=key,
            path=producer.path,
            kind=producer.kind,
            paths_by_group=paths_by_group,
            group_keys=tuple(producer_groups),
            source_step_id=producer.producer_step_index,
            source_step_scope_id=producer.producer_step_scope_id,
        )

    def inject_metadata(self, pattern: Any, inputs: Dict) -> Any:
        """Inject metadata for artifact inputs."""
        for key in inputs:
            if key in METADATA_RESOLVERS and key not in self.planner.declared:
                value = METADATA_RESOLVERS[key]["resolver"](self.planner.ctx)
                pattern = inject_artifact_input_values(pattern, {key: value})
        return pattern

    def inject_injectable_params(
        self,
        pattern: Any,
        snapshot: StepSnapshot,
    ) -> Any:
        """Inject registry-declared injectable params into function kwargs."""
        from openhcs.processing.backends.lib_registry.unified_registry import LibraryRegistryBase

        param_names = [
            param_name
            for param_name, _, _ in LibraryRegistryBase.INJECTABLE_PARAMS
        ]
        param_kwargs = {}
        for param_name in param_names:
            value = snapshot.injectable_values.get(param_name)
            if value is not None:
                param_kwargs[param_name] = value

        if not param_kwargs:
            return pattern

        return inject_kwargs_into_pattern(pattern, param_kwargs)


@dataclass(frozen=True)
class PathPlannerMaterializationStage:
    """Input conversion and materialized-output planning stage."""

    planner: Any

    def materialized_output_dir_for_step(
        self,
        snapshot: StepSnapshot,
    ) -> Optional[Path]:
        """Resolve optional per-step materialization output directory."""
        materialization_config = snapshot.materialization_config
        if not materialization_config or not materialization_config.enabled:
            return None

        step_axis_filters = self.planner.ctx.step_axis_filters.get(
            snapshot.index,
            {},
        )
        materialization_filter = step_axis_filters.get(
            "step_materialization_config"
        )
        if materialization_filter:
            should_materialize = (
                self.planner.ctx.axis_id
                in materialization_filter["resolved_axis_values"]
            )
            if not should_materialize:
                logger.debug(
                    "Skipping materialization for step %s, axis %s (filtered out)",
                    snapshot.name,
                    self.planner.ctx.axis_id,
                )
                return None

        return self.planner.paths.build_output_path(materialization_config)

    def input_conversion_plan_for_step(
        self,
        step_index: int,
        input_dir: Path,
    ) -> Optional[InputConversionPlan]:
        """Resolve optional compiler-provided or config-provided input conversion."""
        existing_plan = self.planner.plans[step_index].input_conversion
        if existing_plan is not None:
            return existing_plan

        output_dir = self.planner.paths.input_conversion_output_path(step_index)
        if output_dir is None:
            return None

        return InputConversionPlan(
            output_dir=output_dir,
            backend=self.planner.vfs.materialization_backend.value,
            uses_virtual_workspace=False,
            original_subdir=input_dir.name,
        )

    def apply_materialization_plan(
        self,
        snapshot: StepSnapshot,
        step_index: int,
        materialized_output_dir: Optional[Path],
    ) -> None:
        """Attach optional materialization path fields to a step plan."""
        if not materialized_output_dir:
            return

        materialization_config = snapshot.materialization_config
        materialized_plate_root = self.planner.paths.build_output_plate_root(
            self.planner.plate_path,
            materialization_config,
            is_per_step_materialization=False,
        )
        self.planner.plans[step_index].materialized_output = MaterializedOutputPlan(
            output_dir=materialized_output_dir,
            backend=self.planner.vfs.materialization_backend.value,
            plate_root=str(materialized_plate_root),
            sub_dir=materialization_config.sub_dir,
            analysis_results_dir=str(
                self.planner.paths.analysis_results_dir_for(materialized_output_dir)
            ),
        )
        self.planner.plans[step_index].materialization_config = materialization_config

    def apply_input_conversion_plan(
        self,
        step_index: int,
        input_conversion_plan: Optional[InputConversionPlan],
    ) -> None:
        """Attach optional input conversion path fields to a step plan."""
        if input_conversion_plan is None:
            return

        self.planner.plans[step_index].input_conversion = input_conversion_plan


@dataclass(frozen=True)
class PathPlannerValidationStage:
    """Connectivity and materialization path validation stage."""

    planner: Any

    def validate(self, pipeline: List):
        """Validate connectivity and materialization paths."""
        for i in range(1, len(self.planner.step_snapshots)):
            curr = self.planner.step_snapshots[i]
            dependency = self.planner.plans[i].main_input_dependency
            if dependency.kind is StepInputDependencyKind.PIPELINE_START:
                continue
            if dependency.kind is not StepInputDependencyKind.STEP_OUTPUT:
                raise ValueError(
                    f"Step {curr.name} has unresolved main input dependency."
                )
            source_step_index = dependency.source_step_index
            if source_step_index is None:
                raise ValueError(
                    f"Step {curr.name} main input dependency is missing source_step_index."
                )
            curr_in = self.planner.plans[i].input_dir
            source_out = self.planner.plans[source_step_index].output_dir
            if curr_in != source_out:
                has_artifact_bridge = any(
                    inp.source_step_id in [source_step_index, "prev"]
                    or inp.source_step_scope_id == dependency.source_step_scope_id
                    for inp in self.planner.plans[i].artifact_inputs.values()
                )
                if not has_artifact_bridge:
                    producer_name = self.planner.step_snapshots[source_step_index].name
                    raise ValueError(f"Disconnect: {producer_name} -> {curr.name}")

        self.validate_materialization_paths(pipeline)

    def validate_materialization_paths(self, pipeline: List[AbstractStep]) -> None:
        """Validate and resolve materialization path collisions."""
        global_path = self.planner.paths.build_output_path(self.planner.cfg)

        mat_steps = [
            (
                snapshot,
                self.planner.plans[i].pipeline_position or i,
                self.planner.paths.build_output_path(snapshot.materialization_config),
            )
            for i, snapshot in enumerate(self.planner.step_snapshots)
            if snapshot.materialization_config
            and snapshot.materialization_config.enabled
        ]

        path_groups = defaultdict(list)
        for snapshot, pos, path in mat_steps:
            if path == global_path:
                self.resolve_and_update_paths(snapshot, pos, path, "main flow")
            else:
                path_groups[str(path)].append((snapshot, pos, path))

        for path_key, step_list in path_groups.items():
            if len(step_list) > 1:
                for snapshot, pos, path in step_list:
                    self.resolve_and_update_paths(snapshot, pos, path, f"pos {pos}")

    def resolve_and_update_paths(
        self,
        snapshot: StepSnapshot,
        position: int,
        original_path: Path,
        conflict_type: str,
    ) -> None:
        """Resolve path conflict by updating the compiled plan only."""
        del original_path, conflict_type
        materialization_config = snapshot.materialization_config

        original_sub_dir = materialization_config.sub_dir
        new_sub_dir = f"{original_sub_dir}_step{position}"

        from dataclasses import replace
        updated_config = replace(materialization_config, sub_dir=new_sub_dir)

        resolved_path = self.planner.paths.build_output_path(updated_config)
        resolved_analysis_results_dir = self.planner.paths.analysis_results_dir_for(
            resolved_path
        )

        if step_plan := self.planner.plans.get(position):
            if step_plan.materialized_output is not None:
                step_plan.materialized_output = MaterializedOutputPlan(
                    output_dir=resolved_path,
                    backend=step_plan.materialized_output.backend,
                    plate_root=step_plan.materialized_output.plate_root,
                    sub_dir=new_sub_dir,
                    analysis_results_dir=str(resolved_analysis_results_dir),
                )
                step_plan.materialization_config = updated_config


@dataclass(frozen=True)
class PathPlannerPathAuthority:
    """Path and grouped-artifact expansion authority for compiled step plans."""

    planner: Any

    @staticmethod
    def build_output_plate_root(
        plate_path: Path,
        path_config,
        is_per_step_materialization: bool = False,
    ) -> Path:
        """Build output plate root directory directly from configuration components.

        Results always use the output plate path so metadata remains colocated with
        processed images instead of the original input images.
        """
        del is_per_step_materialization

        if str(plate_path).startswith("/omero/"):
            base = plate_path.parent
        elif path_config.global_output_folder:
            base = Path(path_config.global_output_folder)
        else:
            base = plate_path.parent

        if not path_config.output_dir_suffix:
            raise ValueError(
                f"output_dir_suffix cannot be None or empty. "
                f"Results must always use output plate path, not input plate path. "
                f"Config: {path_config}"
            )

        return base / f"{plate_path.name}{path_config.output_dir_suffix}"

    @staticmethod
    def paths_by_group(
        base_path: str,
        group_keys: List[Optional[str]],
    ) -> Dict[Optional[str], str]:
        """Expand one artifact path into per-execution-group artifact paths."""
        paths_by_group: Dict[Optional[str], str] = {}
        for group_key in group_keys:
            if group_key is None:
                paths_by_group[group_key] = base_path
            else:
                paths_by_group[group_key] = PipelinePathPlanner.build_dict_pattern_path(
                    base_path,
                    group_key,
                )
        return paths_by_group

    @staticmethod
    def artifact_outputs_by_group(
        artifact_outputs: Dict[str, ArtifactOutputPlan],
    ) -> Dict[Optional[str], OrderedDict]:
        """Expand artifact outputs into per-group plans with finalized paths."""
        if not artifact_outputs:
            return {}

        grouped: Dict[Optional[str], OrderedDict] = defaultdict(OrderedDict)
        for output_key, output_plan in artifact_outputs.items():
            paths_by_group = output_plan.paths_by_group or {None: output_plan.path}
            for group_key in paths_by_group:
                grouped[group_key][output_key] = output_plan.for_group(group_key)
        return dict(grouped)

    @staticmethod
    def artifact_inputs_by_group(
        artifact_inputs: Dict[str, ArtifactInputPlan],
        consumer_groups: List[Optional[str]],
    ) -> Dict[Optional[str], OrderedDict]:
        """Expand artifact inputs into per-group plans with finalized paths."""
        if not artifact_inputs:
            return {}

        grouped: Dict[Optional[str], OrderedDict] = {}
        for group_key in consumer_groups:
            per_group = OrderedDict()
            for input_key, input_plan in artifact_inputs.items():
                group_plan = input_plan.for_group(group_key)
                if group_plan is not None:
                    per_group[input_key] = group_plan
            grouped[group_key] = per_group
        return grouped

    @staticmethod
    def analysis_results_dir_for(image_dir: Path) -> Path:
        """Return the analysis-results sibling directory for an image directory."""
        return image_dir.parent / f"{image_dir.name}_results"

    def build_output_path(self, path_config=None) -> Path:
        """Build complete output path: plate_root + sub_dir."""
        config = path_config or self.planner.cfg
        plate_root = self.build_output_plate_root(
            self.planner.plate_path,
            config,
            is_per_step_materialization=False,
        )
        return plate_root / config.sub_dir

    def input_conversion_output_path(self, step_index: int) -> Optional[Path]:
        """Get input conversion output path if config exists."""
        config = self.planner.plans[step_index].input_conversion_config
        if config is not None:
            return self.build_output_path(config)
        return None

    def results_path(self) -> Path:
        """Get analysis results path from global pipeline configuration."""
        path = self.planner.ctx.global_config.materialization_results_path
        output_plate_root = self.build_output_plate_root(
            self.planner.plate_path,
            self.planner.cfg,
            is_per_step_materialization=False,
        )
        return Path(path) if Path(path).is_absolute() else output_plate_root / path


@dataclass(frozen=True)
class PathPlannerStepAssemblyStage:
    """Per-step dependency, directory, and compiled-plan assembly stage."""

    planner: Any

    def prime_future_artifact_inputs(self) -> None:
        """Precompute artifact input keys used by later steps for each step index."""
        future_inputs: Set[str] = set()
        self.planner.future_artifact_inputs = [
            set() for _ in self.planner.step_snapshots
        ]

        for i in range(len(self.planner.step_snapshots) - 1, -1, -1):
            self.planner.future_artifact_inputs[i] = set(future_inputs)

            snapshot = self.planner.step_snapshots[i]
            if snapshot.is_function_step:
                pattern = (
                    strip_disabled_functions(snapshot.func)
                    if snapshot.func
                    else []
                )
                declarations = extract_artifact_declarations(
                    pattern,
                    declaration_provider=self.planner.declaration_provider,
                    step_context=self.planner.artifacts.artifact_declaration_context(
                        snapshot
                    ),
                )
                step_inputs = set(declarations.inputs.keys())
            else:
                step_inputs = set()

            future_inputs.update(step_inputs)

    def plan_step(self, snapshot: StepSnapshot, step_index: int) -> None:
        """Plan one step's directories, artifacts, and executable pattern."""
        self.planner.plans[step_index].step_scope_id = snapshot.scope_id
        main_input_dependency = self.main_input_dependency(snapshot, step_index)
        input_dir, output_dir = self.step_io_dirs(main_input_dependency, step_index)

        declarations, execution_groups, func_pattern = (
            self.planner.artifacts.prepare_step_declarations(snapshot)
        )
        artifact_maps = self.planner.artifacts.compile_plan_maps(
            snapshot,
            step_index,
            declarations,
            execution_groups,
        )

        if snapshot.is_function_step and any(
            k in METADATA_RESOLVERS for k in declarations.inputs
        ):
            func_pattern = self.planner.artifacts.inject_metadata(
                func_pattern,
                declarations.inputs,
            )

        self.planner.plans[step_index].func = func_pattern
        self.update_core_step_plan(
            snapshot,
            step_index,
            main_input_dependency,
            input_dir,
            output_dir,
            artifact_maps,
            self.planner.artifacts.build_step_compiled_function_pattern(
                snapshot,
                snapshot.is_function_step,
                func_pattern,
                artifact_maps.inputs,
                artifact_maps.outputs,
            ),
        )
        self.planner.materialization.apply_materialization_plan(
            snapshot,
            step_index,
            self.planner.materialization.materialized_output_dir_for_step(snapshot),
        )
        self.planner.materialization.apply_input_conversion_plan(
            step_index,
            self.planner.materialization.input_conversion_plan_for_step(
                step_index,
                input_dir,
            ),
        )

    def update_core_step_plan(
        self,
        snapshot: StepSnapshot,
        step_index: int,
        main_input_dependency: StepInputDependency,
        input_dir: Path,
        output_dir: Path,
        artifact_maps: ArtifactPlanMaps,
        compiled_function_pattern: CompiledFunctionPattern | None,
    ) -> None:
        """Write the always-present path and artifact planning fields."""
        main_plate_root = self.planner.paths.build_output_plate_root(
            self.planner.plate_path,
            self.planner.cfg,
            is_per_step_materialization=False,
        )
        step_plan = self.planner.plans[step_index]
        step_plan.step_scope_id = snapshot.scope_id
        step_plan.input_dir = input_dir
        step_plan.output_dir = output_dir
        step_plan.output_plate_root = str(main_plate_root)
        step_plan.sub_dir = self.planner.cfg.sub_dir
        step_plan.analysis_results_dir = str(
            self.planner.paths.analysis_results_dir_for(Path(output_dir))
        )
        step_plan.pipeline_position = step_index
        step_plan.input_source = self.input_source(snapshot)
        step_plan.main_input_dependency = main_input_dependency
        step_plan.artifact_inputs = artifact_maps.inputs
        step_plan.artifact_outputs = artifact_maps.outputs
        step_plan.artifact_inputs_by_group = artifact_maps.inputs_by_group
        step_plan.artifact_outputs_by_group = artifact_maps.outputs_by_group
        step_plan.execution_groups = artifact_maps.execution_groups
        step_plan.compiled_function_pattern = compiled_function_pattern

    def main_input_dependency(
        self,
        snapshot: StepSnapshot,
        step_index: int,
    ) -> StepInputDependency:
        """Resolve the explicit main-input edge for one step."""
        existing_plan = self.planner.plans.get(step_index)
        if (
            existing_plan is not None
            and existing_plan.main_input_dependency.is_resolved
        ):
            return existing_plan.main_input_dependency

        if step_index == 0 or snapshot.input_source == InputSource.PIPELINE_START:
            return StepInputDependency.pipeline_start()

        producer_index = step_index - 1
        producer_scope_id = self.planner.snapshots_by_index[producer_index].scope_id
        return StepInputDependency.step_output(
            source_step_index=producer_index,
            source_step_scope_id=producer_scope_id,
        )

    def step_io_dirs(
        self,
        main_input_dependency: StepInputDependency,
        step_index: int,
    ) -> tuple[Path, Path]:
        """Resolve read/write directories for one step."""
        plan = self.planner.plans.get(step_index)
        reads_from_pipeline_start = (
            main_input_dependency.kind is StepInputDependencyKind.PIPELINE_START
        )

        if plan is not None and plan.input_dir is not None:
            input_dir = Path(plan.input_dir)
        elif reads_from_pipeline_start:
            input_dir = self.planner.initial_input
        else:
            source_step_index = main_input_dependency.source_step_index
            if source_step_index is None:
                raise ValueError(
                    f"Step {step_index} main input dependency is missing source_step_index."
                )
            input_dir = Path(self.planner.plans[source_step_index].output_dir)

        if plan is not None and plan.output_dir is not None:
            output_dir = Path(plan.output_dir)
        elif reads_from_pipeline_start:
            output_dir = self.planner.paths.build_output_path()
        else:
            output_dir = input_dir

        return input_dir, output_dir

    @staticmethod
    def input_source(snapshot: StepSnapshot) -> str:
        """Get input source string."""
        if snapshot.input_source == InputSource.PIPELINE_START:
            return "PIPELINE_START"
        return "PREVIOUS_STEP"


# ===== PATH PLANNING (NO duplication) =====

class PathPlanner:
    """Minimal path planner with zero duplication."""

    def __init__(
        self,
        planning_context: PathPlanningContext,
    ):
        self.ctx = planning_context.context
        # CRITICAL: pipeline_config is now the merged config (GlobalPipelineConfig) from context.global_config
        # This ensures proper inheritance from global config without needing field-specific code
        self.cfg = planning_context.pipeline_config.path_planning_config
        self.vfs = planning_context.pipeline_config.vfs_config
        self.plans: dict[int, CompiledStepPlan] = self.ctx.step_plans
        self.declared = {}  # Tracks artifact outputs
        self.orchestrator = planning_context.orchestrator
        self.step_snapshots = tuple(planning_context.step_snapshots)
        self.declaration_provider = planning_context.declaration_provider
        self.snapshots_by_index = {
            snapshot.index: snapshot for snapshot in self.step_snapshots
        }
        self.future_artifact_inputs: List[Set[str]] = [
            set() for _ in self.step_snapshots
        ]
        self.execution_groups = PathPlannerExecutionGroups(self)
        self.paths = PathPlannerPathAuthority(self)
        self.artifacts = PathPlannerArtifactStage(self)
        self.materialization = PathPlannerMaterializationStage(self)
        self.validation = PathPlannerValidationStage(self)
        self.steps = PathPlannerStepAssemblyStage(self)

        # Initial input determination (once)
        self.initial_input = Path(self.ctx.input_dir)
        self.plate_path = Path(self.ctx.plate_path)

    def plan(self, pipeline: List[AbstractStep]) -> dict[int, CompiledStepPlan]:
        """Plan all paths with zero duplication."""
        if len(self.step_snapshots) != len(pipeline):
            raise ValueError(
                "PathPlanner requires one StepSnapshot per pipeline step: "
                f"{len(self.step_snapshots)} snapshots for {len(pipeline)} steps."
            )

        self.steps.prime_future_artifact_inputs()
        for i, snapshot in enumerate(self.step_snapshots):
            self.steps.plan_step(snapshot, i)

        self.validation.validate(pipeline)

        # Set output_plate_root and sub_dir for metadata writing
        if pipeline:
            self.ctx.output_plate_root = self.paths.build_output_plate_root(self.plate_path, self.cfg, is_per_step_materialization=False)
            self.ctx.sub_dir = self.cfg.sub_dir

        return self.plans

# ===== PUBLIC API =====

class PipelinePathPlanner:
    """Public API matching original interface."""

    @staticmethod
    def prepare_pipeline_paths(
        context: ProcessingContext,
        pipeline_definition: List[AbstractStep],
        pipeline_config,
        orchestrator=None,
        step_state_map=None,
        step_snapshots: tuple[StepSnapshot, ...] | None = None,
        declaration_provider: InvocationArtifactDeclarationProviderLike = (
            callable_contract_artifact_declarations
        ),
    ) -> Dict:
        """
        Prepare pipeline paths.

        Args:
            context: ProcessingContext with step_plans
            pipeline_definition: List of pipeline steps
            pipeline_config: Merged GlobalPipelineConfig (from context.global_config)
                           NOT the raw PipelineConfig - ensures proper global config inheritance
            orchestrator: Optional orchestrator for component key resolution
            step_state_map: Optional dict mapping step_index to ObjectState for building snapshots
            step_snapshots: Optional prebuilt ObjectState-resolved step snapshots
            declaration_provider: Invocation-aware artifact declaration provider
        """
        if step_snapshots is None:
            if step_state_map is None:
                raise ValueError(
                    "PipelinePathPlanner requires StepSnapshot objects or "
                    "step_state_map to avoid live step/config probing."
                )
            step_snapshots = build_step_snapshots(
                pipeline_definition,
                step_state_map,
            )
        planning_context = PathPlanningContext(
            context=context,
            pipeline_config=pipeline_config,
            orchestrator=orchestrator,
            step_snapshots=step_snapshots,
            declaration_provider=declaration_provider,
        )
        return PathPlanner(planning_context).plan(pipeline_definition)

    @staticmethod
    def build_output_plate_root(
        plate_path: Path,
        path_config,
        is_per_step_materialization: bool = False,
    ) -> Path:
        """Build output plate root from configuration components."""
        return PathPlannerPathAuthority.build_output_plate_root(
            plate_path,
            path_config,
            is_per_step_materialization=is_per_step_materialization,
        )

    @staticmethod
    def _build_axis_filename(axis_id: str, key: str, extension: str = "pkl", step_index: Optional[int] = None) -> str:
        """Build standardized axis-based filename with optional step index.

        Args:
            axis_id: Well/axis identifier (e.g., "R02C02")
            key: Artifact output key (e.g., "match_results")
            extension: File extension (default: "pkl")
            step_index: Optional step index to prevent collisions when multiple steps
                       produce the same artifact output

        Returns:
            Filename string (e.g., "R02C02_match_results_step3.pkl")
        """
        if step_index is not None:
            return f"{axis_id}_{key}_step{step_index}.{extension}"
        return f"{axis_id}_{key}.{extension}"

    @staticmethod
    def build_dict_pattern_path(base_path: str, dict_key: str) -> str:
        """Build channel-specific path for dict patterns.

        Inserts _w{dict_key} after well ID in the filename.
        Example: "dir/A01_rois_step7.pkl" + "1" -> "dir/A01_w1_rois_step7.pkl"

        Args:
            base_path: Base path without channel component
            dict_key: Dict pattern key (e.g., "1" for channel 1)

        Returns:
            Channel-specific path
        """
        # Use Path for cross-platform path handling (Windows uses backslashes)
        path = Path(base_path)
        dir_part = path.parent
        filename = path.name
        well_id, rest = filename.split('_', 1)
        return str(dir_part / f"{well_id}_w{dict_key}_{rest}")




# ===== METADATA =====

METADATA_RESOLVERS = {
    "grid_dimensions": {
        "resolver": lambda context: context.microscope_handler.get_grid_dimensions(context.plate_path),
        "description": "Grid dimensions (num_rows, num_cols) for position generation functions"
    },
}

def resolve_metadata(key: str, context) -> Any:
    """Resolve metadata value."""
    if key not in METADATA_RESOLVERS:
        raise ValueError(f"No resolver for '{key}'")
    return METADATA_RESOLVERS[key]["resolver"](context)




def register_metadata_resolver(key: str, resolver: Callable, description: str):
    """Register metadata resolver."""
    METADATA_RESOLVERS[key] = {"resolver": resolver, "description": description}
