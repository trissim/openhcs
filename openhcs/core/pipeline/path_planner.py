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


# ===== PATH PLANNING (NO duplication) =====

class PathPlanner:
    """Minimal path planner with zero duplication."""

    def __init__(
        self,
        context: ProcessingContext,
        pipeline_config,
        orchestrator=None,
        step_snapshots: tuple[StepSnapshot, ...] = (),
    ):
        self.ctx = context
        # CRITICAL: pipeline_config is now the merged config (GlobalPipelineConfig) from context.global_config
        # This ensures proper inheritance from global config without needing field-specific code
        self.cfg = pipeline_config.path_planning_config
        self.vfs = pipeline_config.vfs_config
        self.plans: dict[int, CompiledStepPlan] = context.step_plans
        self.declared = {}  # Tracks artifact outputs
        self.orchestrator = orchestrator
        self.step_snapshots = tuple(step_snapshots)
        self.snapshots_by_index = {
            snapshot.index: snapshot for snapshot in self.step_snapshots
        }
        self.future_artifact_inputs: List[Set[str]] = [
            set() for _ in self.step_snapshots
        ]

        # Initial input determination (once)
        self.initial_input = Path(context.input_dir)
        self.plate_path = Path(context.plate_path)

    @staticmethod
    def _normalize_group_key(key: Optional[Any]) -> Optional[str]:
        if key is None:
            return None
        return str(key)

    def _get_execution_groups(self, snapshot: StepSnapshot) -> List[Optional[str]]:
        """Determine which component groups this step will execute for."""
        if not snapshot.is_function_step:
            return [None]

        func_pattern = snapshot.func
        if isinstance(func_pattern, dict):
            result = [self._normalize_group_key(k) for k in func_pattern.keys()]
            logger.debug("Dict function pattern groups: %s", result)
            return result

        group_by = snapshot.group_by
        logger.debug(
            "Resolved group_by for step %s via StepSnapshot: %s",
            snapshot.name,
            group_by,
        )

        if not self._group_by_requires_component_keys(group_by):
            logger.debug("No group_by configured; using a single ungrouped execution.")
            return [None]

        if self.orchestrator is None:
            logger.warning(
                "PathPlanner: orchestrator not available; cannot resolve "
                "group_by component keys for artifact planning."
            )
            return [None]

        try:
            result = [self._normalize_group_key(k) for k in self.orchestrator.get_component_keys(group_by)]
            logger.debug("Resolved execution groups from orchestrator: %s", result)
            return result
        except Exception as e:
            logger.warning(f"PathPlanner: failed to resolve component keys for {group_by}: {e}")
            return [None]

    @staticmethod
    def _group_by_requires_component_keys(group_by: Any) -> bool:
        from openhcs.constants import GroupBy

        if group_by is None or group_by == GroupBy.NONE:
            return False
        return group_by.value is not None

    @staticmethod
    def _build_paths_by_group(base_path: str, group_keys: List[Optional[str]]) -> Dict[Optional[str], str]:
        from openhcs.core.pipeline.path_planner import PipelinePathPlanner

        paths_by_group: Dict[Optional[str], str] = {}
        for group_key in group_keys:
            if group_key is None:
                paths_by_group[group_key] = base_path
            else:
                paths_by_group[group_key] = PipelinePathPlanner.build_dict_pattern_path(base_path, group_key)
        return paths_by_group

    @staticmethod
    def _build_artifact_outputs_by_group(
        artifact_outputs: Dict[str, ArtifactOutputPlan]
    ) -> Dict[Optional[str], OrderedDict]:
        """Expand artifact outputs into per-group plans with finalized paths."""
        if not artifact_outputs:
            return {}

        grouped: Dict[Optional[str], OrderedDict] = defaultdict(OrderedDict)
        for output_key, output_plan in artifact_outputs.items():
            paths_by_group = output_plan.paths_by_group or {None: output_plan.path}
            for group_key, group_path in paths_by_group.items():
                grouped[group_key][output_key] = output_plan.for_group(group_key)
        return dict(grouped)

    @staticmethod
    def _build_artifact_inputs_by_group(
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

    def plan(self, pipeline: List[AbstractStep]) -> dict[int, CompiledStepPlan]:
        """Plan all paths with zero duplication."""
        if len(self.step_snapshots) != len(pipeline):
            raise ValueError(
                "PathPlanner requires one StepSnapshot per pipeline step: "
                f"{len(self.step_snapshots)} snapshots for {len(pipeline)} steps."
            )

        self._prime_future_artifact_inputs()
        for i, snapshot in enumerate(self.step_snapshots):
            self._plan_step(snapshot, i)

        self._validate(pipeline)

        # Set output_plate_root and sub_dir for metadata writing
        if pipeline:
            self.ctx.output_plate_root = self.build_output_plate_root(self.plate_path, self.cfg, is_per_step_materialization=False)
            self.ctx.sub_dir = self.cfg.sub_dir



        return self.plans

    def _prime_future_artifact_inputs(self) -> None:
        """Precompute artifact input keys used by later steps for each step index."""
        future_inputs: Set[str] = set()
        self.future_artifact_inputs = [set() for _ in self.step_snapshots]

        for i in range(len(self.step_snapshots) - 1, -1, -1):
            self.future_artifact_inputs[i] = set(future_inputs)

            snapshot = self.step_snapshots[i]
            if snapshot.is_function_step:
                pattern = (
                    strip_disabled_functions(snapshot.func)
                    if snapshot.func
                    else []
                )
                declarations = extract_artifact_declarations(pattern)
                step_inputs = set(declarations.inputs.keys())
            else:
                step_inputs = set()

            future_inputs.update(step_inputs)

    def _prepare_step_declarations(
        self,
        snapshot: StepSnapshot,
    ) -> tuple[ArtifactGraph, List[Optional[str]], Any]:
        """Normalize a step's function pattern and collect artifact declarations."""
        if not snapshot.is_function_step:
            return ArtifactGraph.empty(), [None], None

        func_pattern = self._inject_injectable_params(snapshot.func, snapshot)
        func_pattern = strip_disabled_functions(func_pattern)

        declarations = extract_artifact_declarations(func_pattern if func_pattern else [])
        execution_groups = self._get_execution_groups(snapshot)
        declarations = self._namespace_grouped_outputs_for_runtime_consumers(
            snapshot,
            func_pattern,
            declarations,
            execution_groups,
        )
        return declarations, execution_groups, func_pattern

    def _namespace_grouped_outputs_for_runtime_consumers(
        self,
        snapshot: StepSnapshot,
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

        future_inputs = self.future_artifact_inputs[snapshot.index]
        output_groups = {
            output_key: (
                (None,)
                if output_key in future_inputs
                else tuple(self._normalize_group_key(group) for group in execution_groups)
            )
            for output_key in declarations.output_names
        }
        return declarations.with_output_groups(output_groups)

    def _compile_artifact_plan_maps(
        self,
        snapshot: StepSnapshot,
        step_index: int,
        declarations: ArtifactGraph,
        execution_groups: List[Optional[str]],
    ) -> ArtifactPlanMaps:
        """Compile artifact declarations into runtime I/O maps."""
        step_name = snapshot.name
        artifact_outputs = self._process_artifact_outputs(
            declarations.outputs,
            step_index,
            declarations.output_groups,
            step_name=step_name,
        )
        artifact_inputs = self._process_artifact_inputs(
            declarations.inputs,
            declarations.outputs,
            step_index,
            consumer_groups=execution_groups,
            step_name=step_name,
        )
        normalized_groups = [
            self._normalize_group_key(group) for group in execution_groups
        ]

        return ArtifactPlanMaps(
            declarations=declarations,
            execution_groups=execution_groups,
            inputs=artifact_inputs,
            outputs=artifact_outputs,
            inputs_by_group=self._build_artifact_inputs_by_group(
                artifact_inputs,
                normalized_groups,
            ),
            outputs_by_group=self._build_artifact_outputs_by_group(
                artifact_outputs
            ),
        )

    @staticmethod
    def _build_step_compiled_function_pattern(
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
        )

    @staticmethod
    def _analysis_results_dir_for(image_dir: Path) -> Path:
        """Return the analysis-results sibling directory for an image directory."""
        return image_dir.parent / f"{image_dir.name}_results"

    def _materialized_output_dir_for_step(
        self,
        snapshot: StepSnapshot,
    ) -> Optional[Path]:
        """Resolve optional per-step materialization output directory."""
        materialization_config = snapshot.materialization_config
        if not materialization_config or not materialization_config.enabled:
            return None

        step_axis_filters = self.ctx.step_axis_filters.get(
            snapshot.index,
            {},
        )
        materialization_filter = step_axis_filters.get(
            "step_materialization_config"
        )
        if materialization_filter:
            should_materialize = (
                self.ctx.axis_id
                in materialization_filter["resolved_axis_values"]
            )
            if not should_materialize:
                logger.debug(
                    "Skipping materialization for step %s, axis %s (filtered out)",
                    snapshot.name,
                    self.ctx.axis_id,
                )
                return None

        return self._build_output_path(materialization_config)

    def _input_conversion_plan_for_step(
        self,
        step_index: int,
        input_dir: Path,
    ) -> Optional[InputConversionPlan]:
        """Resolve optional compiler-provided or config-provided input conversion."""
        existing_plan = self.plans[step_index].input_conversion
        if existing_plan is not None:
            return existing_plan

        output_dir = self._input_conversion_output_path(step_index)
        if output_dir is None:
            return None

        return InputConversionPlan(
            output_dir=output_dir,
            backend=self.vfs.materialization_backend.value,
            uses_virtual_workspace=False,
            original_subdir=input_dir.name,
        )

    def _update_core_step_plan(
        self,
        snapshot: StepSnapshot,
        step_index: int,
        input_dir: Path,
        output_dir: Path,
        artifact_maps: ArtifactPlanMaps,
        compiled_function_pattern: CompiledFunctionPattern | None,
    ) -> None:
        """Write the always-present path and artifact planning fields."""
        main_plate_root = self.build_output_plate_root(
            self.plate_path,
            self.cfg,
            is_per_step_materialization=False,
        )
        step_plan = self.plans[step_index]
        step_plan.input_dir = input_dir
        step_plan.output_dir = output_dir
        step_plan.output_plate_root = str(main_plate_root)
        step_plan.sub_dir = self.cfg.sub_dir
        step_plan.analysis_results_dir = str(
            self._analysis_results_dir_for(Path(output_dir))
        )
        step_plan.pipeline_position = step_index
        step_plan.input_source = self._get_input_source(snapshot)
        step_plan.artifact_inputs = artifact_maps.inputs
        step_plan.artifact_outputs = artifact_maps.outputs
        step_plan.artifact_inputs_by_group = artifact_maps.inputs_by_group
        step_plan.artifact_outputs_by_group = artifact_maps.outputs_by_group
        step_plan.execution_groups = artifact_maps.execution_groups
        step_plan.compiled_function_pattern = compiled_function_pattern

    def _apply_materialization_plan(
        self,
        snapshot: StepSnapshot,
        step_index: int,
        materialized_output_dir: Optional[Path],
    ) -> None:
        """Attach optional materialization path fields to a step plan."""
        if not materialized_output_dir:
            return

        materialization_config = snapshot.materialization_config
        materialized_plate_root = self.build_output_plate_root(
            self.plate_path,
            materialization_config,
            is_per_step_materialization=False,
        )
        self.plans[step_index].materialized_output = MaterializedOutputPlan(
            output_dir=materialized_output_dir,
            backend=self.vfs.materialization_backend.value,
            plate_root=str(materialized_plate_root),
            sub_dir=materialization_config.sub_dir,
            analysis_results_dir=str(
                self._analysis_results_dir_for(materialized_output_dir)
            ),
        )
        self.plans[step_index].materialization_config = materialization_config

    def _apply_input_conversion_plan(
        self,
        step_index: int,
        input_conversion_plan: Optional[InputConversionPlan],
    ) -> None:
        """Attach optional input conversion path fields to a step plan."""
        if input_conversion_plan is None:
            return

        self.plans[step_index].input_conversion = input_conversion_plan

    def _plan_step(self, snapshot: StepSnapshot, i: int):
        """Plan one step - no duplicate logic."""
        sid = i  # Use step index instead of step_id

        input_dir = self._input_dir_for_step(snapshot, i)
        output_dir = self._output_dir_for_step(snapshot, i, input_dir)

        declarations, execution_groups, func_pattern = self._prepare_step_declarations(
            snapshot,
        )
        artifact_maps = self._compile_artifact_plan_maps(
            snapshot,
            sid,
            declarations,
            execution_groups,
        )

        # Handle metadata injection after stripping disabled functions
        if snapshot.is_function_step and any(
            k in METADATA_RESOLVERS for k in declarations.inputs
        ):
            func_pattern = self._inject_metadata(func_pattern, declarations.inputs)

        # Ensure step plan references the normalized function pattern
        self.plans[sid].func = func_pattern

        self._update_core_step_plan(
            snapshot,
            sid,
            input_dir,
            output_dir,
            artifact_maps,
            self._build_step_compiled_function_pattern(
                snapshot.is_function_step,
                func_pattern,
                artifact_maps.inputs,
                artifact_maps.outputs,
            ),
        )
        self._apply_materialization_plan(
            snapshot,
            sid,
            self._materialized_output_dir_for_step(snapshot),
        )
        self._apply_input_conversion_plan(
            sid,
            self._input_conversion_plan_for_step(sid, input_dir),
        )

        # PIPELINE_START steps read from original input, not zarr conversion
        # (zarr conversion only applies to normal pipeline flow, not PIPELINE_START jumps)

    def _input_dir_for_step(self, snapshot: StepSnapshot, step_index: int) -> Path:
        """Resolve where this step reads from."""
        if step_index in self.plans and self.plans[step_index].input_dir is not None:
            return Path(self.plans[step_index].input_dir)
        if step_index == 0 or snapshot.input_source == InputSource.PIPELINE_START:
            return self.initial_input
        return Path(self.plans[step_index - 1].output_dir)

    def _output_dir_for_step(
        self,
        snapshot: StepSnapshot,
        step_index: int,
        work_in_place_dir: Path,
    ) -> Path:
        """Resolve where this step writes to."""
        if step_index in self.plans and self.plans[step_index].output_dir is not None:
            return Path(self.plans[step_index].output_dir)
        if step_index == 0 or snapshot.input_source == InputSource.PIPELINE_START:
            return self._build_output_path()
        return work_in_place_dir

    @staticmethod
    def build_output_plate_root(plate_path: Path, path_config, is_per_step_materialization: bool = False) -> Path:
        """Build output plate root directory directly from configuration components.

        Formula: (global_output_folder OR plate_path.parent) + plate_name + output_dir_suffix

        Results (analysis outputs) should ALWAYS use the output plate path, never the input plate path.
        This ensures metadata coherence - ROIs and other analysis results are saved alongside the
        processed images they were created from, not with the original input images.

        Args:
            plate_path: Path to the original plate directory
            path_config: PathPlanningConfig with global_output_folder and output_dir_suffix
            is_per_step_materialization: Unused (kept for API compatibility)

        Returns:
            Path to plate root directory (e.g., "/data/results/plate001_processed")
        """

        # OMERO paths always use /omero as base, ignore global_output_folder
        if str(plate_path).startswith("/omero/"):
            base = plate_path.parent
        elif path_config.global_output_folder:
            base = Path(path_config.global_output_folder)
        else:
            base = plate_path.parent

        # Always append suffix to create output plate path
        # If suffix is None/empty, fail loud - this is a configuration error
        if not path_config.output_dir_suffix:
            raise ValueError(
                f"output_dir_suffix cannot be None or empty. "
                f"Results must always use output plate path, not input plate path. "
                f"Config: {path_config}"
            )

        result = base / f"{plate_path.name}{path_config.output_dir_suffix}"
        return result

    def _build_output_path(self, path_config=None) -> Path:
        """Build complete output path: plate_root + sub_dir"""
        config = path_config or self.cfg

        # Use the config's own output_dir_suffix to determine plate root
        plate_root = self.build_output_plate_root(self.plate_path, config, is_per_step_materialization=False)
        return plate_root / config.sub_dir

    def _input_conversion_output_path(self, step_index: int) -> Optional[Path]:
        """Get input conversion output path if config exists."""
        config = self.plans[step_index].input_conversion_config
        if config is not None:
            return self._build_output_path(config)
        return None

    def _process_artifact_outputs(
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

        results_path = self._get_results_path()
        for key, spec in sorted(outputs.items()):
            # Include step index in filename to prevent collisions when multiple steps
            # produce the same artifact output (e.g., two crop_device steps both producing match_results)
            filename = PipelinePathPlanner._build_axis_filename(
                self.ctx.axis_id,
                key,
                step_index=sid,
            )
            path = results_path / filename
            groups = output_groups.get(key, {None}) if output_groups else {None}
            normalized_groups = sorted({self._normalize_group_key(g) for g in groups})
            paths_by_group = self._build_paths_by_group(
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
                producer_step_name=step_name,
            )
            self.declared[key] = result[key]

        return result

    def _process_artifact_inputs(
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
            self._normalize_group_key(g) for g in consumer_groups
        ]

        for key, input_spec in sorted(inputs.items()):
            if key in self.declared:
                producer = self.declared[key]
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
                    producer_name = (
                        producer.producer_step_name
                        or producer.producer_step_index
                        or "unknown"
                    )
                    consumer_name = step_name or sid
                    raise ValueError(
                        f"Ambiguous artifact input '{key}' in step '{consumer_name}': "
                        f"producer step '{producer_name}' provides group-specific outputs {producer_groups}, "
                        f"but the consumer is not grouped. Use a dict pattern or set group_by to match."
                    )

                if producer_groups != [None]:
                    missing = [
                        group
                        for group in normalized_consumers
                        if group not in producer_groups
                    ]
                    if missing:
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

                result[key] = ArtifactInputPlan(
                    name=key,
                    path=producer.path,
                    kind=producer.kind,
                    paths_by_group=paths_by_group,
                    group_keys=tuple(producer_groups),
                    source_step_id=producer.producer_step_index,
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
                )
            elif key not in METADATA_RESOLVERS:
                raise ValueError(f"Step {sid} needs '{key}' but it's not available")

        return result

    def _inject_metadata(self, pattern: Any, inputs: Dict) -> Any:
        """Inject metadata for artifact inputs."""
        for key in inputs:
            if key in METADATA_RESOLVERS and key not in self.declared:
                value = METADATA_RESOLVERS[key]["resolver"](self.ctx)
                pattern = inject_artifact_input_values(pattern, {key: value})
        return pattern

    def _inject_injectable_params(
        self,
        pattern: Any,
        snapshot: StepSnapshot,
    ) -> Any:
        """Inject injectable param values into function kwargs.

        Injectable params (dtype_config, enabled, etc.) are added to function signatures
        by the unified registry. This method injects those params from the step into the
        func pattern kwargs. Values come from the ObjectState-backed StepSnapshot.

        Args:
            pattern: Function pattern (callable, tuple, list, or dict)
            snapshot: ObjectState-resolved compiler facts for the step

        Returns:
            Modified pattern with param values injected into kwargs
        """
        from openhcs.processing.backends.lib_registry.unified_registry import LibraryRegistryBase

        # Get injectable param names from registry (single source of truth)
        param_names = [param_name for param_name, _, _ in LibraryRegistryBase.INJECTABLE_PARAMS]

        # Build kwargs dict from snapshot values, not live step attributes.
        param_kwargs = {}
        for param_name in param_names:
            value = snapshot.injectable_values.get(param_name)
            if value is not None:
                param_kwargs[param_name] = value

        if not param_kwargs:
            return pattern

        return inject_kwargs_into_pattern(pattern, param_kwargs)

    def _get_input_source(self, snapshot: StepSnapshot) -> str:
        """Get input source string."""
        if snapshot.input_source == InputSource.PIPELINE_START:
            return 'PIPELINE_START'
        return 'PREVIOUS_STEP'

    def _get_results_path(self) -> Path:
        """Get results path from global pipeline configuration.

        Results must always be stored in the OUTPUT plate, not the input plate.
        This ensures metadata coherence - analysis results are saved alongside the
        processed images they were created from.
        """
        # Access materialization_results_path from global config, not path planning config.
        path = self.ctx.global_config.materialization_results_path

        # Build output plate root to ensure results go to output plate.
        output_plate_root = self.build_output_plate_root(self.plate_path, self.cfg, is_per_step_materialization=False)

        return Path(path) if Path(path).is_absolute() else output_plate_root / path

    def _validate(self, pipeline: List):
        """Validate connectivity and materialization paths - no duplication."""
        # Existing connectivity validation
        for i in range(1, len(self.step_snapshots)):
            curr = self.step_snapshots[i]
            prev = self.step_snapshots[i - 1]
            if curr.input_source == InputSource.PIPELINE_START:
                continue
            curr_in = self.plans[i].input_dir
            prev_out = self.plans[i - 1].output_dir
            if curr_in != prev_out:
                has_artifact_bridge = any(
                    inp.source_step_id in [i - 1, 'prev']
                    for inp in self.plans[i].artifact_inputs.values()
                )
                if not has_artifact_bridge:
                    raise ValueError(f"Disconnect: {prev.name} -> {curr.name}")

        # NEW: Materialization path collision validation
        self._validate_materialization_paths(pipeline)

    def _validate_materialization_paths(self, pipeline: List[AbstractStep]) -> None:
        """Validate and resolve materialization path collisions with symmetric conflict resolution."""
        global_path = self._build_output_path(self.cfg)

        # Collect all materialization steps with their paths and positions
        mat_steps = [
            (
                snapshot,
                self.plans[i].pipeline_position or i,
                self._build_output_path(snapshot.materialization_config),
            )
            for i, snapshot in enumerate(self.step_snapshots)
            if snapshot.materialization_config
            and snapshot.materialization_config.enabled
        ]

        # Group by path for conflict detection
        from collections import defaultdict
        path_groups = defaultdict(list)
        for snapshot, pos, path in mat_steps:
            if path == global_path:
                self._resolve_and_update_paths(snapshot, pos, path, "main flow")
            else:
                path_groups[str(path)].append((snapshot, pos, path))

        # Resolve materialization vs materialization conflicts
        for path_key, step_list in path_groups.items():
            if len(step_list) > 1:
                for snapshot, pos, path in step_list:
                    self._resolve_and_update_paths(snapshot, pos, path, f"pos {pos}")

    def _resolve_and_update_paths(
        self,
        snapshot: StepSnapshot,
        position: int,
        original_path: Path,
        conflict_type: str,
    ) -> None:
        """Resolve path conflict by updating the compiled plan only."""
        materialization_config = snapshot.materialization_config

        # Generate unique sub_dir name instead of calculating from paths
        original_sub_dir = materialization_config.sub_dir
        new_sub_dir = f"{original_sub_dir}_step{position}"

        from dataclasses import replace
        updated_config = replace(materialization_config, sub_dir=new_sub_dir)

        # Recalculate the resolved path using the updated config
        resolved_path = self._build_output_path(updated_config)
        resolved_analysis_results_dir = self._analysis_results_dir_for(resolved_path)

        # Update step plans for metadata generation
        if step_plan := self.plans.get(position):
            if step_plan.materialized_output is not None:
                step_plan.materialized_output = MaterializedOutputPlan(
                    output_dir=resolved_path,
                    backend=step_plan.materialized_output.backend,
                    plate_root=step_plan.materialized_output.plate_root,
                    sub_dir=new_sub_dir,
                    analysis_results_dir=str(resolved_analysis_results_dir),
                )
                step_plan.materialization_config = updated_config



# ===== PUBLIC API =====

class PipelinePathPlanner:
    """Public API matching original interface."""

    @staticmethod
    def prepare_pipeline_paths(context: ProcessingContext,
                              pipeline_definition: List[AbstractStep],
                              pipeline_config,
                              orchestrator=None,
                              step_state_map=None,
                              step_snapshots: tuple[StepSnapshot, ...] | None = None) -> Dict:
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
        return PathPlanner(
            context,
            pipeline_config,
            orchestrator=orchestrator,
            step_snapshots=step_snapshots,
        ).plan(pipeline_definition)

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
