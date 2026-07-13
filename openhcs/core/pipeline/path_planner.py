"""
Pipeline path planning - actually reduced duplication.

This version ACTUALLY eliminates duplication instead of adding abstraction theater.
"""

import logging
from collections import defaultdict, OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

from openhcs.constants.constants import READ_BACKEND, WRITE_BACKEND, Backend
from openhcs.constants.input_source import InputSource
from openhcs.core.config import MaterializationBackend
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.formats.func_arg_prep import get_core_callable, iter_pattern_items
from openhcs.core.steps.abstract import AbstractStep
from openhcs.core.steps.function_step import FunctionStep

logger = logging.getLogger(__name__)


# ===== PATTERN NORMALIZATION (ONE place) =====

def normalize_pattern(pattern: Any) -> Iterator[Tuple[Callable, str, int]]:
    """Extract enabled functions from any pattern.

    Renumbers positions after filtering out disabled functions to ensure
    funcplan keys match runtime execution positions.

    For dict patterns, position counters are tracked per dict key.
    For list/single patterns, position counter is global.
    """
    # Track position counters per dict key (for dict patterns) or globally (for list/single patterns)
    position_counters = {}

    for func, key, original_pos in iter_pattern_items(pattern):
        # Skip disabled functions
        if isinstance(func, tuple) and len(func) == 2 and isinstance(func[1], dict):
            if func[1].get('enabled', True) is False:
                continue
        # Extract callable and yield with renumbered position
        if core := get_core_callable(func):
            # Get or initialize position counter for this dict key
            if key not in position_counters:
                position_counters[key] = 0

            yield (core, key, position_counters[key])
            position_counters[key] += 1


def extract_attributes(pattern: Any) -> Dict[str, Any]:
    """Extract special I/O metadata and track per-group ownership."""
    output_names: Set[str] = set()
    output_groups: Dict[str, Set[Optional[str]]] = defaultdict(set)
    inputs, mat_specs = {}, {}

    for func, group_key, _ in normalize_pattern(pattern):
        normalized_key = None if group_key == "default" else group_key

        func_outputs = getattr(func, '__special_outputs__', set())
        output_names.update(func_outputs)
        for output in func_outputs:
            output_groups[output].add(normalized_key)

        inputs.update(getattr(func, '__special_inputs__', {}))
        mat_specs.update(getattr(func, '__materialization_specs__', {}))

    return {
        'outputs': {
            'names': output_names,
            'groups': output_groups
        },
        'inputs': inputs,
        'mat_specs': mat_specs
    }


# ===== PATH PLANNING (NO duplication) =====

class PathPlanner:
    """Minimal path planner with zero duplication."""

    def __init__(self, context: ProcessingContext, pipeline_config, orchestrator=None, step_state_map=None):
        self.ctx = context
        # CRITICAL: pipeline_config is now the merged config (GlobalPipelineConfig) from context.global_config
        # This ensures proper inheritance from global config without needing field-specific code
        self.cfg = pipeline_config.path_planning_config
        self.vfs = pipeline_config.vfs_config
        self.plans = context.step_plans
        self.declared = {}  # Tracks special outputs
        self.orchestrator = orchestrator
        self.step_state_map = step_state_map  # For resolving lazy dataclass attributes via ObjectState

        # Initial input determination (once)
        self.initial_input = Path(context.input_dir)
        self.plate_path = Path(context.plate_path)

    def _get_step_config_value(
        self,
        step: AbstractStep,
        step_index: int,
        dotted_path: str,
        default: Any = None,
    ) -> Any:
        """Read a step configuration value through its compilation snapshot.

        Lazy configuration objects intentionally retain ``None`` for inherited
        fields. Direct attribute access outside an ObjectState context therefore
        falls back to static class defaults instead of pipeline-level overrides.
        Compilation must read every configuration through the step's saved
        ObjectState snapshot so inheritance works for arbitrary config types and
        dotted field paths.

        Direct traversal is retained only for callers that do not provide a
        ``step_state_map`` (primarily isolated utilities and legacy tests).
        """
        if self.step_state_map and step_index in self.step_state_map:
            return self.step_state_map[step_index].get_saved_resolved_value(dotted_path)

        value: Any = step
        for component in dotted_path.split("."):
            if value is None or not hasattr(value, component):
                return default
            value = getattr(value, component)
        return value

    @staticmethod
    def _normalize_group_key(key: Optional[Any]) -> Optional[str]:
        if key is None:
            return None
        return str(key)

    def _get_execution_groups(self, step: AbstractStep, step_index: int) -> List[Optional[str]]:
        """Determine which component groups this step will execute for."""
        from openhcs.constants import GroupBy

        if not isinstance(step, FunctionStep):
            return [None]

        func_pattern = step.func
        if isinstance(func_pattern, dict):
            result = [self._normalize_group_key(k) for k in func_pattern.keys()]
            logger.info(f"🔍 PATH_PLANNER: Dict pattern detected, groups={result}")
            return result

        group_by = self._get_step_config_value(
            step, step_index, "processing_config.group_by"
        )

        if not group_by or group_by == GroupBy.NONE or getattr(group_by, "value", None) is None:
            logger.info(f"🔍 PATH_PLANNER: No group_by, returning [None]")
            return [None]

        if self.orchestrator is None:
            logger.warning(
                "PathPlanner: orchestrator not available; "
                "cannot resolve group_by component keys for special I/O planning."
            )
            return [None]

        try:
            result = [self._normalize_group_key(k) for k in self.orchestrator.get_component_keys(group_by)]
            logger.info(f"🔍 PATH_PLANNER: Resolved groups from orchestrator: {result}")
            return result
        except Exception as e:
            logger.warning(f"PathPlanner: failed to resolve component keys for {group_by}: {e}")
            return [None]

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
    def _build_special_outputs_by_group(special_outputs: Dict) -> Dict[Optional[str], OrderedDict]:
        """Expand special outputs into per-group plans with finalized paths."""
        if not special_outputs:
            return {}

        grouped: Dict[Optional[str], OrderedDict] = defaultdict(OrderedDict)
        for output_key, output_info in special_outputs.items():
            paths_by_group = output_info.get("paths_by_group") or {None: output_info.get("path")}
            for group_key, group_path in paths_by_group.items():
                info = output_info.copy()
                info["path"] = group_path
                grouped[group_key][output_key] = info
        return dict(grouped)

    @staticmethod
    def _build_special_inputs_by_group(special_inputs: Dict, consumer_groups: List[Optional[str]]) -> Dict[Optional[str], OrderedDict]:
        """Expand special inputs into per-group plans with finalized paths."""
        if not special_inputs:
            return {}

        grouped: Dict[Optional[str], OrderedDict] = {}
        for group_key in consumer_groups:
            per_group = OrderedDict()
            for input_key, input_info in special_inputs.items():
                paths_by_group = input_info.get("paths_by_group")
                if paths_by_group:
                    if group_key in paths_by_group:
                        path = paths_by_group[group_key]
                    elif None in paths_by_group:
                        path = paths_by_group[None]
                    else:
                        continue
                else:
                    path = input_info.get("path")
                info = input_info.copy()
                info["path"] = path
                per_group[input_key] = info
            grouped[group_key] = per_group
        return grouped

    def plan(self, pipeline: List[AbstractStep]) -> Dict:
        """Plan all paths with zero duplication."""
        self._prime_future_special_inputs(pipeline)
        for i, step in enumerate(pipeline):
            self._plan_step(step, i, pipeline)

        self._validate(pipeline)

        # Set output_plate_root and sub_dir for metadata writing
        if pipeline:
            self.ctx.output_plate_root = self.build_output_plate_root(self.plate_path, self.cfg, is_per_step_materialization=False)
            self.ctx.sub_dir = self.cfg.sub_dir



        return self.plans

    def _prime_future_special_inputs(self, pipeline: List[AbstractStep]) -> None:
        """Precompute special input keys used by later steps for each step index."""
        future_inputs: Set[str] = set()
        self.future_special_inputs: List[Set[str]] = [set() for _ in pipeline]

        for i in range(len(pipeline) - 1, -1, -1):
            self.future_special_inputs[i] = set(future_inputs)

            step = pipeline[i]
            if isinstance(step, FunctionStep):
                pattern = self._strip_disabled_functions(step.func) if step.func else []
                attrs = extract_attributes(pattern)
                step_inputs = set(attrs.get("inputs", {}).keys())
            else:
                step_inputs = set(self._normalize_attr(getattr(step, 'special_inputs', {}), dict).keys())

            future_inputs.update(step_inputs)

    def _plan_step(self, step: AbstractStep, i: int, pipeline: List):
        """Plan one step - no duplicate logic."""
        sid = i  # Use step index instead of step_id

        # Get paths with unified logic
        input_dir = self._get_dir(step, i, pipeline, 'input')
        output_dir = self._get_dir(step, i, pipeline, 'output', input_dir)

        # Prepare function data if FunctionStep
        if isinstance(step, FunctionStep):
            step.func = self._inject_injectable_params(step.func, step)
            step.func = self._strip_disabled_functions(step.func)
            attrs = extract_attributes(step.func if step.func else [])
            execution_groups = self._get_execution_groups(step, i)  # Pass step_index for ObjectState resolution
            # For non-dict patterns grouped by component, namespace outputs only
            # when they are NOT consumed by any later step.
            if not isinstance(step.func, dict) and execution_groups != [None] and attrs["outputs"]["names"]:
                future_inputs = self.future_special_inputs[i] if hasattr(self, "future_special_inputs") else set()
                for out_key in attrs["outputs"]["names"]:
                    if out_key in future_inputs:
                        attrs["outputs"]["groups"][out_key] = {None}
                    else:
                        attrs["outputs"]["groups"][out_key] = set(execution_groups)

        else:
            execution_groups = [None]
            raw_outputs = self._normalize_attr(getattr(step, 'special_outputs', set()), set)
            default_groups = defaultdict(set)
            for name in raw_outputs:
                default_groups[name].add(None)
            attrs = {
                'outputs': {
                    'names': raw_outputs,
                    'groups': default_groups
                },
                'inputs': self._normalize_attr(getattr(step, 'special_inputs', {}), dict),
                'mat_specs': {}
            }

        # Process special I/O with unified logic
        special_outputs = self._process_special(
            attrs['outputs']['names'],
            attrs['mat_specs'],
            'output',
            sid,
            attrs['outputs'].get('groups'),
            consumer_groups=execution_groups,
            step_name=getattr(step, "name", str(sid))
        )
        special_inputs = self._process_special(
            attrs['inputs'],
            attrs['outputs']['names'],
            'input',
            sid,
            consumer_groups=execution_groups,
            step_name=getattr(step, "name", str(sid))
        )

        # Expand into per-group maps for runtime selection
        special_outputs_by_group = self._build_special_outputs_by_group(special_outputs)
        special_inputs_by_group = self._build_special_inputs_by_group(
            special_inputs,
            [self._normalize_group_key(g) for g in execution_groups]
        )

        # Handle metadata injection after stripping disabled functions
        if isinstance(step, FunctionStep) and any(k in METADATA_RESOLVERS for k in attrs['inputs']):
            step.func = self._inject_metadata(step.func, attrs['inputs'])

        # Ensure step plan references the normalized function pattern
        self.plans.setdefault(sid, {})
        self.plans[sid]['func'] = step.func

        # Generate funcplan (only if needed)
        funcplan = {}
        if isinstance(step, FunctionStep) and special_outputs:
            for func, dk, pos in normalize_pattern(step.func):
                saves = [k for k in special_outputs if k in getattr(func, '__special_outputs__', set())]
                if saves:
                    funcplan[f"{func.__name__}_{dk}_{pos}"] = saves

        # Handle optional materialization and input conversion. Resolve the complete
        # config container so every inherited field comes from the saved hierarchy.
        materialization_config = self._get_step_config_value(
            step, i, "step_materialization_config"
        )
        materialized_output_dir = None
        if materialization_config and materialization_config.enabled:
            # Check if this step has well filters and if current well should be materialized
            step_axis_filters = getattr(self.ctx, 'step_axis_filters', {}).get(sid, {})
            materialization_filter = step_axis_filters.get('step_materialization_config')

            if materialization_filter:
                # Check if current axis is in the resolved values
                # Note: resolved_axis_values already has mode (INCLUDE/EXCLUDE) applied
                should_materialize = self.ctx.axis_id in materialization_filter['resolved_axis_values']

                if should_materialize:
                    materialized_output_dir = self._build_output_path(materialization_config)
                else:
                    logger.debug(f"Skipping materialization for step {step.name}, axis {self.ctx.axis_id} (filtered out)")
            else:
                # No axis filter - create materialization path as normal
                materialized_output_dir = self._build_output_path(materialization_config)

        # Check if input_conversion_dir is already set by compiler (direct path)
        # Otherwise try to calculate from input_conversion_config (legacy)
        if "input_conversion_dir" in self.plans[sid]:
            input_conversion_dir = Path(self.plans[sid]["input_conversion_dir"])
        else:
            input_conversion_dir = self._get_optional_path("input_conversion_config", sid)

        # Calculate main pipeline plate root for this step
        main_plate_root = self.build_output_plate_root(self.plate_path, self.cfg, is_per_step_materialization=False)

        # Calculate analysis results directory (sibling to output_dir with _results suffix)
        # This ensures results are saved alongside images at the same hierarchical level
        # Example: images/ -> images_results/, checkpoints_step3/ -> checkpoints_step3_results/
        output_dir_path = Path(output_dir)
        dir_name = output_dir_path.name
        analysis_results_dir = output_dir_path.parent / f"{dir_name}_results"

        # Single update
        self.plans[sid].update({
            'input_dir': str(input_dir),
            'output_dir': str(output_dir),
            'output_plate_root': str(main_plate_root),
            'sub_dir': self.cfg.sub_dir,  # Store resolved sub_dir for main pipeline
            'analysis_results_dir': str(analysis_results_dir),  # Pre-calculated results directory
            'pipeline_position': i,
            'input_source': self._get_input_source(step, i),
            'special_inputs': special_inputs,
            'special_outputs': special_outputs,
            'special_inputs_by_group': special_inputs_by_group,
            'special_outputs_by_group': special_outputs_by_group,
            'execution_groups': execution_groups,
            'funcplan': funcplan,
        })

        # Add optional paths if configured
        if materialized_output_dir:
            # Per-step materialization uses its own config to determine plate root
            materialized_plate_root = self.build_output_plate_root(self.plate_path, materialization_config, is_per_step_materialization=False)

            # Calculate analysis results directory for materialized output
            materialized_dir_path = Path(materialized_output_dir)
            materialized_dir_name = materialized_dir_path.name
            materialized_analysis_results_dir = materialized_dir_path.parent / f"{materialized_dir_name}_results"

            self.plans[sid].update({
                'materialized_output_dir': str(materialized_output_dir),
                'materialized_plate_root': str(materialized_plate_root),
                'materialized_sub_dir': materialization_config.sub_dir,  # Store resolved sub_dir for materialization
                'materialized_analysis_results_dir': str(materialized_analysis_results_dir),  # Pre-calculated materialized results directory
                'materialized_backend': self.vfs.materialization_backend.value,
                'materialization_config': materialization_config
            })
        if input_conversion_dir:
            self.plans[sid].update({
                'input_conversion_dir': str(input_conversion_dir),
                'input_conversion_backend': self.vfs.materialization_backend.value
            })

        # PIPELINE_START steps read from original input, not zarr conversion
        # (zarr conversion only applies to normal pipeline flow, not PIPELINE_START jumps)

    def _get_dir(self, step: AbstractStep, i: int, pipeline: List,
                 dir_type: str, fallback: Path = None) -> Path:
        """Unified directory resolution - no duplication."""
        sid = i  # Use step index instead of step_id

        # Check overrides (same for input/output)
        if override := self.plans.get(sid, {}).get(f'{dir_type}_dir'):
            return Path(override)
        if override := getattr(step, f'__{dir_type}_dir__', None):
            return Path(override)

        # Type-specific logic
        if dir_type == 'input':
            input_source = self._get_step_config_value(
                step, i, "processing_config.input_source"
            )
            if i == 0 or input_source == InputSource.PIPELINE_START:
                return self.initial_input
            prev_step_index = i - 1  # Use previous step index instead of step_id
            return Path(self.plans[prev_step_index]['output_dir'])
        else:  # output
            input_source = self._get_step_config_value(
                step, i, "processing_config.input_source"
            )
            if i == 0 or input_source == InputSource.PIPELINE_START:
                return self._build_output_path()
            return fallback  # Work in place

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

    def _calculate_materialized_output_path(self, materialization_config) -> Path:
        """Calculate materialized output path using custom PathPlanningConfig."""
        return self._build_output_path(materialization_config)

    def _calculate_input_conversion_path(self, conversion_config) -> Path:
        """Calculate input conversion path using custom PathPlanningConfig."""
        return self._build_output_path(conversion_config)

    def _get_optional_path(self, config_key: str, step_index: int) -> Optional[Path]:
        """Get optional path if config exists."""
        if config_key in self.plans[step_index]:
            config = self.plans[step_index][config_key]
            return self._build_output_path(config)
        return None

    def _process_special(
        self,
        items: Any,
        extra: Any,
        io_type: str,
        sid: str,
        output_groups: Optional[Dict[str, Set[Optional[str]]]] = None,
        consumer_groups: Optional[List[Optional[str]]] = None,
        step_name: Optional[str] = None
    ) -> Dict:
        """Unified special I/O processing - no duplication."""
        result = {}

        if io_type == 'output' and items:  # Special outputs
            results_path = self._get_results_path()
            for key in sorted(items):
                # Include step index in filename to prevent collisions when multiple steps
                # produce the same special output (e.g., two crop_device steps both producing match_results)
                filename = PipelinePathPlanner._build_axis_filename(self.ctx.axis_id, key, step_index=sid)
                path = results_path / filename
                groups = output_groups.get(key, {None}) if output_groups else {None}
                normalized_groups = sorted({self._normalize_group_key(g) for g in groups})
                paths_by_group = self._build_paths_by_group(str(path), normalized_groups)
                result[key] = {
                    'path': str(path),
                    'materialization_spec': extra.get(key),  # extra is mat_specs
                    'group_keys': normalized_groups,
                    'paths_by_group': paths_by_group
                }
                self.declared[key] = {
                    'path': str(path),
                    'group_keys': normalized_groups,
                    'paths_by_group': paths_by_group,
                    'step_index': sid,
                    'step_name': step_name
                }

        elif io_type == 'input' and items:  # Special inputs
            consumer_groups = consumer_groups or [None]
            normalized_consumers = [self._normalize_group_key(g) for g in consumer_groups]

            for key in sorted(items.keys() if isinstance(items, dict) else items):
                if key in self.declared:
                    producer = self.declared[key]
                    producer_groups = producer.get("group_keys") or [None]

                    if producer_groups != [None] and normalized_consumers == [None]:
                        producer_name = producer.get("step_name", producer.get("step_index", "unknown"))
                        consumer_name = step_name or sid
                        raise ValueError(
                            f"Ambiguous special input '{key}' in step '{consumer_name}': "
                            f"producer step '{producer_name}' provides group-specific outputs {producer_groups}, "
                            f"but the consumer is not grouped. Use a dict pattern or set group_by to match."
                        )

                    if producer_groups != [None]:
                        missing = [g for g in normalized_consumers if g not in producer_groups]
                        if missing:
                            producer_name = producer.get("step_name", producer.get("step_index", "unknown"))
                            consumer_name = step_name or sid
                            raise ValueError(
                                f"Special input '{key}' in step '{consumer_name}' cannot be resolved: "
                                f"producer step '{producer_name}' provides groups {producer_groups}, "
                                f"but consumer needs {missing}."
                            )
                        paths_by_group = {
                            g: producer["paths_by_group"][g]
                            for g in normalized_consumers
                            if g in producer.get("paths_by_group", {})
                        }
                    else:
                        # Global output: reuse same path for all consumer groups
                        paths_by_group = {g: producer["path"] for g in normalized_consumers}

                    result[key] = {
                        'path': producer["path"],
                        'paths_by_group': paths_by_group,
                        'group_keys': producer_groups,
                        'source_step_id': producer.get("step_index", "prev")
                    }
                elif key in extra:  # extra is outputs (self-fulfilling)
                    result[key] = {'path': 'self', 'source_step_id': sid}
                elif key not in METADATA_RESOLVERS:
                    raise ValueError(f"Step {sid} needs '{key}' but it's not available")

        return result

    def _inject_metadata(self, pattern: Any, inputs: Dict) -> Any:
        """Inject metadata for special inputs."""
        for key in inputs:
            if key in METADATA_RESOLVERS and key not in self.declared:
                value = METADATA_RESOLVERS[key]["resolver"](self.ctx)
                pattern = self._inject_into_pattern(pattern, key, value)
        return pattern

    def _inject_injectable_params(self, pattern: Any, step) -> Any:
        """Inject injectable param values into function kwargs.

        Injectable params (dtype_config, enabled, etc.) are added to function signatures
        by the unified registry. This method injects those params from the step into the
        func pattern kwargs. The values will be resolved during Phase 5 (lazy resolution).

        Args:
            pattern: Function pattern (callable, tuple, list, or dict)
            step: FunctionStep instance with config attributes

        Returns:
            Modified pattern with param values injected into kwargs
        """
        from openhcs.processing.backends.lib_registry.unified_registry import LibraryRegistryBase

        # Get injectable param names from registry (single source of truth)
        param_names = [param_name for param_name, _, _ in LibraryRegistryBase.INJECTABLE_PARAMS]

        # Build kwargs dict from step attributes (keep lazy configs as-is for Phase 5 resolution)
        param_kwargs = {}
        for param_name in param_names:
            if hasattr(step, param_name):
                value = getattr(step, param_name)
                if value is not None:
                    param_kwargs[param_name] = value

        if not param_kwargs:
            return pattern

        return self._inject_params_into_pattern(pattern, param_kwargs)

    def _inject_into_pattern(self, pattern: Any, key: str, value: Any) -> Any:
        """Inject value into pattern - only for functions that declare the special input.

        FunctionReference objects preserve __special_inputs__ via __getattr__, so they
        work the same as regular callables here.
        """
        from openhcs.core.pipeline.compiler import FunctionReference

        # Handle FunctionReference and callable objects
        if isinstance(pattern, FunctionReference) or callable(pattern):
            # Only inject if THIS specific function needs this metadata
            if key in getattr(pattern, '__special_inputs__', {}):
                return (pattern, {key: value})
            return pattern  # Don't modify if function doesn't need it

        if isinstance(pattern, tuple) and len(pattern) == 2:
            func, kwargs = pattern
            # Only inject if THIS specific function needs this metadata
            if (isinstance(func, FunctionReference) or callable(func)) and key in getattr(func, '__special_inputs__', {}):
                return (func, {**kwargs, key: value})
            return pattern  # Don't modify if function doesn't need it

        if isinstance(pattern, list):
            # Recursively process each element (selective injection per function)
            return [self._inject_into_pattern(item, key, value) for item in pattern]

        if isinstance(pattern, dict):
            # Recursively process each value (selective injection per function)
            return {k: self._inject_into_pattern(v, key, value) for k, v in pattern.items()}

        raise ValueError(f"Cannot inject into pattern type: {type(pattern)}")

    def _inject_params_into_pattern(self, pattern: Any, resolved_kwargs: Dict[str, Any]) -> Any:
        """Inject resolved param values into function pattern kwargs.

        Unlike metadata injection which is selective (only for functions with @special_inputs),
        injectable param injection is universal - all registered functions accept dtype_config, enabled, etc.

        Args:
            pattern: Function pattern (callable, tuple, list, or dict)
            resolved_kwargs: Dict of resolved param values to inject

        Returns:
            Modified pattern with params injected into kwargs
        """
        from openhcs.core.pipeline.compiler import FunctionReference

        # Handle FunctionReference and callable objects
        if isinstance(pattern, FunctionReference) or callable(pattern):
            # Always inject params (all registered functions accept them)
            return (pattern, resolved_kwargs)

        if isinstance(pattern, tuple) and len(pattern) == 2:
            func, kwargs = pattern
            # Merge resolved_kwargs with existing kwargs (existing kwargs take precedence)
            merged_kwargs = {**resolved_kwargs, **kwargs}
            return (func, merged_kwargs)

        if isinstance(pattern, list):
            # Recursively process each element
            return [self._inject_params_into_pattern(item, resolved_kwargs) for item in pattern]

        if isinstance(pattern, dict):
            # Recursively process each value
            return {k: self._inject_params_into_pattern(v, resolved_kwargs) for k, v in pattern.items()}

        return pattern

    def _strip_disabled_functions(self, pattern: Any) -> Any:
        """
        Remove disabled functions (enabled=False) from any pattern structure.

        Ensures downstream planning (special outputs, funcplan, materialization)
        never sees disabled functions.
        """
        if isinstance(pattern, tuple) and len(pattern) == 2 and isinstance(pattern[1], dict):
            if pattern[1].get('enabled', True) is False:
                return None
            return pattern

        if isinstance(pattern, list):
            stripped = [self._strip_disabled_functions(item) for item in pattern]
            return [item for item in stripped if item not in (None, [], {})]

        if isinstance(pattern, dict):
            stripped = {k: self._strip_disabled_functions(v) for k, v in pattern.items()}
            return {
                k: v for k, v in stripped.items()
                if v not in (None, [], {})
            }

        return pattern

    def _normalize_attr(self, attr: Any, target_type: type) -> Any:
        """Normalize step attributes - 5 lines, no duplication."""
        if target_type == set:
            return {attr} if isinstance(attr, str) else set(attr) if isinstance(attr, (list, set)) else set()
        else:  # dict
            return {attr: True} if isinstance(attr, str) else {k: True for k in attr} if isinstance(attr, list) else attr if isinstance(attr, dict) else {}

    def _get_input_source(self, step: AbstractStep, i: int) -> str:
        """Get input source string."""
        input_source = self._get_step_config_value(
            step, i, "processing_config.input_source"
        )
        if input_source == InputSource.PIPELINE_START:
            return 'PIPELINE_START'
        return 'PREVIOUS_STEP'

    def _get_results_path(self) -> Path:
        """Get results path from global pipeline configuration.

        Results must always be stored in the OUTPUT plate, not the input plate.
        This ensures metadata coherence - analysis results are saved alongside the
        processed images they were created from.
        """
        try:
            # Access materialization_results_path from global config, not path planning config
            path = self.ctx.global_config.materialization_results_path

            # Build output plate root to ensure results go to output plate
            output_plate_root = self.build_output_plate_root(self.plate_path, self.cfg, is_per_step_materialization=False)

            return Path(path) if Path(path).is_absolute() else output_plate_root / path
        except AttributeError as e:
            # Fallback with clear error message if global config is unavailable
            raise RuntimeError(f"Cannot access global config for materialization_results_path: {e}") from e

    def _validate(self, pipeline: List):
        """Validate connectivity and materialization paths - no duplication."""
        # Existing connectivity validation
        for i in range(1, len(pipeline)):
            curr, prev = pipeline[i], pipeline[i-1]
            input_source = self._get_step_config_value(
                curr, i, "processing_config.input_source"
            )
            if input_source == InputSource.PIPELINE_START:
                continue
            curr_in = self.plans[i]['input_dir']  # Use step index i
            prev_out = self.plans[i-1]['output_dir']  # Use step index i-1
            if curr_in != prev_out:
                has_special = any(inp.get('source_step_id') in [i-1, 'prev']  # Check both step index and 'prev'
                                for inp in self.plans[i].get('special_inputs', {}).values())  # Use step index i
                if not has_special:
                    raise ValueError(f"Disconnect: {prev.name} -> {curr.name}")

        # NEW: Materialization path collision validation
        self._validate_materialization_paths(pipeline)

    def _validate_materialization_paths(self, pipeline: List[AbstractStep]) -> None:
        """Validate and resolve materialization path collisions with symmetric conflict resolution."""
        global_path = self._build_output_path(self.cfg)

        # Collect all materialization steps with fully resolved configurations.
        mat_steps = []
        for i, step in enumerate(pipeline):
            materialization_config = self._get_step_config_value(
                step, i, "step_materialization_config"
            )
            if materialization_config and materialization_config.enabled:
                mat_steps.append(
                    (
                        step,
                        self.plans.get(i, {}).get('pipeline_position', 0),
                        self._build_output_path(materialization_config),
                        materialization_config,
                    )
                )

        # Group by path for conflict detection
        from collections import defaultdict
        path_groups = defaultdict(list)
        for step, pos, path, materialization_config in mat_steps:
            if path == global_path:
                self._resolve_and_update_paths(
                    step, pos, path, "main flow", materialization_config
                )
            else:
                path_groups[str(path)].append(
                    (step, pos, path, materialization_config)
                )

        # Resolve materialization vs materialization conflicts
        for path_key, step_list in path_groups.items():
            if len(step_list) > 1:
                for step, pos, path, materialization_config in step_list:
                    self._resolve_and_update_paths(
                        step,
                        pos,
                        path,
                        f"pos {pos}",
                        materialization_config,
                    )

    def _resolve_and_update_paths(
        self,
        step: AbstractStep,
        position: int,
        original_path: Path,
        conflict_type: str,
        materialization_config,
    ) -> None:
        """Resolve path conflict by updating sub_dir configuration directly."""
        # Generate unique sub_dir name instead of calculating from paths
        original_sub_dir = materialization_config.sub_dir
        new_sub_dir = f"{original_sub_dir}_step{position}"

        # Update the already-resolved config without re-entering lazy resolution.
        from dataclasses import replace
        updated_config = replace(materialization_config, sub_dir=new_sub_dir)
        step.step_materialization_config = updated_config

        # Recalculate the resolved path using the updated config
        resolved_path = self._build_output_path(updated_config)

        # Update step plans for metadata generation
        if step_plan := self.plans.get(position):  # Use position (step index) instead of step_id
            if 'materialized_output_dir' in step_plan:
                step_plan['materialized_output_dir'] = str(resolved_path)
                step_plan['materialized_sub_dir'] = new_sub_dir  # Update stored sub_dir
                step_plan['materialization_config'] = updated_config



# ===== PUBLIC API =====

class PipelinePathPlanner:
    """Public API matching original interface."""

    @staticmethod
    def prepare_pipeline_paths(context: ProcessingContext,
                              pipeline_definition: List[AbstractStep],
                              pipeline_config,
                              orchestrator=None,
                              step_state_map=None) -> Dict:
        """
        Prepare pipeline paths.

        Args:
            context: ProcessingContext with step_plans
            pipeline_definition: List of pipeline steps
            pipeline_config: Merged GlobalPipelineConfig (from context.global_config)
                           NOT the raw PipelineConfig - ensures proper global config inheritance
            orchestrator: Optional orchestrator for component key resolution
            step_state_map: Optional dict mapping step_index to ObjectState for resolving lazy dataclass attributes
        """
        return PathPlanner(context, pipeline_config, orchestrator=orchestrator, step_state_map=step_state_map).plan(pipeline_definition)

    @staticmethod
    def _build_axis_filename(axis_id: str, key: str, extension: str = "pkl", step_index: Optional[int] = None) -> str:
        """Build standardized axis-based filename with optional step index.

        Args:
            axis_id: Well/axis identifier (e.g., "R02C02")
            key: Special output key (e.g., "match_results")
            extension: File extension (default: "pkl")
            step_index: Optional step index to prevent collisions when multiple steps
                       produce the same special output

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


# ===== SCOPE PROMOTION (separate concern) =====

def _apply_scope_promotion_rules(dict_pattern, special_outputs, declared_outputs, step_index, position):
    """Scope promotion for single-key dict patterns - 15 lines."""
    if len(dict_pattern) != 1:
        return special_outputs, declared_outputs

    key_prefix = f"{list(dict_pattern.keys())[0]}_0_"
    promoted_out, promoted_decl = special_outputs.copy(), declared_outputs.copy()

    for out_key in list(special_outputs.keys()):
        if out_key.startswith(key_prefix):
            promoted_key = out_key[len(key_prefix):]
            if promoted_key in promoted_decl:
                raise ValueError(f"Collision: {promoted_key} already exists")
            promoted_out[promoted_key] = special_outputs[out_key]
            promoted_decl[promoted_key] = {
                "step_index": step_index, "position": position,
                "path": special_outputs[out_key]["path"]
            }

    return promoted_out, promoted_decl
