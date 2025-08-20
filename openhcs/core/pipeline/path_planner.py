"""
Pipeline path planning - actually reduced duplication.

This version ACTUALLY eliminates duplication instead of adding abstraction theater.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

from openhcs.constants.constants import READ_BACKEND, WRITE_BACKEND, Backend
from openhcs.constants.input_source import InputSource
from openhcs.core.config import MaterializationBackend
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.pipeline.pipeline_utils import get_core_callable
from openhcs.core.steps.abstract import AbstractStep
from openhcs.core.steps.function_step import FunctionStep

logger = logging.getLogger(__name__)


# ===== PATTERN NORMALIZATION (ONE place) =====

def normalize_pattern(pattern: Any) -> Iterator[Tuple[Callable, str, int]]:
    """THE single pattern normalizer - 15 lines, no duplication."""
    if isinstance(pattern, dict):
        for key, value in pattern.items():
            for pos, func in enumerate(value if isinstance(value, list) else [value]):
                if callable_func := get_core_callable(func):
                    yield (callable_func, key, pos)
    elif isinstance(pattern, list):
        for pos, func in enumerate(pattern):
            if callable_func := get_core_callable(func):
                yield (callable_func, "default", pos)
    elif callable_func := get_core_callable(pattern):
        yield (callable_func, "default", 0)


def extract_attributes(pattern: Any) -> Dict[str, Any]:
    """Extract all function attributes in one pass - 10 lines."""
    outputs, inputs, mat_funcs = set(), {}, {}
    for func, _, _ in normalize_pattern(pattern):
        outputs.update(getattr(func, '__special_outputs__', set()))
        inputs.update(getattr(func, '__special_inputs__', {}))
        mat_funcs.update(getattr(func, '__materialization_functions__', {}))
    return {'outputs': outputs, 'inputs': inputs, 'mat_funcs': mat_funcs}


# ===== PATH PLANNING (NO duplication) =====

class PathPlanner:
    """Minimal path planner with zero duplication."""

    def __init__(self, context: ProcessingContext):
        self.ctx = context
        self.cfg = context.get_path_planning_config()
        self.vfs = context.get_vfs_config()
        self.plans = context.step_plans
        self.declared = {}  # Tracks special outputs

        # Initial input determination (once)
        self.initial_input = Path(context.input_dir)
        self.plate_path = Path(context.plate_path)

    def plan(self, pipeline: List[AbstractStep]) -> Dict:
        """Plan all paths with zero duplication."""
        for i, step in enumerate(pipeline):
            self._plan_step(step, i, pipeline)

        self._validate(pipeline)

        # Set output_plate_root and sub_dir for metadata writing
        if pipeline:
            self.ctx.output_plate_root = self.build_output_plate_root(self.plate_path, self.cfg, is_per_step_materialization=False)
            self.ctx.sub_dir = self.cfg.sub_dir



        return self.plans

    def _plan_step(self, step: AbstractStep, i: int, pipeline: List):
        """Plan one step - no duplicate logic."""
        sid = i  # Use step index instead of step_id

        # Get paths with unified logic
        input_dir = self._get_dir(step, i, pipeline, 'input')
        output_dir = self._get_dir(step, i, pipeline, 'output', input_dir)

        # Extract function data if FunctionStep
        attrs = extract_attributes(step.func) if isinstance(step, FunctionStep) else {
            'outputs': self._normalize_attr(getattr(step, 'special_outputs', set()), set),
            'inputs': self._normalize_attr(getattr(step, 'special_inputs', {}), dict),
            'mat_funcs': {}
        }

        # Process special I/O with unified logic
        special_outputs = self._process_special(attrs['outputs'], attrs['mat_funcs'], 'output', sid)
        special_inputs = self._process_special(attrs['inputs'], attrs['outputs'], 'input', sid)

        # Handle metadata injection
        if isinstance(step, FunctionStep) and any(k in METADATA_RESOLVERS for k in attrs['inputs']):
            step.func = self._inject_metadata(step.func, attrs['inputs'])

        # Generate funcplan (only if needed)
        funcplan = {}
        if isinstance(step, FunctionStep) and special_outputs:
            for func, dk, pos in normalize_pattern(step.func):
                saves = [k for k in special_outputs if k in getattr(func, '__special_outputs__', set())]
                if saves:
                    funcplan[f"{func.__name__}_{dk}_{pos}"] = saves

        # Handle optional materialization and input conversion
        # Read materialization_config directly from step object (not step plans, which aren't populated yet)
        materialized_output_dir = None
        if step.materialization_config:
            # Check if this step has well filters and if current well should be materialized
            step_well_filter = getattr(self.ctx, 'step_well_filters', {}).get(sid)

            if step_well_filter:
                # Inline simple conditional logic for well filtering
                from openhcs.core.config import WellFilterMode
                well_in_filter = self.ctx.well_id in step_well_filter['resolved_wells']
                should_materialize = (
                    well_in_filter if step_well_filter['filter_mode'] == WellFilterMode.INCLUDE
                    else not well_in_filter
                )

                if should_materialize:
                    materialized_output_dir = self._build_output_path(step.materialization_config)
                else:
                    logger.debug(f"Skipping materialization for step {step.name}, well {self.ctx.well_id} (filtered out)")
            else:
                # No well filter - create materialization path as normal
                materialized_output_dir = self._build_output_path(step.materialization_config)

        input_conversion_dir = self._get_optional_path("input_conversion_config", sid)

        # Calculate main pipeline plate root for this step
        main_plate_root = self.build_output_plate_root(self.plate_path, self.cfg, is_per_step_materialization=False)

        # Single update
        self.plans[sid].update({
            'input_dir': str(input_dir),
            'output_dir': str(output_dir),
            'output_plate_root': str(main_plate_root),
            'sub_dir': self.cfg.sub_dir,  # Store resolved sub_dir for main pipeline
            'pipeline_position': i,
            'input_source': self._get_input_source(step, i),
            'special_inputs': special_inputs,
            'special_outputs': special_outputs,
            'funcplan': funcplan,
        })

        # Add optional paths if configured
        if materialized_output_dir:
            # Per-step materialization uses its own config to determine plate root
            materialized_plate_root = self.build_output_plate_root(self.plate_path, step.materialization_config, is_per_step_materialization=False)
            self.plans[sid].update({
                'materialized_output_dir': str(materialized_output_dir),
                'materialized_plate_root': str(materialized_plate_root),
                'materialized_sub_dir': step.materialization_config.sub_dir,  # Store resolved sub_dir for materialization
                'materialized_backend': self.vfs.materialization_backend.value,
                'materialization_config': step.materialization_config  # Store config for well filtering (will be resolved by compiler)
            })
        if input_conversion_dir:
            self.plans[sid].update({
                'input_conversion_dir': str(input_conversion_dir),
                'input_conversion_backend': self.vfs.materialization_backend.value
            })

        # Set backend if needed
        if getattr(step, 'input_source', None) == InputSource.PIPELINE_START:
            self.plans[sid][READ_BACKEND] = self.vfs.materialization_backend.value

            # If zarr conversion occurred, redirect input_dir to zarr store
            if self.vfs.materialization_backend == MaterializationBackend.ZARR and pipeline:
                first_step_plan = self.plans.get(0, {})  # Use step index 0 instead of step_id
                if "input_conversion_dir" in first_step_plan:
                    self.plans[sid]['input_dir'] = first_step_plan['input_conversion_dir']

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
            if i == 0 or getattr(step, 'input_source', None) == InputSource.PIPELINE_START:
                return self.initial_input
            prev_step_index = i - 1  # Use previous step index instead of step_id
            return Path(self.plans[prev_step_index]['output_dir'])
        else:  # output
            if i == 0 or getattr(step, 'input_source', None) == InputSource.PIPELINE_START:
                return self._build_output_path()
            return fallback  # Work in place

    @staticmethod
    def build_output_plate_root(plate_path: Path, path_config, is_per_step_materialization: bool = False) -> Path:
        """Build output plate root directory directly from configuration components.

        Formula:
        - If output_dir_suffix is empty and NOT per-step materialization: use main pipeline output directory
        - If output_dir_suffix is empty and IS per-step materialization: use plate_path directly
        - Otherwise: (global_output_folder OR plate_path.parent) + plate_name + output_dir_suffix

        Args:
            plate_path: Path to the original plate directory
            path_config: PathPlanningConfig with global_output_folder and output_dir_suffix
            is_per_step_materialization: True if this is per-step materialization (no auto suffix)

        Returns:
            Path to plate root directory (e.g., "/data/results/plate001_processed")
        """


        base = Path(path_config.global_output_folder) if path_config.global_output_folder else plate_path.parent

        # Handle empty suffix differently for per-step vs pipeline-level materialization
        if not path_config.output_dir_suffix:
            if is_per_step_materialization:
                # Per-step materialization: use exact path without automatic suffix
                return base / plate_path.name
            else:
                # Pipeline-level materialization: trust lazy inheritance system
                return base / plate_path.name

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

    def _process_special(self, items: Any, extra: Any, io_type: str, sid: str) -> Dict:
        """Unified special I/O processing - no duplication."""
        result = {}

        if io_type == 'output' and items:  # Special outputs
            results_path = self._get_results_path()
            for key in sorted(items):
                filename = PipelinePathPlanner._build_well_filename(self.ctx.well_id, key)
                path = results_path / filename
                result[key] = {
                    'path': str(path),
                    'materialization_function': extra.get(key)  # extra is mat_funcs
                }
                self.declared[key] = str(path)

        elif io_type == 'input' and items:  # Special inputs
            for key in sorted(items.keys() if isinstance(items, dict) else items):
                if key in self.declared:
                    result[key] = {'path': self.declared[key], 'source_step_id': 'prev'}
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

    def _inject_into_pattern(self, pattern: Any, key: str, value: Any) -> Any:
        """Inject value into pattern - handles all cases in 6 lines."""
        if callable(pattern):
            return (pattern, {key: value})
        if isinstance(pattern, tuple) and len(pattern) == 2:
            return (pattern[0], {**pattern[1], key: value})
        if isinstance(pattern, list) and len(pattern) == 1:
            return [self._inject_into_pattern(pattern[0], key, value)]
        raise ValueError(f"Cannot inject into pattern type: {type(pattern)}")

    def _normalize_attr(self, attr: Any, target_type: type) -> Any:
        """Normalize step attributes - 5 lines, no duplication."""
        if target_type == set:
            return {attr} if isinstance(attr, str) else set(attr) if isinstance(attr, (list, set)) else set()
        else:  # dict
            return {attr: True} if isinstance(attr, str) else {k: True for k in attr} if isinstance(attr, list) else attr if isinstance(attr, dict) else {}

    def _get_input_source(self, step: AbstractStep, i: int) -> str:
        """Get input source string."""
        if getattr(step, 'input_source', None) == InputSource.PIPELINE_START:
            return 'PIPELINE_START'
        return 'PREVIOUS_STEP'

    def _get_results_path(self) -> Path:
        """Get results path from global pipeline configuration."""
        try:
            # Access materialization_results_path from global config, not path planning config
            path = self.ctx.global_config.materialization_results_path
            return Path(path) if Path(path).is_absolute() else self.plate_path / path
        except AttributeError as e:
            # Fallback with clear error message if global config is unavailable
            raise RuntimeError(f"Cannot access global config for materialization_results_path: {e}") from e

    def _validate(self, pipeline: List):
        """Validate connectivity and materialization paths - no duplication."""
        # Existing connectivity validation
        for i in range(1, len(pipeline)):
            curr, prev = pipeline[i], pipeline[i-1]
            if getattr(curr, 'input_source', None) == InputSource.PIPELINE_START:
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

        # Collect all materialization steps with their paths and positions
        mat_steps = [
            (step, self.plans.get(i, {}).get('pipeline_position', 0), self._build_output_path(step.materialization_config))
            for i, step in enumerate(pipeline) if step.materialization_config
        ]

        # Group by path for conflict detection
        from collections import defaultdict
        path_groups = defaultdict(list)
        for step, pos, path in mat_steps:
            if path == global_path:
                self._resolve_and_update_paths(step, pos, path, "main flow")
            else:
                path_groups[str(path)].append((step, pos, path))

        # Resolve materialization vs materialization conflicts
        for path_key, step_list in path_groups.items():
            if len(step_list) > 1:
                print(f"⚠️  Materialization path collision detected for {len(step_list)} steps at: {path_key}")
                for step, pos, path in step_list:
                    self._resolve_and_update_paths(step, pos, path, f"pos {pos}")

    def _resolve_and_update_paths(self, step: AbstractStep, position: int, original_path: Path, conflict_type: str) -> None:
        """Resolve path conflict by updating sub_dir configuration directly."""
        # Generate unique sub_dir name instead of calculating from paths
        original_sub_dir = step.materialization_config.sub_dir
        new_sub_dir = f"{original_sub_dir}_step{position}"

        # Update step materialization config with new sub_dir
        config_class = type(step.materialization_config)
        step.materialization_config = config_class(**{**step.materialization_config.__dict__, 'sub_dir': new_sub_dir})

        # Recalculate the resolved path using the new sub_dir
        resolved_path = self._build_output_path(step.materialization_config)

        # Update step plans for metadata generation
        if step_plan := self.plans.get(position):  # Use position (step index) instead of step_id
            if 'materialized_output_dir' in step_plan:
                step_plan['materialized_output_dir'] = str(resolved_path)
                step_plan['materialized_sub_dir'] = new_sub_dir  # Update stored sub_dir

        print(f"    - step '{step.name}' ({conflict_type}) → {resolved_path}")



# ===== PUBLIC API =====

class PipelinePathPlanner:
    """Public API matching original interface."""

    @staticmethod
    def prepare_pipeline_paths(context: ProcessingContext,
                              pipeline_definition: List[AbstractStep]) -> Dict:
        """Prepare pipeline paths."""
        return PathPlanner(context).plan(pipeline_definition)

    @staticmethod
    def _build_well_filename(well_id: str, key: str, extension: str = "pkl") -> str:
        """Build standardized well-based filename."""
        return f"{well_id}_{key}.{extension}"




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