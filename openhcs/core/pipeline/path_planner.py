"""
Pipeline path planning - actually reduced duplication.

This version ACTUALLY eliminates duplication instead of adding abstraction theater.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Set, Tuple

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
        self.initial_input = Path(context.zarr_conversion_path or context.input_dir)
        self.plate_path = Path(context.plate_path)
    
    def plan(self, pipeline: List[AbstractStep]) -> Dict:
        """Plan all paths with zero duplication."""
        for i, step in enumerate(pipeline):
            self._plan_step(step, i, pipeline)
        
        self._validate(pipeline)
        self._apply_overrides(pipeline)
        return self.plans
    
    def _plan_step(self, step: AbstractStep, i: int, pipeline: List):
        """Plan one step - no duplicate logic."""
        sid = step.step_id
        
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

        # Handle per-step materialization
        materialized_output_dir = None
        if "materialization_config" in self.plans[sid]:
            materialization_config = self.plans[sid]["materialization_config"]
            materialized_output_dir = self._calculate_materialized_output_path(materialization_config)

        # Single update
        self.plans[sid].update({
            'input_dir': str(input_dir),
            'output_dir': str(output_dir),
            'pipeline_position': i,
            'input_source': self._get_input_source(step, i),
            'special_inputs': special_inputs,
            'special_outputs': special_outputs,
            'funcplan': funcplan,
        })

        # Add materialized output if configured
        if materialized_output_dir:
            self.plans[sid]['materialized_output_dir'] = str(materialized_output_dir)
            self.plans[sid]['materialized_backend'] = self.vfs.materialization_backend.value
        
        # Set backend if needed
        if getattr(step, 'input_source', None) == InputSource.PIPELINE_START:
            self.plans[sid][READ_BACKEND] = self.vfs.materialization_backend.value
    
    def _get_dir(self, step: AbstractStep, i: int, pipeline: List, 
                 dir_type: str, fallback: Path = None) -> Path:
        """Unified directory resolution - no duplication."""
        sid = step.step_id
        
        # Check overrides (same for input/output)
        if override := self.plans.get(sid, {}).get(f'{dir_type}_dir'):
            return Path(override)
        if override := getattr(step, f'{dir_type}_dir', None):
            return Path(override)
        
        # Type-specific logic
        if dir_type == 'input':
            if i == 0 or getattr(step, 'input_source', None) == InputSource.PIPELINE_START:
                return self.initial_input
            prev_sid = pipeline[i-1].step_id
            return Path(self.plans[prev_sid]['output_dir'])
        else:  # output
            if i == 0 or getattr(step, 'input_source', None) == InputSource.PIPELINE_START:
                return self._build_output_path()
            return fallback  # Work in place
    
    def _build_output_path(self, path_config=None) -> Path:
        """Build output path - 8 lines, no duplication."""
        config = path_config or self.cfg
        name = f"{self.plate_path.name}{config.output_dir_suffix}"
        path = Path(name)
        if config.sub_dir:
            path = path / config.sub_dir
        if self.vfs.materialization_backend == MaterializationBackend.ZARR:
            path = path.with_suffix('.zarr')
        base = Path(config.global_output_folder) if config.global_output_folder else self.plate_path.parent
        return base / path

    def _calculate_materialized_output_path(self, materialization_config) -> Path:
        """Calculate materialized output path using custom PathPlanningConfig."""
        return self._build_output_path(materialization_config)
    
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
        """Get results path - 3 lines."""
        path = self.cfg.materialization_results_path
        return Path(path) if Path(path).is_absolute() else self.plate_path / path
    
    def _validate(self, pipeline: List):
        """Validate connectivity - 10 lines, no duplication."""
        for i in range(1, len(pipeline)):
            curr, prev = pipeline[i], pipeline[i-1]
            if getattr(curr, 'input_source', None) == InputSource.PIPELINE_START:
                continue
            curr_in = self.plans[curr.step_id]['input_dir']
            prev_out = self.plans[prev.step_id]['output_dir']
            if curr_in != prev_out:
                has_special = any(inp.get('source_step_id') == prev.step_id 
                                for inp in self.plans[curr.step_id].get('special_inputs', {}).values())
                if not has_special:
                    raise ValueError(f"Disconnect: {prev.name} -> {curr.name}")
    
    def _apply_overrides(self, pipeline: List):
        """Apply final overrides - 8 lines."""
        if self.ctx.zarr_conversion_path and pipeline:
            first = pipeline[0]
            self.plans[first.step_id]['input_dir'] = self.ctx.original_input_dir
            self.plans[first.step_id]['convert_to_zarr'] = str(
                Path(self.ctx.zarr_conversion_path) / f"{self.cfg.sub_dir}.zarr"
            )
        if pipeline:
            first_out = Path(self.plans[pipeline[0].step_id]['output_dir'])
            self.ctx.output_plate_root = first_out.parent if self.cfg.sub_dir and first_out.name in (self.cfg.sub_dir, f"{self.cfg.sub_dir}.zarr") else first_out


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

    @staticmethod
    def resolve_output_plate_root(step_output_dir: Path, path_config) -> Path:
        """Resolve output plate root directory from step output directory."""
        step_output_path = Path(step_output_dir)
        if not path_config.sub_dir:
            return step_output_path
        # Remove sub_dir component: if path ends with sub_dir(.zarr), return parent
        if step_output_path.name in (path_config.sub_dir, f"{path_config.sub_dir}.zarr"):
            return step_output_path.parent
        return step_output_path


# ===== METADATA =====

METADATA_RESOLVERS = {
    "grid_dimensions": {
        "resolver": lambda context: context.microscope_handler.get_grid_dimensions(context.input_dir),
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

def _apply_scope_promotion_rules(dict_pattern, special_outputs, declared_outputs, step_id, position):
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
                "step_id": step_id, "position": position, 
                "path": special_outputs[out_key]["path"]
            }
    
    return promoted_out, promoted_decl