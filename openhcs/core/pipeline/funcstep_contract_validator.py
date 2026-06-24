"""
FuncStep memory contract validator for OpenHCS.

This module provides the FuncStepContractValidator class, which is responsible for
validating memory type declarations for FunctionStep instances in a pipeline.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import logging
import os
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Set, Tuple

from openhcs.constants.constants import (
    AllComponents,
    VALID_MEMORY_TYPES,
    get_openhcs_config,
)
from openhcs.core.artifact_materialization_policy import NO_ARTIFACT_MATERIALIZATION
from openhcs.core.artifacts import ArtifactOutputPlan
from openhcs.core.callable_contract import CallableContract
from openhcs.core.compiled_step_plan import CompiledStepPlan
from openhcs.core.function_patterns import (
    CompiledFunctionPattern,
    FunctionPatternSyntax,
    normalize_function_pattern,
)
from openhcs.core.steps.function_step import FunctionStep

from openhcs.core.components.validation import GenericValidator

# Import ObjectState - it's always available
from objectstate import ObjectState

if TYPE_CHECKING:
    from openhcs.core.context.processing_context import ProcessingContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParameterKindPolicy:
    """Validation policy for an inspect.Parameter kind."""

    kind: inspect._ParameterKind
    required_in_kwargs: bool


@dataclass(frozen=True)
class FunctionStepArtifactContractScope:
    """Compiled execution scope for FunctionStep artifact contract validation."""

    step_name: str
    variable_components: tuple[Enum, ...]
    group_by: Enum | None
    artifact_outputs: Mapping[str, ArtifactOutputPlan]
    compiled_function_pattern: CompiledFunctionPattern
    variable_component_key_counts: Mapping[str, int] | None = None
    source_identity_stack_axes: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.variable_component_key_counts is None:
            return
        missing_axes = self.variable_component_axes() - frozenset(
            self.variable_component_key_counts
        )
        if missing_axes:
            raise ValueError(
                "FunctionStepArtifactContractScope missing component-key "
                f"count(s) for variable axis: {', '.join(sorted(missing_axes))}."
            )

    @classmethod
    def from_step_plan(
        cls,
        step_plan: CompiledStepPlan,
        *,
        group_by: Enum | None = None,
        variable_components: tuple[Enum, ...] | None = None,
        variable_component_key_counts: Mapping[str, int] | None = None,
    ) -> "FunctionStepArtifactContractScope":
        resolved_components = step_plan.variable_components
        if resolved_components is None:
            resolved_components = ()
        resolved_outputs = step_plan.artifact_outputs
        return cls(
            step_name=step_plan.step_name,
            variable_components=(
                tuple(resolved_components)
                if variable_components is None
                else variable_components
            ),
            group_by=step_plan.group_by if group_by is None else group_by,
            artifact_outputs=resolved_outputs,
            compiled_function_pattern=step_plan.compiled_function_pattern,
            variable_component_key_counts=variable_component_key_counts,
            source_identity_stack_axes=step_plan.source_identity_stack_axes,
        )

    def variable_component_axes(self) -> frozenset[str]:
        """Return axes stacked inside one function invocation."""
        return frozenset(
            str(component.value)
            for component in self.variable_components
            if component.value is not None
        )

    def expansion_axes(self) -> frozenset[str]:
        """Return execution axes that can fan out one semantic invocation."""
        axes = set(self.variable_component_axes())
        if self.group_by is not None and self.group_by.value is not None:
            axes.add(str(self.group_by.value))
        return frozenset(axes)

    def multi_plane_variable_axes(self) -> frozenset[str]:
        """Return variable axes that can contribute multiple planes."""
        axes = self.variable_component_axes() - self.source_identity_stack_axes
        if self.variable_component_key_counts is None:
            return axes
        return frozenset(
            axis for axis in axes if self.variable_component_key_counts[axis] > 1
        )

    def artifact_managed_invocation_names(self) -> tuple[str, ...]:
        """Return runtime-adapter invocation names that consume and produce artifacts."""
        names: list[str] = []
        for invocation in self.compiled_function_pattern.iter_invocations():
            if not invocation.runtime_domain.adapter_manages_artifact_inputs:
                continue
            if not any(
                key in self.artifact_outputs
                for key in invocation.artifact_output_keys
            ):
                continue
            names.append(invocation.key.function_name)
        return tuple(names)

    def source_identity_materialized_outputs(
        self,
    ) -> tuple[ArtifactOutputPlan, ...]:
        """Return outputs whose materialized filenames need scalar source identity."""
        return tuple(
            output
            for output in self.artifact_outputs.values()
            if output.materialization is not NO_ARTIFACT_MATERIALIZATION
            and output.kind.materialization_uses_source_identity_filename
        )


@dataclass(frozen=True)
class ArtifactManagedRuntimeScopePolicy:
    """Validation policy for adapter-managed runtime artifact execution axes."""

    forbidden_expansion_axes: frozenset[str]

    def validate(self, scope: FunctionStepArtifactContractScope) -> None:
        invocation_names = scope.artifact_managed_invocation_names()
        if not invocation_names:
            return

        forbidden_axes = (
            self.forbidden_expansion_axes
            & scope.expansion_axes()
            - scope.source_identity_stack_axes
        )
        if not forbidden_axes:
            return

        raise ValueError(
            "Adapter-managed runtime artifact step "
            f"{scope.step_name!r} cannot expand named runtime artifacts by "
            f"{', '.join(sorted(forbidden_axes)).upper()}. "
            "Named CellProfiler artifacts already encode semantic image source "
            "identity; split by SITE/TIMEPOINT or group an explicit source-image "
            "stack instead. "
            f"Artifact-managed invocation(s): {', '.join(invocation_names)}."
        )


_ARTIFACT_MANAGED_RUNTIME_SCOPE_POLICY = ArtifactManagedRuntimeScopePolicy(
    forbidden_expansion_axes=frozenset((AllComponents.CHANNEL.value,)),
)


@dataclass(frozen=True)
class SourceIdentityMaterializationPolicy:
    """Validation policy for source-identity-named artifact materialization."""

    def validate(self, scope: FunctionStepArtifactContractScope) -> None:
        variable_axes = scope.multi_plane_variable_axes()
        if not variable_axes:
            return

        outputs = scope.source_identity_materialized_outputs()
        if not outputs:
            return

        output_labels = ", ".join(
            f"{output.name} ({output.kind.value})" for output in outputs
        )
        axis_labels = ", ".join(sorted(axis.upper() for axis in variable_axes))
        raise ValueError(
            "FunctionStep "
            f"{scope.step_name!r} materializes source-identity-named artifact "
            f"output(s) {output_labels} while processing multi-plane variable "
            f"component(s) {axis_labels}. Runtime artifact materialization "
            "requires one source image identity per output record; split those "
            "component(s) across invocations before materialization."
        )


_SOURCE_IDENTITY_MATERIALIZATION_POLICY = SourceIdentityMaterializationPolicy()


def _parameter_kind_policy_by_kind(
    rows: tuple[ParameterKindPolicy, ...],
) -> Mapping[inspect._ParameterKind, ParameterKindPolicy]:
    by_kind = {row.kind: row for row in rows}
    if set(by_kind) != set(inspect._ParameterKind):
        raise TypeError("Incomplete inspect.Parameter kind policy table.")
    return MappingProxyType(by_kind)


_PARAMETER_KIND_POLICY_BY_KIND = _parameter_kind_policy_by_kind(
    (
        ParameterKindPolicy(inspect.Parameter.POSITIONAL_ONLY, True),
        ParameterKindPolicy(inspect.Parameter.POSITIONAL_OR_KEYWORD, True),
        ParameterKindPolicy(inspect.Parameter.VAR_POSITIONAL, False),
        ParameterKindPolicy(inspect.Parameter.KEYWORD_ONLY, False),
        ParameterKindPolicy(inspect.Parameter.VAR_KEYWORD, False),
    )
)

# ===== DECLARATIVE DEFAULT VALUES =====
# These declarations control defaults and may be moved to configuration in the future

# Simple, direct error messages
def missing_memory_type_error(func_name, step_name):
    return (
        f"Function '{func_name}' in step '{step_name}' needs memory type decorator (@numpy, @cupy, @torch, etc.)\n"
        f"\n"
        f"💡 SOLUTION: Use OpenHCS registry functions instead of raw external library functions:\n"
        f"\n"
        f"❌ WRONG:\n"
        f"   import pyclesperanto as cle\n"
        f"   step = FunctionStep(func=cle.{func_name}, name='{step_name}')\n"
        f"\n"
        f"✅ CORRECT:\n"
        f"   from openhcs.processing.func_registry import get_function_by_name\n"
        f"   {func_name}_func = get_function_by_name('{func_name}', 'pyclesperanto')  # or 'numpy', 'cupy'\n"
        f"   step = FunctionStep(func={func_name}_func, name='{step_name}')\n"
        f"\n"
        f"📋 Available functions: Use get_all_function_names('pyclesperanto') to see all options"
    )

def inconsistent_memory_types_error(step_name, func1, func2):
    return f"Functions in step '{step_name}' have different memory types: {func1} vs {func2}"

def invalid_memory_type_error(func_name, input_type, output_type, valid_types):
    return f"Function '{func_name}' has invalid memory types: {input_type}/{output_type}. Valid: {valid_types}"

def invalid_pattern_error(pattern):
    return f"Invalid function pattern: {pattern}"

def missing_required_args_error(func_name, step_name, missing_args):
    return f"Function '{func_name}' in step '{step_name}' missing required args: {missing_args}"

def complex_pattern_error(step_name):
    return f"Step '{step_name}' with special decorators must use simple function pattern"

def missing_external_library_error(func_name, step_name, module_name, install_command=None):
    error_msg = (
        f"Function '{func_name}' in step '{step_name}' requires external library '{module_name}' which is not installed.\n"
        f"\n"
        f"💡 SOLUTION: Install the required library before compiling the pipeline.\n"
        f"\n"
    )
    if install_command:
        error_msg += f"Install with: {install_command}\n"
    return error_msg


class ImportStatementExtractor(ast.NodeVisitor):
    """
    AST visitor to extract import statements from a function's source code.

    This visitor identifies explicit import statements (import x, from x import y)
    both at the module level and inside functions. It does not analyze attribute
    access patterns, avoiding false positives from local aliases like 'np' instead
    of 'numpy'.
    """

    def __init__(self, module_name: Optional[str] = None):
        """
        Initialize the extractor.

        Args:
            module_name: The name of the module being analyzed (for resolving relative imports)
        """
        self.modules: Set[str] = set()
        self.module_name = module_name
        # Common Python standard library modules to skip
        self.stdlib_modules = {
            'os', 'sys', 're', 'math', 'json', 'collections', 'itertools',
            'functools', 'typing', 'datetime', 'time', 'pathlib', 'io',
            'logging', 'warnings', 'contextlib', 'copy', 'pickle', 'random',
            'string', 'enum', 'dataclasses', 'inspect', 'ast', 'importlib',
            'types', 'numbers', 'abc', 'threading', 'multiprocessing',
            'concurrent', 'queue', 'subprocess', 'shutil', 'tempfile',
            'glob', 'fnmatch', 'hashlib', 'base64', 'uuid', 'decimal',
            'fractions', 'statistics', 'secrets', 'textwrap', 'unicodedata',
            'codecs', 'csv', 'configparser', 'xml', 'html', 'urllib',
            'http', 'email', 'mimetypes', 'socket', 'ssl', 'hashlib',
            'hmac', 'secrets', 'zipfile', 'tarfile', 'gzip', 'bz2', 'lzma',
            'sqlite3', 'decimal', 'fractions', 'statistics', 'typing',
            'typing_extensions', 'builtins', '__future__', 'warnings',
        }

    def visit_Import(self, node: ast.Import) -> None:
        """Visit import statements."""
        for alias in node.names:
            module_name = alias.name.split('.')[0]
            self._add_module_if_external(module_name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definitions to extract inline imports."""
        self.visit_function_scope(node)

    def visit_function_scope(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        """Visit any function-like scope while preserving inline import discovery."""
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Visit from-import statements (AST uses node.level for relative imports)."""
        level = node.level or 0

        if level > 0:
            # Relative import: use node.level to determine how many levels to go up
            absolute_module = self._resolve_relative_import(node.module, level)
            if absolute_module:
                # Only consider the true top-level package
                self._add_module_if_external(absolute_module.split(".")[0])
        elif node.module:
            # Absolute import: no level means it's an absolute import
            self._add_module_if_external(node.module.split(".")[0])

        self.generic_visit(node)

    def _resolve_relative_import(self, module: Optional[str], level: Optional[int] = None) -> Optional[str]:
        """
        Resolve an ImportFrom-relative import (module + level) to an absolute module name.

        This method supports two calling conventions for backward compatibility:
        1. New interface: _resolve_relative_import(module, level) - AST-based
        2. Old interface: _resolve_relative_import(relative_module) - string-based

        Args:
            module: The ImportFrom module (e.g., 'percentile_utils' for `from .percentile_utils import ...`)
                    OR the relative module string (e.g., '.percentile_utils') for old interface
            level: The ImportFrom level (1='.', 2='..', ...) for new interface, or None for old interface

        Returns:
            Absolute module name if resolution succeeds, None otherwise
        """
        if self.module_name is None:
            return None

        # Handle old interface (string-based) for backward compatibility
        if level is None:
            # Old interface: module is the relative module string (e.g., '.percentile_utils')
            relative_module = module
            if relative_module is None:
                return None

            # Count the number of dots in the relative import
            # e.g., '.' -> 1 (current package), '..' -> 2 (parent package), '...' -> 3 (grandparent package)
            level = 0
            for char in relative_module:
                if char == '.':
                    level += 1
                else:
                    break

            # Get the package part of the relative import (after the dots)
            # e.g., '.percentile_utils' -> 'percentile_utils'
            # e.g., '..utils' -> 'utils'
            package_part = relative_module[level:]
        else:
            # New interface: module is the module name (without dots), level is provided separately
            package_part = module

        # Split the current module name into parts
        # e.g., 'openhcs.processing.backends.processors.numpy_processor'
        # -> ['openhcs', 'processing', 'backends', 'processors', 'numpy_processor']
        module_parts = self.module_name.split('.')

        # Remove the last part (the module name itself)
        # e.g., ['openhcs', 'processing', 'backends', 'processors', 'numpy_processor']
        # -> ['openhcs', 'processing', 'backends', 'processors']
        module_parts = module_parts[:-1]

        # Go up the specified number of levels
        # In Python relative imports:
        # - '.' (level=1) means current package (same directory) -> don't go up
        # - '..' (level=2) means parent package -> go up 1 level
        # - '...' (level=3) means grandparent package -> go up 2 levels
        # So we need to go up (level - 1) levels
        levels_to_go_up = max(level - 1, 0)

        if levels_to_go_up >= len(module_parts):
            return None

        module_parts = module_parts[:-levels_to_go_up] if levels_to_go_up > 0 else module_parts

        # Add the module path parts (may be nested like "utils.foo")
        if package_part:
            package_parts = package_part.split(".")
            module_parts.extend(package_parts)

        # Join to get the absolute module name
        absolute_module = '.'.join(module_parts)
        return absolute_module

    def _add_module_if_external(self, module_name: str) -> None:
        """
        Add a module if it's external (not stdlib or openhcs).

        Args:
            module_name: The module name to check
        """
        # Skip openhcs internal modules
        if module_name == 'openhcs':
            return

        # Skip standard library modules
        if module_name in self.stdlib_modules:
            return

        # Skip built-in modules
        if module_name in ('builtins', '__builtins__'):
            return

        # Add the module
        self.modules.add(module_name)


def extract_import_statements(func: Callable) -> Set[str]:
    """
    Extract explicit import statements from a function's module's source code.

    This function parses the entire module's source code using AST and identifies
    only explicit import statements (import x, from x import y), avoiding
    false positives from attribute access patterns.

    Args:
        func: The function to analyze for import statements

    Returns:
        Set of top-level module names that are explicitly imported
    """
    # Get the module name from the function
    module_name = func.__module__
    if module_name is None:
        return set()

    try:
        module = importlib.import_module(module_name)
        module_file = module.__file__
        if module_file is None:
            return set()
        stat = os.stat(module_file)
    except Exception:
        # Can't get source code
        return set()

    return set(
        _extract_import_statements_from_module_file(
            module_name,
            module_file,
            stat.st_mtime_ns,
            stat.st_size,
        )
    )


@lru_cache(maxsize=512)
def _extract_import_statements_from_module_file(
    module_name: str,
    module_file: str,
    mtime_ns: int,
    size_bytes: int,
) -> frozenset[str]:
    del mtime_ns, size_bytes
    try:
        with open(module_file, 'r', encoding='utf-8') as f:
            source = f.read()
    except Exception:
        return frozenset()

    try:
        # Parse the source code into an AST
        tree = ast.parse(source)
    except SyntaxError:
        # Can't parse the source
        return frozenset()

    # Extract import statements using the AST visitor
    extractor = ImportStatementExtractor(module_name)
    extractor.visit(tree)

    return frozenset(extractor.modules)

class FuncStepContractValidator:
    """
    Validator for FunctionStep memory type contracts.

    Enforces that all FunctionStep instances require explicit memory type declarations
    and named positional arguments for all functions.

    Key principles:
    1. All functions in a FunctionStep must have consistent memory types
    2. The shared memory types are set as the step's memory types in the step plan
    3. Memory types must be validated at plan time, not runtime
    4. No fallback or inference of memory types is allowed
    5. All function patterns (callable, tuple, list, dict) are supported
    6. When using (func, kwargs) pattern, all required positional arguments must be
        explicitly provided in the kwargs dict
    """

    @staticmethod
    def validate_external_library_installation(func: Callable, step_name: str) -> None:
        """
        Validate that external libraries required by a function are installed.

        This function uses a combined approach:
        1. For openhcs modules: Parse the module's source code to find all import statements
        2. For external modules: Import the module directly to verify it's installed

        This approach is more reliable than AST-based analysis because:
        1. It actually tests if dependencies work
        2. No false positives from local aliases (e.g., np instead of numpy)
        3. Works for any importable function
        4. The import error message will identify the missing dependency
        5. Catches dependencies in helper functions called by the main function
        6. Doesn't incorrectly flag openhcs internal modules as external

        Note: For openhcs modules, we parse the source code to find all import statements
        (including those in helper functions) and try to import each one. For external
        modules, we import the module directly.

        Args:
            func: The function to check for external library dependencies
            step_name: The name of the step containing the function

        Raises:
            ValueError: If the external library required by the function is not installed
        """
        # Get the module name from the function
        module_name = func.__module__
        if module_name is None:
            # No module info, skip validation (e.g., built-in or dynamically created)
            return

        # Extract the top-level package name
        # e.g., "openhcs.processing.backends.analysis.skan_axon_analysis" -> "openhcs"
        # e.g., "skimage.measure" -> "skimage"
        # e.g., "skan" -> "skan"
        top_level_package = module_name.split('.')[0]

        # For openhcs modules, parse source code for import statements
        if top_level_package == 'openhcs':
            # Extract import statements from the module's source code
            import_statements = extract_import_statements(func)

            # Try to import each module to verify it's installed
            for module_name_to_import in import_statements:
                try:
                    importlib.import_module(module_name_to_import)
                except ImportError as e:
                    # Parse the error message to extract the missing module name
                    error_str = str(e)
                    missing_module = module_name_to_import

                    # Try to extract the missing module from the error message
                    if "No module named" in error_str:
                        import re
                        match = re.search(r"No module named '([^']+)'", error_str)
                        if match:
                            missing_module = match.group(1)

                    # Generate a generic install command for the module
                    install_command = f'pip install {missing_module}'

                    raise ValueError(missing_external_library_error(
                        func.__name__, step_name, missing_module, install_command
                    )) from e
        else:
            # For external modules, try to import the module directly
            try:
                importlib.import_module(module_name)
            except ImportError as e:
                # Parse the error message to extract the missing module name
                # ImportError messages typically look like:
                # "No module named 'numpy'"
                # "cannot import name 'something' from 'module'"
                error_str = str(e)
                missing_module = top_level_package  # Default to the top-level package

                # Try to extract the missing module from the error message
                if "No module named" in error_str:
                    # Extract module name from quotes
                    import re
                    match = re.search(r"No module named '([^']+)'", error_str)
                    if match:
                        missing_module = match.group(1)
                elif "cannot import name" in error_str or "cannot import" in error_str:
                    # Use the top-level package as fallback
                    missing_module = top_level_package

                # Generate a generic install command for the module
                install_command = f'pip install {missing_module}'

                raise ValueError(missing_external_library_error(
                    func.__name__, step_name, missing_module, install_command
                )) from e

    @staticmethod
    def normalized_group_by(
        group_by,
        variable_components,
        step_name: str,
    ):
        """Return compiled grouping semantics after conflict normalization."""
        if group_by and group_by.value in [vc.value for vc in variable_components]:
            from openhcs.constants import GroupBy

            logger.warning(
                f"Step '{step_name}': Auto-resolved group_by conflict. "
                f"Set group_by to GroupBy.NONE due to conflict with "
                f"variable_components {[vc.value for vc in variable_components]}. "
                f"Original group_by was {group_by.value}."
            )
            return GroupBy.NONE
        return group_by

    @staticmethod
    def validate_pipeline(
        steps: List[Any],
        pipeline_context: ProcessingContext | None = None,
        step_state_map: Optional[Dict[int, ObjectState]] = None,
        orchestrator=None,
    ) -> None:
        """
        Validate memory type contracts and function patterns for all FunctionStep instances in a pipeline.

        This validator must run after materialization and path planners to ensure
        proper plan integration. It verifies that these planners have run by checking
        pipeline_context for planner execution flags and by validating presence
        of required fields in step plans.

        Args:
            steps: The steps in the pipeline
            pipeline_context: Optional context object with planner execution flags
            step_state_map: Map of step index to ObjectState for accessing config values
            orchestrator: Optional orchestrator for dict pattern key validation

        Raises:
            ValueError: If any FunctionStep violates memory type contracts or dict pattern validation
            AssertionError: If required planners have not run before this validator
        """
        # Validate steps
        if not steps:
            logger.warning("No steps provided to FuncStepContractValidator")
            return

        if pipeline_context is None:
            raise ValueError(
                "FuncStepContractValidator requires a compiled ProcessingContext. "
                "Validate raw patterns with validate_function_pattern(...) before "
                "compiler planning, or validate compiled step plans here."
            )

        if not pipeline_context.step_plans:
            raise AssertionError(
                "Clause 101 Violation: Step plans must be initialized before "
                "FuncStepContractValidator."
            )

        sample_step_index = next(iter(pipeline_context.step_plans.keys()))
        sample_plan = pipeline_context.step_plans[sample_step_index]
        if sample_plan.read_backend is None or sample_plan.write_backend is None:
            raise AssertionError(
                "Clause 101 Violation: Materialization planner must run before "
                "FuncStepContractValidator. Step plans missing "
                "read_backend/write_backend fields."
            )

        # Process each step in the pipeline
        for i, step in enumerate(steps):
            # Only validate FunctionStep instances
            if isinstance(step, FunctionStep):
                if i not in pipeline_context.step_plans:
                    raise AssertionError(
                        f"Clause 101 Violation: Step {step.name} (index: {i}) missing from step_plans."
                    )
                step_plan = pipeline_context.step_plans[i]
                if step_plan.compiled_function_pattern is None:
                    raise AssertionError(
                        f"Clause 101 Violation: Step {step.name} (index: {i}) missing compiled_function_pattern."
                    )
                FuncStepContractValidator.validate_compiled_step_plan(
                    step_plan,
                    orchestrator,
                )
                input_type, output_type = (
                    FuncStepContractValidator.validate_compiled_function_pattern(
                        step_plan.compiled_function_pattern,
                        step_plan.step_name,
                    )
                )
                step_plan.input_memory_type = input_type
                step_plan.output_memory_type = output_type

    @staticmethod
    def validate_compiled_step_plan(step_plan, orchestrator=None) -> None:
        """Validate FunctionStep structure from the compiled plan SSOT."""
        func_pattern = step_plan.func
        step_name = step_plan.step_name
        FuncStepContractValidator._contracts_from_pattern(
            func_pattern,
            step_name,
        )

        config = get_openhcs_config()
        validator = GenericValidator(config)
        group_by = step_plan.group_by
        if step_plan.variable_components is None:
            variable_components = ()
        else:
            variable_components = step_plan.variable_components

        group_by = FuncStepContractValidator.normalized_group_by(
            group_by,
            variable_components,
            step_name,
        )

        validation_result = validator.validate_step(
            variable_components,
            group_by,
            func_pattern,
            step_name,
        )
        if not validation_result.is_valid:
            raise ValueError(validation_result.error_message)

        if orchestrator is not None and isinstance(func_pattern, dict) and group_by is not None:
            dict_validation_result = validator.validate_dict_pattern_keys(
                func_pattern,
                group_by,
                step_name,
                orchestrator,
            )
            if not dict_validation_result.is_valid:
                raise ValueError(dict_validation_result.error_message)

        artifact_scope = FunctionStepArtifactContractScope.from_step_plan(
            step_plan,
            group_by=group_by,
            variable_components=tuple(variable_components),
            variable_component_key_counts=(
                FuncStepContractValidator.variable_component_key_counts(
                    orchestrator,
                    tuple(variable_components),
                )
            ),
        )
        FuncStepContractValidator.validate_artifact_contract_scope(artifact_scope)

    @staticmethod
    def validate_artifact_contract_scope(
        scope: FunctionStepArtifactContractScope,
    ) -> None:
        """Validate every artifact contract policy for one compiled step scope."""
        _ARTIFACT_MANAGED_RUNTIME_SCOPE_POLICY.validate(scope)
        _SOURCE_IDENTITY_MATERIALIZATION_POLICY.validate(scope)

    @staticmethod
    def variable_component_key_counts(
        orchestrator,
        variable_components: tuple[Enum, ...],
    ) -> Mapping[str, int] | None:
        """Return concrete component-key cardinality when compile owns it."""
        if orchestrator is None:
            return None
        counts: dict[str, int] = {}
        for component in variable_components:
            if component.value is None:
                continue
            axis = str(component.value)
            counts[axis] = len(tuple(orchestrator.get_component_keys(component)))
        return MappingProxyType(counts)

    @staticmethod
    def validate_artifact_managed_runtime_scope(
        step_plan,
        *,
        group_by: Enum | None = None,
        variable_components: tuple[Enum, ...] | None = None,
    ) -> None:
        """Reject execution axes that duplicate named runtime artifact semantics."""
        scope = FunctionStepArtifactContractScope.from_step_plan(
            step_plan,
            group_by=group_by,
            variable_components=variable_components,
        )
        _ARTIFACT_MANAGED_RUNTIME_SCOPE_POLICY.validate(scope)

    @staticmethod
    def validate_source_identity_materialization_scope(
        step_plan,
        *,
        group_by: Enum | None = None,
        variable_components: tuple[Enum, ...] | None = None,
    ) -> None:
        """Reject multi-plane invocation shapes for scalar-source materialization."""
        scope = FunctionStepArtifactContractScope.from_step_plan(
            step_plan,
            group_by=group_by,
            variable_components=variable_components,
        )
        _SOURCE_IDENTITY_MATERIALIZATION_POLICY.validate(scope)

    @staticmethod
    def validate_funcstep(
        step: FunctionStep,
        orchestrator=None,
        step_objectstate: Optional[ObjectState] = None,
    ) -> None:
        """
        Validate memory type contracts, func_pattern structure, and dict pattern keys for a FunctionStep instance.

        Args:
            step: The FunctionStep to validate
            orchestrator: Optional orchestrator for dict pattern key validation
            step_objectstate: ObjectState for accessing config values

        Raises:
            ValueError: If FunctionStep violates memory type contracts, structural rules,
                        or dict pattern key validation.
        """
        # Extracting config values via ObjectState get_saved_resolved_value()
        if step_objectstate is None:
            raise ValueError(f"Step '{step.name}': ObjectState is required for config access")

        variable_components = step_objectstate.get_saved_resolved_value('processing_config.variable_components')
        group_by = step_objectstate.get_saved_resolved_value('processing_config.group_by')

        # Extracting function pattern and name from step
        func_pattern = step.func
        step_name = step.name

        # Validate pattern structure before generic config validation.
        FuncStepContractValidator._contracts_from_pattern(func_pattern, step_name)

        # Validate using generic validation system
        config = get_openhcs_config()
        validator = GenericValidator(config)

        # Check for constraint violation: group_by ∈ variable_components
        group_by = FuncStepContractValidator.normalized_group_by(
            group_by,
            variable_components,
            step_name,
        )

        # Sequential processing validation removed - it's now pipeline-level, not per-step

        # Validate step configuration after auto-resolution
        validation_result = validator.validate_step(
            variable_components, group_by, func_pattern, step_name
        )
        if not validation_result.is_valid:
            raise ValueError(validation_result.error_message)

        # Validate dict pattern keys if orchestrator is available
        if orchestrator is not None and isinstance(func_pattern, dict) and group_by is not None:
            dict_validation_result = validator.validate_dict_pattern_keys(
                func_pattern, group_by, step_name, orchestrator
            )
            if not dict_validation_result.is_valid:
                raise ValueError(dict_validation_result.error_message)

    @staticmethod
    def validate_compiled_function_pattern(
        compiled_pattern,
        step_name: str,
    ) -> Tuple[str, str]:
        """Validate memory contracts from the compiled function-pattern graph."""
        invocations = tuple(compiled_pattern.iter_invocations())
        if not invocations:
            raise ValueError(f"No valid functions found in compiled pattern for step {step_name}")

        first = invocations[0]
        input_type, output_type = (
            FuncStepContractValidator._validate_invocation_contract(
                first,
                step_name,
            )
        )

        for invocation in invocations[1:]:
            FuncStepContractValidator._validate_invocation_contract(
                invocation,
                step_name,
            )

        return input_type, invocations[-1].output_memory_type

    @staticmethod
    def _validate_invocation_contract(invocation, step_name: str) -> Tuple[str, str]:
        """Validate one compiled invocation's callable contract."""
        contract = invocation.contract
        FuncStepContractValidator.validate_external_library_installation(
            contract.func,
            step_name,
        )

        input_type = contract.input_memory_type
        output_type = contract.output_memory_type
        if input_type is None or output_type is None:
            raise ValueError(
                missing_memory_type_error(contract.function_name, step_name)
            )
        if input_type not in VALID_MEMORY_TYPES or output_type not in VALID_MEMORY_TYPES:
            raise ValueError(
                invalid_memory_type_error(
                    (
                        f"{contract.function_name}"
                        f"[{invocation.key.group_key}:{invocation.key.position}]"
                    ),
                    input_type,
                    output_type,
                    ", ".join(sorted(VALID_MEMORY_TYPES)),
                )
            )
        return input_type, output_type

    @staticmethod
    def validate_function_pattern(
        func: Any,
        step_name: str
    ) -> Tuple[str, str]:
        """
        Validate memory type contracts for a function pattern.

        Args:
            func: The function pattern to validate
            step_name: The name of the step containing the function

        Returns:
            Tuple of (input_memory_type, output_memory_type)

        Raises:
            ValueError: If the function pattern violates memory type contracts
        """
        contracts = FuncStepContractValidator._contracts_from_pattern(
            func,
            step_name,
        )
        first_contract = contracts[0]
        first_fn = first_contract.func

        # Validate that external libraries are installed (compile-time check)
        # This catches missing dependencies like 'skan' before execution
        FuncStepContractValidator.validate_external_library_installation(first_fn, step_name)

        # Validate that the function has explicit memory type declarations
        input_type = first_contract.input_memory_type
        output_type = first_contract.output_memory_type
        if input_type is None or output_type is None:
            raise ValueError(
                missing_memory_type_error(first_contract.function_name, step_name)
            )

        # Validate memory types against known valid types
        if input_type not in VALID_MEMORY_TYPES or output_type not in VALID_MEMORY_TYPES:
            raise ValueError(invalid_memory_type_error(
                first_contract.function_name, input_type, output_type, ", ".join(sorted(VALID_MEMORY_TYPES))
            ))

        # Validate that all functions have valid memory type declarations
        for contract in contracts[1:]:
            fn_input_type = contract.input_memory_type
            fn_output_type = contract.output_memory_type
            if fn_input_type is None or fn_output_type is None:
                raise ValueError(
                    missing_memory_type_error(contract.function_name, step_name)
                )

            # Validate memory types against known valid types
            if fn_input_type not in VALID_MEMORY_TYPES or fn_output_type not in VALID_MEMORY_TYPES:
                raise ValueError(invalid_memory_type_error(
                    contract.function_name, fn_input_type, fn_output_type, ", ".join(sorted(VALID_MEMORY_TYPES))
                ))

        # Return first function's input type and last function's output type
        return input_type, contracts[-1].output_memory_type

    @staticmethod
    def _validate_required_args(func: Callable, kwargs: Dict[str, Any], step_name: str) -> None:
        """
        Validate that all required positional arguments are provided in kwargs.

        All required positional arguments must be explicitly provided in the kwargs dict
        when using the (func, kwargs) pattern.

        Args:
            func: The function to validate
            kwargs: The kwargs dict to check
            step_name: The name of the step containing the function

        Raises:
            ValueError: If any required positional arguments are missing from kwargs
        """
        # Get the function signature
        sig = inspect.signature(func)

        # Collect names of required positional arguments
        required_args = []
        for name, param in sig.parameters.items():
            policy = _PARAMETER_KIND_POLICY_BY_KIND[param.kind]
            if policy.required_in_kwargs:
                # Check if parameter has no default value
                if param.default is inspect.Parameter.empty:
                    required_args.append(name)

        # Check if all required args are in kwargs
        missing_args = [arg for arg in required_args if arg not in kwargs]

        # Raise error if any required args are missing
        if missing_args:
            raise ValueError(missing_required_args_error(func.__name__, step_name, missing_args))

    @staticmethod
    def _validate_dict_pattern_keys(
        func_pattern: dict,
        group_by,
        step_name: str,
        orchestrator
    ) -> None:
        """
        Validate that dict function pattern keys match available component keys.

        This validation ensures compile-time guarantee that dict patterns will work
        at runtime by checking that all dict keys exist in the actual component data.

        Args:
            func_pattern: Dict function pattern to validate
            group_by: GroupBy enum specifying component type
            step_name: Name of the step containing the function
            orchestrator: Orchestrator for component key access

        Raises:
            ValueError: If dict pattern keys don't match available component keys
        """
        # Get available component keys from orchestrator
        try:
            available_keys = orchestrator.get_component_keys(group_by)
            available_keys_set = set(str(key) for key in available_keys)
        except Exception as e:
            raise ValueError(f"Failed to get component keys for {group_by.value}: {e}")

        # Check each dict key against available keys
        pattern_keys = list(func_pattern.keys())
        pattern_keys_set = set(str(key) for key in pattern_keys)

        # Try direct string match first
        missing_keys = pattern_keys_set - available_keys_set

        if missing_keys:
            # Try integer conversion for missing keys
            still_missing = set()
            for key in missing_keys:
                try:
                    # Try converting pattern key to int and check if int version exists
                    key_as_int = int(key)
                    if str(key_as_int) in available_keys_set:
                        continue  # Key exists as integer, not missing
                except (ValueError, TypeError):
                    still_missing.add(key)

            if still_missing:
                raise ValueError(
                    f"Function pattern keys not found in available {group_by.value} components for step '{step_name}'. "
                    f"Missing keys: {sorted(still_missing)}. "
                    f"Available keys: {sorted(available_keys)}. "
                    f"Function pattern keys must match component values from the plate data."
                )

    @staticmethod
    def validate_pattern_structure(
        func: FunctionPatternSyntax,
        step_name: str
    ) -> List[Callable]:
        """
        Validate and extract all functions from a function pattern.

        This wraps the function-pattern normalizer so validation shares the same
        traversal and disabled-item semantics as compiler/runtime planning.

        Supports nested patterns of arbitrary depth, including:
        - Direct callable
        - Tuple of (callable/FunctionReference, kwargs)
        - List of callables or patterns
        - Dict of keyed callables or patterns

        Args:
            func: The function pattern to validate and extract functions from
            step_name: The name of the step or component containing the function

        Returns:
            List of functions in the pattern

        Raises:
            ValueError: If the function pattern is invalid
        """
        contracts = FuncStepContractValidator._contracts_from_pattern(
            func,
            step_name,
        )
        return [contract.func for contract in contracts]

    @staticmethod
    def _contracts_from_pattern(
        func: FunctionPatternSyntax,
        step_name: str,
    ) -> list[CallableContract]:
        """Return callable contracts from the function-pattern authority."""
        try:
            normalized = normalize_function_pattern(func)
        except (TypeError, ValueError) as exc:
            raise ValueError(invalid_pattern_error(func)) from exc
        contracts = [item.contract for item in normalized.iter_items()]
        if not contracts:
            raise ValueError(f"No valid functions found in pattern for step {step_name}")
        return contracts
