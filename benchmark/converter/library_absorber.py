"""
Library Absorber - One-time absorption of CellProfiler's algorithm library.

Converts the entire CellProfiler library to OpenHCS format once, with:
1. LLM conversion of each function
2. Runtime contract inference
3. Syntax validation
4. Storage in benchmark/cellprofiler_library/

After absorption, .cppipe conversion is instant (no LLM needed).
"""

import ast
import json
import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .source_locator import SourceLocator, SourceLocation
from .llm_converter import LLMFunctionConverter, ConversionResult
from .contract_inference import ContractInference, InferredContract, ContractInferenceResult

logger = logging.getLogger(__name__)


@dataclass
class AbsorbedFunction:
    """A successfully absorbed CellProfiler function."""

    # Identity
    cp_module_name: str  # Original CellProfiler module name
    openhcs_function_name: str  # snake_case function name

    # Contract & Category (LLM-inferred)
    inferred_contract: str  # ProcessingContract: PURE_2D, PURE_3D, FLEXIBLE, VOLUMETRIC_TO_SLICE
    category: str  # Semantic category: image_operation, z_projection, channel_operation
    contract_confidence: float
    contract_notes: str = ""

    # Source paths
    source_file: str = ""  # Where the converted function is stored
    original_cp_file: str = ""  # Original CellProfiler source

    # Status
    validated: bool = False
    validation_errors: List[str] = field(default_factory=list)


@dataclass
class AbsorptionResult:
    """Result of absorbing the CellProfiler library."""
    
    absorbed: List[AbsorbedFunction] = field(default_factory=list)
    failed: List[Tuple[str, str]] = field(default_factory=list)  # (name, error)
    
    @property
    def success_count(self) -> int:
        return len(self.absorbed)
    
    @property
    def failure_count(self) -> int:
        return len(self.failed)
    
    def to_registry(self) -> Dict[str, str]:
        """Generate module name → function name mapping."""
        return {
            f.cp_module_name: f.openhcs_function_name
            for f in self.absorbed
            if f.validated
        }


class LibraryAbsorber:
    """
    One-time absorption of CellProfiler library into OpenHCS.

    Workflow:
    1. Scan benchmark/cellprofiler_source/library/modules/_*.py (pure algorithms)
    2. Scan benchmark/cellprofiler_source/modules/*.py (full classes, for modules not in library)
    3. For each file:
       a. LLM convert to OpenHCS format (extracts algorithm from class cruft)
       b. Validate syntax
       c. (Optional) Run contract inference
       d. Write to benchmark/cellprofiler_library/functions/
    4. Generate registry mapping
    5. Write contracts.json with inferred contracts
    """
    
    def __init__(
        self,
        source_root: Optional[Path] = None,
        output_root: Optional[Path] = None,
        llm_converter: Optional[LLMFunctionConverter] = None,
    ):
        """
        Initialize absorber.
        
        Args:
            source_root: Root of CellProfiler source
            output_root: Where to write absorbed functions
            llm_converter: LLM converter instance
        """
        self.source_root = source_root or Path(__file__).parent.parent / "cellprofiler_source"
        self.output_root = output_root or Path(__file__).parent.parent / "cellprofiler_library"
        self.llm_converter = llm_converter
        
        self.source_locator = SourceLocator(self.source_root)
        self.contract_inference = ContractInference()
    
    def absorb_all(
        self,
        skip_existing: bool = True,
        run_contract_inference: bool = False,  # Expensive - requires working functions
    ) -> AbsorptionResult:
        """
        Absorb entire CellProfiler library.
        
        Args:
            skip_existing: Skip modules already converted
            run_contract_inference: Run runtime contract inference (slow)
            
        Returns:
            AbsorptionResult with all absorption details
        """
        result = AbsorptionResult()

        # Ensure output directories exist
        functions_dir = self.output_root / "functions"
        functions_dir.mkdir(parents=True, exist_ok=True)

        # Collect all modules to absorb: (module_file, module_name, is_library_module)
        modules_to_absorb: List[Tuple[Path, str, bool]] = []
        absorbed_names: set = set()

        # 1. Library modules first (pure algorithms - preferred source)
        library_modules_dir = self.source_root / "library" / "modules"
        if library_modules_dir.exists():
            for module_file in sorted(library_modules_dir.glob("_*.py")):
                if module_file.name == "__init__.py":
                    continue
                module_name = self._file_to_module_name(module_file.name)
                modules_to_absorb.append((module_file, module_name, True))
                absorbed_names.add(module_name.lower())
            logger.info(f"Found {len(modules_to_absorb)} pure library modules")
        else:
            logger.warning(f"Library modules directory not found: {library_modules_dir}")

        # 2. Full module classes (for modules NOT already in library)
        full_modules_dir = self.source_root / "modules"
        if full_modules_dir.exists():
            full_module_count = 0
            for module_file in sorted(full_modules_dir.glob("*.py")):
                if module_file.name.startswith("_") or module_file.name == "__init__.py":
                    continue
                module_name = self._file_to_module_name(module_file.name)
                if module_name.lower() not in absorbed_names:
                    modules_to_absorb.append((module_file, module_name, False))
                    absorbed_names.add(module_name.lower())
                    full_module_count += 1
            logger.info(f"Found {full_module_count} additional full module classes")
        else:
            logger.warning(f"Full modules directory not found: {full_modules_dir}")

        logger.info(f"Total modules to absorb: {len(modules_to_absorb)}")

        for module_file, module_name, is_library in modules_to_absorb:
            func_name = self._module_to_function_name(module_name)
            output_file = functions_dir / f"{func_name}.py"

            # Skip if exists
            if skip_existing and output_file.exists():
                logger.info(f"Skipping {module_name} (already exists)")
                result.absorbed.append(AbsorbedFunction(
                    cp_module_name=module_name,
                    openhcs_function_name=func_name,
                    inferred_contract="unknown",
                    category="image_operation",  # default
                    contract_confidence=0.0,
                    source_file=str(output_file),
                    original_cp_file=str(module_file),
                    validated=True,
                ))
                continue

            # Absorb this module
            source_type = "library" if is_library else "full-class"
            try:
                absorbed = self._absorb_module(
                    module_file=module_file,
                    module_name=module_name,
                    func_name=func_name,
                    output_file=output_file,
                    run_contract_inference=run_contract_inference,
                )
                result.absorbed.append(absorbed)
                logger.info(f"  [{source_type}] {module_name} -> {func_name}")

            except Exception as e:
                logger.error(f"Failed to absorb {module_name} [{source_type}]: {e}")
                result.failed.append((module_name, str(e)))

        # Write registry
        self._write_registry(result)
        
        return result
    
    def _absorb_module(
        self,
        module_file: Path,
        module_name: str,
        func_name: str,
        output_file: Path,
        run_contract_inference: bool,
    ) -> AbsorbedFunction:
        """Absorb a single module."""
        logger.info(f"Absorbing {module_name}...")
        
        # Read source
        source_code = module_file.read_text()
        
        # Check LLM converter
        if self.llm_converter is None:
            raise RuntimeError("LLM converter not initialized")
        
        # Create minimal module block for converter
        from openhcs.interop.cellprofiler.parser import ModuleBlock
        module_block = ModuleBlock(
            name=module_name,
            module_num=0,
            settings={},
        )
        
        source_location = SourceLocation(
            module_name=module_name,
            library_module_path=module_file,
            source_code=source_code,
        )
        
        # LLM convert with retry on validation failure
        max_retries = 2
        conversion = None
        validation_errors = []

        for attempt in range(max_retries + 1):
            if attempt > 0:
                logger.warning(f"  Retry attempt {attempt}/{max_retries} for {module_name}")

            conversion = self.llm_converter.convert(module_block, source_location)

            if not conversion.success:
                if attempt < max_retries:
                    continue
                raise RuntimeError(f"LLM conversion failed: {conversion.error_message}")

            # Validate syntax and contract compliance
            validation_errors = self._validate_syntax(conversion.converted_code)
            if not validation_errors:
                # Success!
                break

            # Log validation errors
            for err in validation_errors:
                logger.error(f"  Validation: {err}")

            # If this was the last attempt, raise
            if attempt >= max_retries:
                raise RuntimeError(f"Validation failed after {max_retries + 1} attempts: {validation_errors}")

        # At this point, conversion succeeded and validation passed
        assert conversion is not None
        assert not validation_errors

        # Inject parameter mapping into docstring
        code_with_mapping = self._inject_parameter_mapping(
            conversion.converted_code,
            conversion.parameter_mapping
        )

        # Write output (only if valid)
        output_file.write_text(code_with_mapping)
        logger.info(f"Wrote {output_file}")

        # Use LLM-inferred contract and category (the LLM read the source and understood it)
        inferred_contract = conversion.inferred_contract.lower()  # normalize to lowercase
        category = conversion.category
        contract_confidence = conversion.confidence
        contract_notes = conversion.reasoning

        logger.info(f"  LLM inference: contract={inferred_contract}, category={category}, confidence={contract_confidence:.2f}")

        # Optional: Runtime contract validation (expensive but validates LLM inference)
        if run_contract_inference and len(validation_errors) == 0:
            contract_result = self._infer_contract_runtime(output_file, func_name)
            if contract_result:
                runtime_contract = contract_result.contract.value
                if runtime_contract != inferred_contract:
                    logger.warning(f"  Runtime inference ({runtime_contract}) differs from LLM ({inferred_contract})")
                    # Trust runtime over LLM if they disagree
                    inferred_contract = runtime_contract
                    contract_confidence = contract_result.confidence
                    contract_notes = f"Runtime override: {contract_result.notes}"

        # Create result
        absorbed = AbsorbedFunction(
            cp_module_name=module_name,
            openhcs_function_name=func_name,
            inferred_contract=inferred_contract,
            category=category,
            contract_confidence=contract_confidence,
            contract_notes=contract_notes,
            source_file=str(output_file),
            original_cp_file=str(module_file),
            validated=len(validation_errors) == 0,
            validation_errors=validation_errors,
        )

        return absorbed

    def _infer_contract_runtime(self, module_file: Path, func_name: str) -> Optional[ContractInferenceResult]:
        """
        Import the converted function and run contract inference with test images.
        """
        import importlib.util

        try:
            # Load module dynamically
            spec = importlib.util.spec_from_file_location(func_name, module_file)
            if spec is None or spec.loader is None:
                logger.warning(f"Could not load {module_file} for contract inference")
                return None

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Get the function
            func = getattr(module, func_name, None)
            if func is None:
                logger.warning(f"Function {func_name} not found in {module_file}")
                return None

            # Run contract inference
            result = self.contract_inference.infer(func)
            return result

        except Exception as e:
            logger.warning(f"Contract inference failed for {func_name}: {e}")
            return None
    
    def _validate_syntax(self, code: str) -> List[str]:
        """Validate Python syntax and OpenHCS contract compliance."""
        errors = []
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
            return errors

        # Check function signatures - only for @numpy decorated functions (main entry points)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Only validate functions with @numpy decorator (the main entry point)
                has_numpy_decorator = any(
                    (isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == 'numpy')
                    or (isinstance(d, ast.Name) and d.id == 'numpy')
                    for d in node.decorator_list
                )
                if not has_numpy_decorator:
                    continue  # Skip helper functions
                if not node.args.args:
                    errors.append(f"{node.name}: no parameters (must have 'image' as first)")
                elif node.args.args[0].arg != 'image':
                    errors.append(f"{node.name}: first param is '{node.args.args[0].arg}', must be 'image'")

        # Check for hallucinated imports
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                # level > 0 means relative import (level=1 is '.', level=2 is '..')
                if node.level > 0:
                    dots = '.' * node.level
                    errors.append(f"Hallucinated relative import: from {dots}{node.module or ''}")
                if node.module and 'functions.' in node.module:
                    errors.append(f"Hallucinated import: from {node.module}")

        return errors
    
    def _write_registry(self, result: AbsorptionResult) -> None:
        """Write registry files."""
        # Write contracts.json with contract, category, confidence
        contracts_file = self.output_root / "contracts.json"
        contracts_data = {
            f.cp_module_name: {
                "function_name": f.openhcs_function_name,
                "contract": f.inferred_contract,
                "category": f.category,
                "confidence": f.contract_confidence,
                "reasoning": f.contract_notes,
                "validated": f.validated,
            }
            for f in result.absorbed
        }
        contracts_file.write_text(json.dumps(contracts_data, indent=2))
        logger.info(f"Wrote {contracts_file}")

        # Write __init__.py with registry
        init_file = self.output_root / "__init__.py"
        init_content = self._generate_init(result)
        init_file.write_text(init_content)
        logger.info(f"Wrote {init_file}")

    def _generate_init(self, result: AbsorptionResult) -> str:
        """Generate __init__.py with registry mapping."""
        lines = [
            '"""',
            'CellProfiler Library - Absorbed into OpenHCS',
            '',
            'Auto-generated by LibraryAbsorber.',
            'Maps CellProfiler module names to OpenHCS functions.',
            '"""',
            '',
            'from typing import Dict, Callable',
            '',
            '# Function imports',
        ]

        # Add imports for validated functions
        for f in result.absorbed:
            if f.validated:
                lines.append(f'from .functions.{f.openhcs_function_name} import {f.openhcs_function_name}')

        lines.extend([
            '',
            '',
            '# Registry mapping CellProfiler module names to OpenHCS functions',
            'CELLPROFILER_MODULES: Dict[str, Callable] = {',
        ])

        for f in result.absorbed:
            if f.validated:
                lines.append(f'    "{f.cp_module_name}": {f.openhcs_function_name},')

        lines.extend([
            '}',
            '',
            '',
            'def get_function(module_name: str) -> Callable:',
            '    """Get OpenHCS function for CellProfiler module name."""',
            '    if module_name not in CELLPROFILER_MODULES:',
            '        raise KeyError(f"Unknown CellProfiler module: {module_name}")',
            '    return CELLPROFILER_MODULES[module_name]',
            '',
            '',
            '__all__ = [',
            '    "CELLPROFILER_MODULES",',
            '    "get_function",',
        ])

        for f in result.absorbed:
            if f.validated:
                lines.append(f'    "{f.openhcs_function_name}",')

        lines.append(']')

        return '\n'.join(lines)

    def _file_to_module_name(self, filename: str) -> str:
        """Convert _threshold.py to Threshold or identifyprimaryobjects.py to IdentifyPrimaryObjects."""
        # Remove .py and leading underscore
        name = filename.replace('.py', '').lstrip('_')
        # Convert to proper CamelCase (capitalize each word after underscore)
        parts = name.split('_')
        return ''.join(word.capitalize() for word in parts)

    def _inject_parameter_mapping(self, code: str, mapping: Dict[str, any]) -> str:
        """
        Inject parameter mapping into the function's docstring.

        Args:
            code: The converted Python code
            mapping: Dict mapping CellProfiler setting names to Python parameter names

        Returns:
            Code with mapping injected into docstring
        """
        if not mapping:
            return code

        lines = code.split('\n')

        # Find the first docstring (should be the function docstring)
        docstring_start = None
        docstring_end = None
        in_docstring = False

        for i, line in enumerate(lines):
            if '"""' in line and not in_docstring:
                docstring_start = i
                in_docstring = True
                # Check if it's a one-liner
                if line.count('"""') == 2:
                    docstring_end = i
                    break
            elif '"""' in line and in_docstring:
                docstring_end = i
                break

        if docstring_start is None or docstring_end is None:
            logger.warning("Could not find docstring to inject parameter mapping")
            return code

        # Build mapping section
        mapping_lines = [
            "",
            "    CellProfiler Parameter Mapping:",
            "    (CellProfiler setting → Python parameter)",
        ]

        for cp_setting, py_param in mapping.items():
            if py_param is None:
                mapping_lines.append(f"        '{cp_setting}' → (no mapping - handled by pipeline)")
            elif isinstance(py_param, list):
                params_str = ', '.join(py_param)
                mapping_lines.append(f"        '{cp_setting}' → [{params_str}]")
            else:
                mapping_lines.append(f"        '{cp_setting}' → {py_param}")

        # Insert before closing docstring
        lines.insert(docstring_end, '\n'.join(mapping_lines))

        return '\n'.join(lines)

    def _module_to_function_name(self, module_name: str) -> str:
        """Convert ModuleName to module_name (snake_case)."""
        # Insert underscore before capitals and lowercase
        return re.sub(r'([A-Z])', r'_\1', module_name).lower().lstrip('_')
