"""
Library Absorber - One-time absorption of CellProfiler's algorithm library.

Converts the entire CellProfiler library to OpenHCS format once, with:
1. LLM conversion of each function
2. Syntax validation
3. Storage in benchmark/cellprofiler_library/

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

logger = logging.getLogger(__name__)


@dataclass
class AbsorbedFunction:
    """A successfully absorbed CellProfiler function."""

    cp_module_name: str
    openhcs_function_name: str
    confidence: float = 0.0
    notes: str = ""
    source_file: str = ""
    original_cp_file: str = ""
    validated: bool = False
    validation_errors: List[str] = field(default_factory=list)


@dataclass
class AbsorptionResult:
    """Result of absorbing the CellProfiler library."""

    absorbed: List[AbsorbedFunction] = field(default_factory=list)
    failed: List[Tuple[str, str]] = field(default_factory=list)

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
       c. Write to benchmark/cellprofiler_library/functions/
    4. Generate registry mapping
    5. Write contracts.json with function identity metadata
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
        self.source_root = (
            source_root or Path(__file__).parent.parent / "cellprofiler_source"
        )
        self.output_root = (
            output_root or Path(__file__).parent.parent / "cellprofiler_library"
        )
        self.llm_converter = llm_converter
        self.source_locator = SourceLocator(self.source_root)

    def absorb_all(self, skip_existing: bool = True) -> AbsorptionResult:
        """
        Absorb entire CellProfiler library.

        Args:
            skip_existing: Skip modules already converted

        Returns:
            AbsorptionResult with all absorption details
        """
        result = AbsorptionResult()
        functions_dir = self.output_root / "functions"
        functions_dir.mkdir(parents=True, exist_ok=True)
        modules_to_absorb: List[Tuple[Path, str, bool]] = []
        absorbed_names: set = set()
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
            logger.warning(
                f"Library modules directory not found: {library_modules_dir}"
            )
        full_modules_dir = self.source_root / "modules"
        if full_modules_dir.exists():
            full_module_count = 0
            for module_file in sorted(full_modules_dir.glob("*.py")):
                if (
                    module_file.name.startswith("_")
                    or module_file.name == "__init__.py"
                ):
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
            if skip_existing and output_file.exists():
                logger.info(f"Skipping {module_name} (already exists)")
                result.absorbed.append(
                    AbsorbedFunction(
                        cp_module_name=module_name,
                        openhcs_function_name=func_name,
                        source_file=str(output_file),
                        original_cp_file=str(module_file),
                        validated=True,
                    )
                )
                continue
            source_type = "library" if is_library else "full-class"
            try:
                absorbed = self._absorb_module(
                    module_file=module_file,
                    module_name=module_name,
                    func_name=func_name,
                    output_file=output_file,
                )
                result.absorbed.append(absorbed)
                logger.info(f"  [{source_type}] {module_name} -> {func_name}")
            except Exception as e:
                logger.error(f"Failed to absorb {module_name} [{source_type}]: {e}")
                result.failed.append((module_name, str(e)))
        self._write_registry(result)
        return result

    def _absorb_module(
        self, module_file: Path, module_name: str, func_name: str, output_file: Path
    ) -> AbsorbedFunction:
        """Absorb a single module."""
        logger.info(f"Absorbing {module_name}...")
        source_code = module_file.read_text()
        if self.llm_converter is None:
            raise RuntimeError("LLM converter not initialized")
        from openhcs.interop.cellprofiler.parser import ModuleBlock

        module_block = ModuleBlock(name=module_name, module_num=0, settings={})
        source_location = SourceLocation(
            module_name=module_name,
            library_module_path=module_file,
            source_code=source_code,
        )
        max_retries = 2
        conversion = None
        validation_errors = []
        for attempt in range(max_retries + 1):
            if attempt > 0:
                logger.warning(
                    f"  Retry attempt {attempt}/{max_retries} for {module_name}"
                )
            conversion = self.llm_converter.convert(module_block, source_location)
            if not conversion.success:
                if attempt < max_retries:
                    continue
                raise RuntimeError(f"LLM conversion failed: {conversion.error_message}")
            validation_errors = self._validate_syntax(conversion.converted_code)
            if not validation_errors:
                break
            for err in validation_errors:
                logger.error(f"  Validation: {err}")
            if attempt >= max_retries:
                raise RuntimeError(
                    f"Validation failed after {max_retries + 1} attempts: {validation_errors}"
                )
        assert conversion is not None
        assert not validation_errors
        code_with_mapping = self._inject_parameter_mapping(
            conversion.converted_code, conversion.parameter_mapping
        )
        output_file.write_text(code_with_mapping)
        logger.info(f"Wrote {output_file}")
        absorbed = AbsorbedFunction(
            cp_module_name=module_name,
            openhcs_function_name=func_name,
            source_file=str(output_file),
            original_cp_file=str(module_file),
            validated=len(validation_errors) == 0,
            validation_errors=validation_errors,
        )
        return absorbed

    def _validate_syntax(self, code: str) -> List[str]:
        """Validate Python syntax and OpenHCS contract compliance."""
        errors = []
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
            return errors
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                has_numpy_decorator = any(
                    (
                        isinstance(d, ast.Call)
                        and isinstance(d.func, ast.Name)
                        and (d.func.id == "numpy")
                        or (isinstance(d, ast.Name) and d.id == "numpy")
                        for d in node.decorator_list
                    )
                )
                if not has_numpy_decorator:
                    continue
                if not node.args.args:
                    errors.append(
                        f"{node.name}: no parameters (must have 'image' as first)"
                    )
                elif node.args.args[0].arg != "image":
                    errors.append(
                        f"{node.name}: first param is '{node.args.args[0].arg}', must be 'image'"
                    )
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level > 0:
                    dots = "." * node.level
                    errors.append(
                        f"Hallucinated relative import: from {dots}{node.module or ''}"
                    )
                if node.module and "functions." in node.module:
                    errors.append(f"Hallucinated import: from {node.module}")
        return errors

    def _write_registry(self, result: AbsorptionResult) -> None:
        """Write registry files."""
        contracts_file = self.output_root / "contracts.json"
        contracts_data = {
            f.cp_module_name: {
                "function_name": f.openhcs_function_name,
                "confidence": f.confidence,
                "reasoning": f.notes,
                "validated": f.validated,
            }
            for f in result.absorbed
        }
        contracts_file.write_text(json.dumps(contracts_data, indent=2))
        logger.info(f"Wrote {contracts_file}")
        init_file = self.output_root / "__init__.py"
        init_content = self._generate_init(result)
        init_file.write_text(init_content)
        logger.info(f"Wrote {init_file}")

    def _generate_init(self, result: AbsorptionResult) -> str:
        """Generate __init__.py with registry mapping."""
        lines = [
            '"""',
            "CellProfiler Library - Absorbed into OpenHCS",
            "",
            "Auto-generated by LibraryAbsorber.",
            "Maps CellProfiler module names to OpenHCS functions.",
            '"""',
            "",
            "from typing import Dict, Callable",
            "",
            "# Function imports",
        ]
        for f in result.absorbed:
            if f.validated:
                lines.append(
                    f"from .functions.{f.openhcs_function_name} import {f.openhcs_function_name}"
                )
        lines.extend(
            [
                "",
                "",
                "# Registry mapping CellProfiler module names to OpenHCS functions",
                "CELLPROFILER_MODULES: Dict[str, Callable] = {",
            ]
        )
        for f in result.absorbed:
            if f.validated:
                lines.append(f'    "{f.cp_module_name}": {f.openhcs_function_name},')
        lines.extend(
            [
                "}",
                "",
                "",
                "def get_function(module_name: str) -> Callable:",
                '    """Get OpenHCS function for CellProfiler module name."""',
                "    if module_name not in CELLPROFILER_MODULES:",
                '        raise KeyError(f"Unknown CellProfiler module: {module_name}")',
                "    return CELLPROFILER_MODULES[module_name]",
                "",
                "",
                "__all__ = [",
                '    "CELLPROFILER_MODULES",',
                '    "get_function",',
            ]
        )
        for f in result.absorbed:
            if f.validated:
                lines.append(f'    "{f.openhcs_function_name}",')
        lines.append("]")
        return "\n".join(lines)

    def _file_to_module_name(self, filename: str) -> str:
        """Convert _threshold.py to Threshold or identifyprimaryobjects.py to IdentifyPrimaryObjects."""
        name = filename.replace(".py", "").lstrip("_")
        parts = name.split("_")
        return "".join((word.capitalize() for word in parts))

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
        lines = code.split("\n")
        docstring_start = None
        docstring_end = None
        in_docstring = False
        for i, line in enumerate(lines):
            if '"""' in line and (not in_docstring):
                docstring_start = i
                in_docstring = True
                if line.count('"""') == 2:
                    docstring_end = i
                    break
            elif '"""' in line and in_docstring:
                docstring_end = i
                break
        if docstring_start is None or docstring_end is None:
            logger.warning("Could not find docstring to inject parameter mapping")
            return code
        mapping_lines = [
            "",
            "    CellProfiler Parameter Mapping:",
            "    (CellProfiler setting → Python parameter)",
        ]
        for cp_setting, py_param in mapping.items():
            if py_param is None:
                mapping_lines.append(
                    f"        '{cp_setting}' → (no mapping - handled by pipeline)"
                )
            elif isinstance(py_param, list):
                params_str = ", ".join(py_param)
                mapping_lines.append(f"        '{cp_setting}' → [{params_str}]")
            else:
                mapping_lines.append(f"        '{cp_setting}' → {py_param}")
        lines.insert(docstring_end, "\n".join(mapping_lines))
        return "\n".join(lines)

    def _module_to_function_name(self, module_name: str) -> str:
        """Convert ModuleName to module_name (snake_case)."""
        return re.sub("([A-Z])", "_\\1", module_name).lower().lstrip("_")
