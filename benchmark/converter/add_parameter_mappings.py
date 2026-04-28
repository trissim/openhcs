"""
Add CellProfiler parameter name mappings to absorbed function docstrings.

Parses .cppipe files to extract CellProfiler setting names, then updates
function docstrings with a mapping section showing which CellProfiler
settings correspond to which simplified parameter names.

Single source of truth: mappings live in the docstrings themselves.
"""

import ast
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import inspect

from benchmark.converter.settings_binder import normalize_cellprofiler_setting_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParameterMappingGenerator:
    """Generate parameter mappings from CellProfiler settings to simplified names."""

    def __init__(self):
        """Initialize generator."""
        self.library_root = Path(__file__).parent.parent / "cellprofiler_library"
        self.pipelines_root = Path(__file__).parent.parent / "cellprofiler_pipelines"

    def extract_cellprofiler_settings(self, module_name: str) -> List[Tuple[str, str]]:
        """
        Extract CellProfiler setting names from .cppipe files.

        Args:
            module_name: CellProfiler module name (e.g., "IdentifyPrimaryObjects")

        Returns:
            List of (setting_key, setting_value) tuples
        """
        settings = []

        # Search all .cppipe files for this module
        for cppipe_file in self.pipelines_root.glob("*.cppipe"):
            content = cppipe_file.read_text()

            # Find module blocks
            pattern = rf'^{module_name}:\[.*?\]$'
            matches = re.finditer(pattern, content, re.MULTILINE)

            for match in matches:
                # Find settings after this module header
                start_pos = match.end()
                lines = content[start_pos:].split('\n')

                for line in lines:
                    # Stop at next module
                    if line and not line.startswith('    '):
                        break

                    # Parse setting line: "    Setting name:value"
                    if ':' in line:
                        key, value = line.strip().split(':', 1)
                        settings.append((key, value))

        return settings

    def _extract_function_parameters(self, lines: List[str], func_start: int) -> List[str]:
        """Extract parameter names from function signature."""
        params = []

        # Find the closing paren of the function signature
        in_signature = False
        for i in range(func_start, min(func_start + 30, len(lines))):
            line = lines[i]

            if 'def ' in line:
                in_signature = True

            if in_signature:
                # Extract parameter names from this line
                # Match patterns like "param_name: type = default" or "param_name: type"
                matches = re.findall(r'(\w+)\s*:', line)
                for match in matches:
                    if match != 'image' and match not in params:
                        params.append(match)

                if ')' in line and '->' in line:
                    break

        return params

    def _match_parameter(self, normalized_setting: str, func_params: List[str]) -> Optional[str]:
        """
        Match a normalized CellProfiler setting to a function parameter.

        Uses fuzzy matching and common patterns.
        """
        # Direct match
        if normalized_setting in func_params:
            return normalized_setting

        # Check for partial matches
        for param in func_params:
            # Check if param is a substring of setting or vice versa
            if param in normalized_setting or normalized_setting in param:
                return param

            # Check for common abbreviations
            if 'diameter' in normalized_setting and 'diameter' in param:
                if 'min' in normalized_setting and 'min' in param:
                    return param
                if 'max' in normalized_setting and 'max' in param:
                    return param

            if 'discard' in normalized_setting or 'exclude' in normalized_setting:
                if 'exclude' in param or 'discard' in param:
                    return param

            if 'border' in normalized_setting and 'border' in param:
                return param

        return None

    def update_function_docstring(self, module_name: str, function_name: str):
        """
        Update a function's docstring with CellProfiler parameter mapping.

        Args:
            module_name: CellProfiler module name (e.g., "IdentifyPrimaryObjects")
            function_name: Python function name (e.g., "identify_primary_objects")
        """
        # File name is function name without underscores
        file_name = function_name.replace('_', '')
        func_file = self.library_root / "functions" / f"{file_name}.py"
        if not func_file.exists():
            logger.warning(f"Function file not found: {func_file}")
            return

        # Extract CellProfiler settings
        settings = self.extract_cellprofiler_settings(module_name)
        if not settings:
            logger.info(f"  No settings found for {module_name}, skipping")
            return

        # Read current file
        code = func_file.read_text()
        lines = code.split('\n')

        # Find the function definition and its docstring
        func_start = None
        docstring_start = None
        docstring_end = None

        for i, line in enumerate(lines):
            if f'def {function_name}(' in line:
                func_start = i
                # Look for docstring (might be several lines after due to multi-line signature)
                for j in range(i + 1, min(i + 30, len(lines))):
                    if '"""' in lines[j]:
                        if docstring_start is None:
                            docstring_start = j
                        elif '"""' in lines[j] and j > docstring_start:
                            docstring_end = j
                            break
                break

        if func_start is None:
            logger.warning(f"  Could not find function {function_name}")
            return

        if docstring_start is None:
            logger.warning(f"  Could not find docstring for {function_name}")
            return

        # Get function parameters
        func_params = self._extract_function_parameters(lines, func_start)

        # Build mapping section with actual parameter names
        mapping_lines = [
            "",
            "    CellProfiler Parameter Mapping:",
            "    (CellProfiler setting → Python parameter)",
        ]

        for setting_key, setting_value in settings[:15]:  # Limit for readability
            normalized = normalize_cellprofiler_setting_name(setting_key)

            # Try to find matching parameter
            matched_param = self._match_parameter(normalized, func_params)

            if matched_param:
                mapping_lines.append(f"        '{setting_key}' → {matched_param}")
            else:
                mapping_lines.append(f"        '{setting_key}' → (no direct mapping)")

        # Insert mapping before closing docstring
        if docstring_end:
            lines.insert(docstring_end, '\n'.join(mapping_lines))
        else:
            # No closing docstring found, append before function body
            lines.insert(docstring_start + 1, '\n'.join(mapping_lines) + '\n    """')

        # Write back
        func_file.write_text('\n'.join(lines))
        logger.info(f"  ✅ Updated {function_name}")

    def update_all_docstrings(self):
        """Update docstrings for all absorbed functions."""
        # Load contracts to get all function names
        contracts_file = self.library_root / "contracts.json"
        contracts = json.loads(contracts_file.read_text())

        for module_name, meta in contracts.items():
            function_name = meta["function_name"]
            logger.info(f"Processing {module_name} → {function_name}")
            self.update_function_docstring(module_name, function_name)


def main():
    """Main entry point."""
    import sys

    generator = ParameterMappingGenerator()

    if len(sys.argv) > 1:
        # Update specific function
        module_name = sys.argv[1]
        contracts_file = generator.library_root / "contracts.json"
        contracts = json.loads(contracts_file.read_text())

        if module_name in contracts:
            function_name = contracts[module_name]["function_name"]
            generator.update_function_docstring(module_name, function_name)
        else:
            logger.error(f"Module {module_name} not found in contracts")
    else:
        # Update all functions
        generator.update_all_docstrings()


if __name__ == "__main__":
    main()
