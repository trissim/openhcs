# File: openhcs/textual_tui/widgets/shared/signature_analyzer.py

import inspect
import dataclasses
import re
from typing import Any, Dict, Callable, get_type_hints, NamedTuple, Union, Optional

class ParameterInfo(NamedTuple):
    """Information about a parameter."""
    name: str
    param_type: type
    default_value: Any
    is_required: bool
    description: Optional[str] = None  # Add parameter description from docstring

class DocstringInfo(NamedTuple):
    """Information extracted from a docstring."""
    summary: Optional[str] = None  # First line or brief description
    description: Optional[str] = None  # Full description
    parameters: Dict[str, str] = None  # Parameter name -> description mapping
    returns: Optional[str] = None  # Return value description
    examples: Optional[str] = None  # Usage examples

class DocstringExtractor:
    """Extract structured information from docstrings."""

    @staticmethod
    def extract(target: Union[Callable, type]) -> DocstringInfo:
        """Extract docstring information from function or class.

        Args:
            target: Function, method, or class to extract docstring from

        Returns:
            DocstringInfo with parsed docstring components
        """
        if not target:
            return DocstringInfo()

        docstring = inspect.getdoc(target)
        if not docstring:
            return DocstringInfo()

        return DocstringExtractor._parse_docstring(docstring)

    @staticmethod
    def _parse_docstring(docstring: str) -> DocstringInfo:
        """Parse a docstring into structured components.

        Supports multiple docstring formats:
        - Google style (Args:, Returns:, Examples:)
        - NumPy style (Parameters, Returns, Examples)
        - Sphinx style (:param name:, :returns:)
        - Simple format (just description)
        """
        lines = docstring.strip().split('\n')

        # Extract summary (first non-empty line)
        summary = None
        description_lines = []
        parameters = {}
        returns = None
        examples = None

        current_section = 'description'
        current_param = None

        for line in lines:
            line = line.strip()

            # Skip empty lines in description
            if not line and current_section == 'description':
                if description_lines:  # Only add if we have content
                    description_lines.append('')
                continue

            # Check for section headers
            if line.lower() in ('args:', 'arguments:', 'parameters:'):
                current_section = 'parameters'
                continue
            elif line.lower() in ('returns:', 'return:'):
                current_section = 'returns'
                continue
            elif line.lower() in ('examples:', 'example:'):
                current_section = 'examples'
                continue

            # Parse content based on current section
            if current_section == 'description':
                if not summary and line:
                    summary = line
                else:
                    description_lines.append(line)

            elif current_section == 'parameters':
                # Handle different parameter formats
                param_match = re.match(r'^(\w+):\s*(.+)', line)
                sphinx_match = re.match(r'^:param\s+(\w+):\s*(.+)', line)

                if param_match:
                    param_name, param_desc = param_match.groups()
                    parameters[param_name] = param_desc.strip()
                    current_param = param_name
                elif sphinx_match:
                    param_name, param_desc = sphinx_match.groups()
                    parameters[param_name] = param_desc.strip()
                    current_param = param_name
                elif current_param and line.startswith(' '):
                    # Continuation of previous parameter description
                    parameters[current_param] += ' ' + line.strip()
                elif line and not line.startswith(' '):
                    # New parameter without colon format (simple list)
                    words = line.split()
                    if words:
                        param_name = words[0].rstrip(':')
                        param_desc = ' '.join(words[1:]) if len(words) > 1 else ''
                        parameters[param_name] = param_desc
                        current_param = param_name

            elif current_section == 'returns':
                if returns is None:
                    returns = line
                else:
                    returns += ' ' + line

            elif current_section == 'examples':
                if examples is None:
                    examples = line
                else:
                    examples += '\n' + line

        # Clean up description
        description = '\n'.join(description_lines).strip() if description_lines else None
        if description == summary:
            description = None  # Avoid duplication

        return DocstringInfo(
            summary=summary,
            description=description,
            parameters=parameters or {},
            returns=returns,
            examples=examples
        )

class SignatureAnalyzer:
    """Universal analyzer for extracting parameter information from any target."""
    
    @staticmethod
    def analyze(target: Union[Callable, type]) -> Dict[str, ParameterInfo]:
        """Extract parameter information from any target: function, constructor, or dataclass.
        
        Args:
            target: Function, constructor, or dataclass type
            
        Returns:
            Dict mapping parameter names to ParameterInfo
        """
        if not target:
            return {}
        
        # Dispatch based on target type
        if dataclasses.is_dataclass(target):
            return SignatureAnalyzer._analyze_dataclass(target)
        else:
            return SignatureAnalyzer._analyze_callable(target)
    
    @staticmethod
    def _analyze_callable(callable_obj: Callable) -> Dict[str, ParameterInfo]:
        """Extract parameter information from callable signature."""
        try:
            sig = inspect.signature(callable_obj)
            type_hints = get_type_hints(callable_obj)

            # Extract docstring information
            docstring_info = DocstringExtractor.extract(callable_obj)

            parameters = {}

            param_list = list(sig.parameters.items())

            for i, (param_name, param) in enumerate(param_list):
                # Skip self, cls, kwargs - parent can filter more if needed
                if param_name in ('self', 'cls', 'kwargs'):
                    continue

                # Skip the first parameter (after self/cls) - this is always the image/tensor
                # that gets passed automatically by the processing system
                if i == 0 or (i == 1 and param_list[0][0] in ('self', 'cls')):
                    continue

                param_type = type_hints.get(param_name, str)
                default_value = param.default if param.default != inspect.Parameter.empty else None
                is_required = param.default == inspect.Parameter.empty

                # Get parameter description from docstring
                param_description = docstring_info.parameters.get(param_name)

                parameters[param_name] = ParameterInfo(
                    name=param_name,
                    param_type=param_type,
                    default_value=default_value,
                    is_required=is_required,
                    description=param_description
                )

            return parameters
            
        except Exception:
            # Return empty dict on error
            return {}
    
    @staticmethod
    def _analyze_dataclass(dataclass_type: type) -> Dict[str, ParameterInfo]:
        """Extract parameter information from dataclass fields."""
        try:
            type_hints = get_type_hints(dataclass_type)

            # Extract docstring information from dataclass
            docstring_info = DocstringExtractor.extract(dataclass_type)

            parameters = {}

            for field in dataclasses.fields(dataclass_type):
                param_type = type_hints.get(field.name, str)

                # Get default value
                if field.default != dataclasses.MISSING:
                    default_value = field.default
                    is_required = False
                elif field.default_factory != dataclasses.MISSING:
                    default_value = field.default_factory()
                    is_required = False
                else:
                    default_value = None
                    is_required = True

                # Get field description from docstring or field metadata
                field_description = None
                if hasattr(field, 'metadata') and 'description' in field.metadata:
                    field_description = field.metadata['description']
                else:
                    field_description = docstring_info.parameters.get(field.name)

                parameters[field.name] = ParameterInfo(
                    name=field.name,
                    param_type=param_type,
                    default_value=default_value,
                    is_required=is_required,
                    description=field_description
                )

            return parameters
            
        except Exception:
            # Return empty dict on error
            return {}
