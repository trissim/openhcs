# File: openhcs/textual_tui/widgets/shared/signature_analyzer.py

import ast
import inspect
import dataclasses
import re
from typing import Any, Dict, Callable, get_type_hints, NamedTuple, Union, Optional, Type

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

        # Try AST-based parsing first for better accuracy
        try:
            return DocstringExtractor._parse_docstring_ast(target, docstring)
        except Exception:
            # Fall back to regex-based parsing
            return DocstringExtractor._parse_docstring(docstring)

    @staticmethod
    def _parse_docstring_ast(target: Union[Callable, type], docstring: str) -> DocstringInfo:
        """Parse docstring using AST for more accurate extraction.

        This method uses AST to parse the source code and extract docstring
        information more accurately, especially for complex multiline descriptions.
        """
        try:
            # Get source code
            source = inspect.getsource(target)
            tree = ast.parse(source)

            # Find the function/class node
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    if ast.get_docstring(node) == docstring:
                        return DocstringExtractor._parse_ast_docstring(node, docstring)

            # Fallback to regex parsing if AST parsing fails
            return DocstringExtractor._parse_docstring(docstring)

        except Exception:
            # Fallback to regex parsing
            return DocstringExtractor._parse_docstring(docstring)

    @staticmethod
    def _parse_ast_docstring(node: Union[ast.FunctionDef, ast.ClassDef], docstring: str) -> DocstringInfo:
        """Parse docstring from AST node with enhanced multiline support."""
        # For now, use the improved regex parser
        # This can be extended later with more sophisticated AST-based parsing
        return DocstringExtractor._parse_docstring(docstring)

    @staticmethod
    def _parse_docstring(docstring: str) -> DocstringInfo:
        """Parse a docstring into structured components with improved multiline support.

        Supports multiple docstring formats:
        - Google style (Args:, Returns:, Examples:)
        - NumPy style (Parameters, Returns, Examples)
        - Sphinx style (:param name:, :returns:)
        - Simple format (just description)

        Uses improved parsing for multiline parameter descriptions that continues
        until a blank line or new parameter/section is encountered.
        """
        lines = docstring.strip().split('\n')

        summary = None
        description_lines = []
        parameters = {}
        returns = None
        examples = None

        current_section = 'description'
        current_param = None
        current_param_lines = []

        def _finalize_current_param():
            """Finalize the current parameter description."""
            if current_param and current_param_lines:
                param_desc = '\n'.join(current_param_lines).strip()
                parameters[current_param] = param_desc
            
        for i, line in enumerate(lines):
            original_line = line
            line = line.strip()

            # Handle both Google/Sphinx style (with colons) and NumPy style (without colons)
            if line.lower() in ('args:', 'arguments:', 'parameters:'):
                _finalize_current_param()
                current_param = None
                current_param_lines = []
                current_section = 'parameters'
                if i + 1 < len(lines) and lines[i+1].strip().startswith('---'): # Skip NumPy style separator
                    continue
                continue
            elif line.lower() in ('args', 'arguments', 'parameters') and i + 1 < len(lines) and lines[i+1].strip().startswith('-'):
                # NumPy-style section headers (without colons, followed by dashes)
                _finalize_current_param()
                current_param = None
                current_param_lines = []
                current_section = 'parameters'
                continue
            elif line.lower() in ('returns:', 'return:'):
                _finalize_current_param()
                current_param = None
                current_param_lines = []
                current_section = 'returns'
                if i + 1 < len(lines) and lines[i+1].strip().startswith('---'): # Skip NumPy style separator
                    continue
                continue
            elif line.lower() in ('returns', 'return') and i + 1 < len(lines) and lines[i+1].strip().startswith('-'):
                # NumPy-style returns section
                _finalize_current_param()
                current_param = None
                current_param_lines = []
                current_section = 'returns'
                continue
            elif line.lower() in ('examples:', 'example:'):
                _finalize_current_param()
                current_param = None
                current_param_lines = []
                current_section = 'examples'
                if i + 1 < len(lines) and lines[i+1].strip().startswith('---'): # Skip NumPy style separator
                    continue
                continue
            elif line.lower() in ('examples', 'example') and i + 1 < len(lines) and lines[i+1].strip().startswith('-'):
                # NumPy-style examples section
                _finalize_current_param()
                current_param = None
                current_param_lines = []
                current_section = 'examples'
                continue

            if current_section == 'description':
                if not summary and line:
                    summary = line
                else:
                    description_lines.append(original_line) # Keep original indentation

            elif current_section == 'parameters':
                # Enhanced parameter parsing to handle multiple formats
                param_match_google = re.match(r'^(\w+):\s*(.+)', line)
                param_match_sphinx = re.match(r'^:param\s+(\w+):\s*(.+)', line)
                param_match_numpy = re.match(r'^(\w+)\s*:\s*(.+)', line)
                # New: Handle pyclesperanto-style inline parameters (param_name: type description)
                param_match_inline = re.match(r'^(\w+):\s*(\w+(?:\[.*?\])?|\w+(?:\s*\|\s*\w+)*)\s+(.+)', line)
                # New: Handle parameters that start with bullet points or dashes
                param_match_bullet = re.match(r'^[-•*]\s*(\w+):\s*(.+)', line)

                if param_match_google or param_match_sphinx or param_match_numpy or param_match_inline or param_match_bullet:
                    _finalize_current_param()

                    if param_match_google:
                        param_name, param_desc = param_match_google.groups()
                    elif param_match_sphinx:
                        param_name, param_desc = param_match_sphinx.groups()
                    elif param_match_numpy:
                        param_name, param_desc = param_match_numpy.groups()
                    elif param_match_inline:
                        param_name, param_type, param_desc = param_match_inline.groups()
                        param_desc = f"{param_type} - {param_desc}"  # Include type in description
                    elif param_match_bullet:
                        param_name, param_desc = param_match_bullet.groups()

                    current_param = param_name
                    current_param_lines = [param_desc.strip()]
                elif current_param and (original_line.startswith('    ') or original_line.startswith('\t')):
                    # Indented continuation line
                    current_param_lines.append(line)
                elif not line:
                    _finalize_current_param()
                    current_param = None
                    current_param_lines = []
                elif current_param:
                    # Non-indented continuation line (part of the same block)
                    current_param_lines.append(line)
                else:
                    # Try to parse inline parameter definitions in a single block
                    # This handles cases where parameters are listed without clear separation
                    inline_params = DocstringExtractor._parse_inline_parameters(line)
                    for param_name, param_desc in inline_params.items():
                        parameters[param_name] = param_desc
            
            elif current_section == 'returns':
                if returns is None:
                    returns = line
                else:
                    returns += '\n' + line
            
            elif current_section == 'examples':
                if examples is None:
                    examples = line
                else:
                    examples += '\n' + line

        _finalize_current_param()

        description = '\n'.join(description_lines).strip()
        if description == summary:
            description = None

        return DocstringInfo(
            summary=summary,
            description=description,
            parameters=parameters or {},
            returns=returns,
            examples=examples
        )

    @staticmethod
    def _parse_inline_parameters(line: str) -> Dict[str, str]:
        """Parse parameters from a single line containing multiple parameter definitions.

        Handles formats like:
        - "input_image: Image Input image to process. footprint: Image Structuring element..."
        - "param1: type1 description1. param2: type2 description2."
        """
        parameters = {}

        import re

        # Strategy: Use a flexible pattern that works with the pyclesperanto format
        # Pattern matches: param_name: everything up to the next param_name: or end of string
        param_pattern = r'(\w+):\s*([^:]*?)(?=\s+\w+:|$)'
        matches = re.findall(param_pattern, line)

        for param_name, param_desc in matches:
            if param_desc.strip():
                # Clean up the description (remove trailing periods, extra whitespace)
                clean_desc = param_desc.strip().rstrip('.')
                parameters[param_name] = clean_desc

        return parameters


class SignatureAnalyzer:
    """Universal analyzer for extracting parameter information from any target."""
    
    @staticmethod
    def analyze(target: Union[Callable, Type, object]) -> Dict[str, ParameterInfo]:
        """Extract parameter information from any target: function, constructor, dataclass, or instance.

        Args:
            target: Function, constructor, dataclass type, or dataclass instance

        Returns:
            Dict mapping parameter names to ParameterInfo
        """
        if not target:
            return {}

        # Dispatch based on target type
        if inspect.isclass(target):
            if dataclasses.is_dataclass(target):
                return SignatureAnalyzer._analyze_dataclass(target)
            else:
                # Try to analyze constructor
                return SignatureAnalyzer._analyze_callable(target.__init__)
        elif dataclasses.is_dataclass(target):
            # Instance of dataclass
            return SignatureAnalyzer._analyze_dataclass_instance(target)
        else:
            # Function, method, or other callable
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
                # Skip self, cls - parent can filter more if needed
                if param_name in ('self', 'cls'):
                    continue

                # Skip the first parameter (after self/cls) - this is always the image/tensor
                # that gets passed automatically by the processing system
                if i == 0 or (i == 1 and param_list[0][0] in ('self', 'cls')):
                    continue

                # Handle **kwargs parameters - try to extract original function signature
                if param.kind == inspect.Parameter.VAR_KEYWORD:
                    # Try to find the original function if this is a wrapper
                    original_params = SignatureAnalyzer._extract_original_parameters(callable_obj)
                    if original_params:
                        parameters.update(original_params)
                    continue

                from typing import Any
                param_type = type_hints.get(param_name, Any)
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
    def _extract_original_parameters(callable_obj: Callable) -> Dict[str, ParameterInfo]:
        """
        Extract parameters from the original function if this is a wrapper with **kwargs.

        This handles cases where scikit-image or other auto-registered functions
        are wrapped with (image, **kwargs) signatures.
        """
        try:
            # Check if this function has access to the original function
            # Common patterns: __wrapped__, closure variables, etc.

            # Pattern 1: Check if it's a functools.wraps wrapper
            if hasattr(callable_obj, '__wrapped__'):
                return SignatureAnalyzer._analyze_callable(callable_obj.__wrapped__)

            # Pattern 2: Check closure for original function reference
            if hasattr(callable_obj, '__closure__') and callable_obj.__closure__:
                for cell in callable_obj.__closure__:
                    if hasattr(cell.cell_contents, '__call__'):
                        # Found a callable in closure - might be the original function
                        try:
                            orig_sig = inspect.signature(cell.cell_contents)
                            # Skip if it also has **kwargs (avoid infinite recursion)
                            if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in orig_sig.parameters.values()):
                                continue
                            return SignatureAnalyzer._analyze_callable(cell.cell_contents)
                        except:
                            continue

            # Pattern 3: Try to extract from function name and module
            # This is a fallback for scikit-image functions
            if hasattr(callable_obj, '__name__') and hasattr(callable_obj, '__module__'):
                func_name = callable_obj.__name__
                module_name = callable_obj.__module__

                # Try to find the original function in scikit-image
                if 'skimage' in module_name:
                    try:
                        import importlib
                        # Extract the actual module path (remove wrapper module parts)
                        if 'scikit_image_registry' in module_name:
                            # This is our wrapper, try to find the original in skimage
                            for skimage_module in ['skimage.filters', 'skimage.morphology',
                                                 'skimage.segmentation', 'skimage.feature',
                                                 'skimage.measure', 'skimage.transform',
                                                 'skimage.restoration', 'skimage.exposure']:
                                try:
                                    mod = importlib.import_module(skimage_module)
                                    if hasattr(mod, func_name):
                                        orig_func = getattr(mod, func_name)
                                        return SignatureAnalyzer._analyze_callable(orig_func)
                                except:
                                    continue
                    except:
                        pass

            return {}

        except Exception:
            return {}

    @staticmethod
    def _analyze_dataclass(dataclass_type: type) -> Dict[str, ParameterInfo]:
        """Extract parameter information from dataclass fields."""
        try:
            type_hints = get_type_hints(dataclass_type)

            # Extract docstring information from dataclass
            docstring_info = DocstringExtractor.extract(dataclass_type)

            # Extract inline field documentation using AST
            inline_docs = SignatureAnalyzer._extract_inline_field_docs(dataclass_type)

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

                # Get field description from multiple sources (priority order)
                field_description = None

                # 1. Field metadata (highest priority)
                if hasattr(field, 'metadata') and 'description' in field.metadata:
                    field_description = field.metadata['description']
                # 2. Inline documentation strings (new!)
                elif field.name in inline_docs:
                    field_description = inline_docs[field.name]
                # 3. Docstring parameters (fallback)
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

    @staticmethod
    def _extract_inline_field_docs(dataclass_type: type) -> Dict[str, str]:
        """Extract inline field documentation strings using AST parsing.

        This handles multiple patterns used for field documentation:

        Pattern 1 - Next line string literal:
        @dataclass
        class Config:
            field_name: str = "default"
            '''Field description here.'''

        Pattern 2 - Same line string literal (less common):
        @dataclass
        class Config:
            field_name: str = "default"  # '''Field description'''

        Pattern 3 - Traditional docstring parameters (handled by DocstringExtractor):
        @dataclass
        class Config:
            '''
            Args:
                field_name: Field description here.
            '''
            field_name: str = "default"
        """
        try:
            import ast
            import re

            source = inspect.getsource(dataclass_type)
            tree = ast.parse(source)

            # Find the class definition
            class_node = None
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == dataclass_type.__name__:
                    class_node = node
                    break

            if not class_node:
                return {}

            field_docs = {}
            source_lines = source.split('\n')

            # Method 1: Look for field assignments followed by string literals (next line)
            for i, node in enumerate(class_node.body):
                if isinstance(node, ast.AnnAssign) and hasattr(node.target, 'id'):
                    field_name = node.target.id

                    # Check if the next node is a string literal (documentation)
                    if i + 1 < len(class_node.body):
                        next_node = class_node.body[i + 1]
                        if isinstance(next_node, ast.Expr) and isinstance(next_node.value, ast.Constant):
                            if isinstance(next_node.value.value, str):
                                field_docs[field_name] = next_node.value.value.strip()
                                continue

                    # Method 2: Check for inline comments on the same line
                    # Get the line number of the field definition
                    field_line_num = node.lineno - 1  # Convert to 0-based indexing
                    if 0 <= field_line_num < len(source_lines):
                        line = source_lines[field_line_num]

                        # Look for string literals in comments on the same line
                        # Pattern: field: type = value  # """Documentation"""
                        comment_match = re.search(r'#\s*["\']([^"\']+)["\']', line)
                        if comment_match:
                            field_docs[field_name] = comment_match.group(1).strip()
                            continue

                        # Look for triple-quoted strings on the same line
                        # Pattern: field: type = value  """Documentation"""
                        triple_quote_match = re.search(r'"""([^"]+)"""|\'\'\'([^\']+)\'\'\'', line)
                        if triple_quote_match:
                            doc_text = triple_quote_match.group(1) or triple_quote_match.group(2)
                            field_docs[field_name] = doc_text.strip()

            return field_docs

        except Exception as e:
            # Return empty dict if AST parsing fails
            # Could add logging here for debugging: logger.debug(f"AST parsing failed: {e}")
            return {}

    @staticmethod
    def _analyze_dataclass_instance(instance: object) -> Dict[str, ParameterInfo]:
        """Extract parameter information from a dataclass instance."""
        try:
            # Get the type and analyze it
            dataclass_type = type(instance)
            parameters = SignatureAnalyzer._analyze_dataclass(dataclass_type)

            # Update default values with current instance values
            for name, param_info in parameters.items():
                if hasattr(instance, name):
                    current_value = getattr(instance, name)
                    # Create new ParameterInfo with current value as default
                    parameters[name] = ParameterInfo(
                        name=param_info.name,
                        param_type=param_info.param_type,
                        default_value=current_value,
                        is_required=param_info.is_required,
                        description=param_info.description
                    )

            return parameters

        except Exception:
            return {}

    @staticmethod
    def _analyze_dataclass_instance(instance: object) -> Dict[str, ParameterInfo]:
        """Extract parameter information from a dataclass instance."""
        try:
            # Get the type and analyze it
            dataclass_type = type(instance)
            parameters = SignatureAnalyzer._analyze_dataclass(dataclass_type)

            # Update default values with current instance values
            for name, param_info in parameters.items():
                if hasattr(instance, name):
                    current_value = getattr(instance, name)
                    # Create new ParameterInfo with current value as default
                    parameters[name] = ParameterInfo(
                        name=param_info.name,
                        param_type=param_info.param_type,
                        default_value=current_value,
                        is_required=param_info.is_required,
                        description=param_info.description
                    )

            return parameters

        except Exception:
            return {}
