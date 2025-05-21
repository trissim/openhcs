"""
Side effect detector module for AST verification.

This module provides functionality for detecting potential side effects of code changes.
"""

import ast
import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from semantic_matrix_analyzer.verification.suggestion import CodeSuggestion

logger = logging.getLogger(__name__)


class SideEffectDetector:
    """Detects potential side effects of code changes."""
    
    def detect_side_effects(self, suggestion: CodeSuggestion) -> List[str]:
        """Detect potential side effects of a code change.
        
        Args:
            suggestion: The code suggestion to check.
            
        Returns:
            A list of potential side effects.
        """
        side_effects = []
        
        try:
            # Parse the original and suggested code
            original_ast = ast.parse(suggestion.original_code)
            suggested_ast = ast.parse(suggestion.suggested_code)
            
            # Check for changes to function signatures
            function_changes = self._detect_function_signature_changes(original_ast, suggested_ast)
            side_effects.extend(function_changes)
            
            # Check for changes to class interfaces
            class_changes = self._detect_class_interface_changes(original_ast, suggested_ast)
            side_effects.extend(class_changes)
            
            # Check for changes to global variables
            global_changes = self._detect_global_variable_changes(original_ast, suggested_ast)
            side_effects.extend(global_changes)
            
            # Check for changes to imports
            import_changes = self._detect_import_changes(original_ast, suggested_ast)
            side_effects.extend(import_changes)
            
            return side_effects
        except Exception as e:
            logger.error(f"Error detecting side effects: {e}")
            return [f"Error detecting side effects: {str(e)}"]
    
    def _detect_function_signature_changes(self, original_ast: ast.AST, modified_ast: ast.AST) -> List[str]:
        """Detect changes to function signatures.
        
        Args:
            original_ast: The original AST.
            modified_ast: The modified AST.
            
        Returns:
            A list of function signature changes.
        """
        changes = []
        
        # Extract function signatures from both ASTs
        original_signatures = self._extract_function_signatures(original_ast)
        modified_signatures = self._extract_function_signatures(modified_ast)
        
        # Check for changes to existing functions
        for name, sig in original_signatures.items():
            if name in modified_signatures:
                # Check if the signature has changed
                if sig != modified_signatures[name]:
                    original_args, original_defaults = sig
                    modified_args, modified_defaults = modified_signatures[name]
                    
                    # Check for added or removed parameters
                    if len(original_args) != len(modified_args):
                        changes.append(f"Changed number of parameters for function '{name}'")
                    
                    # Check for renamed parameters
                    for i, (orig_arg, mod_arg) in enumerate(zip(original_args, modified_args)):
                        if orig_arg != mod_arg:
                            changes.append(f"Renamed parameter '{orig_arg}' to '{mod_arg}' in function '{name}'")
                    
                    # Check for changed default values
                    for i, (orig_default, mod_default) in enumerate(zip(original_defaults, modified_defaults)):
                        if orig_default != mod_default:
                            changes.append(f"Changed default value for parameter in function '{name}'")
            else:
                # Function was removed
                changes.append(f"Removed function '{name}'")
        
        # Check for added functions
        for name in modified_signatures:
            if name not in original_signatures:
                changes.append(f"Added function '{name}'")
        
        return changes
    
    def _extract_function_signatures(self, tree: ast.AST) -> Dict[str, Tuple[List[str], List[str]]]:
        """Extract function signatures from an AST.
        
        Args:
            tree: The AST to extract function signatures from.
            
        Returns:
            A dictionary mapping function names to (args, defaults).
        """
        signatures = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                args = [arg.arg for arg in node.args.args]
                defaults = [ast.dump(default) for default in node.args.defaults]
                signatures[node.name] = (args, defaults)
        
        return signatures
    
    def _detect_class_interface_changes(self, original_ast: ast.AST, modified_ast: ast.AST) -> List[str]:
        """Detect changes to class interfaces.
        
        Args:
            original_ast: The original AST.
            modified_ast: The modified AST.
            
        Returns:
            A list of class interface changes.
        """
        changes = []
        
        # Extract class interfaces from both ASTs
        original_interfaces = self._extract_class_interfaces(original_ast)
        modified_interfaces = self._extract_class_interfaces(modified_ast)
        
        # Check for changes to existing classes
        for name, interface in original_interfaces.items():
            if name in modified_interfaces:
                # Check if the interface has changed
                if interface != modified_interfaces[name]:
                    # Check for added methods
                    added_methods = modified_interfaces[name] - interface
                    if added_methods:
                        changes.append(f"Added methods to class '{name}': {', '.join(added_methods)}")
                    
                    # Check for removed methods
                    removed_methods = interface - modified_interfaces[name]
                    if removed_methods:
                        changes.append(f"Removed methods from class '{name}': {', '.join(removed_methods)}")
            else:
                # Class was removed
                changes.append(f"Removed class '{name}'")
        
        # Check for added classes
        for name in modified_interfaces:
            if name not in original_interfaces:
                changes.append(f"Added class '{name}'")
        
        return changes
    
    def _extract_class_interfaces(self, tree: ast.AST) -> Dict[str, Set[str]]:
        """Extract class interfaces from an AST.
        
        Args:
            tree: The AST to extract class interfaces from.
            
        Returns:
            A dictionary mapping class names to sets of method names.
        """
        interfaces = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = set()
                for child in node.body:
                    if isinstance(child, ast.FunctionDef):
                        methods.add(child.name)
                interfaces[node.name] = methods
        
        return interfaces
    
    def _detect_global_variable_changes(self, original_ast: ast.AST, modified_ast: ast.AST) -> List[str]:
        """Detect changes to global variables.
        
        Args:
            original_ast: The original AST.
            modified_ast: The modified AST.
            
        Returns:
            A list of global variable changes.
        """
        changes = []
        
        # Extract global variables from both ASTs
        original_globals = self._extract_global_variables(original_ast)
        modified_globals = self._extract_global_variables(modified_ast)
        
        # Check for changes to existing globals
        for name, value in original_globals.items():
            if name in modified_globals:
                # Check if the value has changed
                if value != modified_globals[name]:
                    changes.append(f"Changed value of global variable '{name}'")
            else:
                # Global was removed
                changes.append(f"Removed global variable '{name}'")
        
        # Check for added globals
        for name in modified_globals:
            if name not in original_globals:
                changes.append(f"Added global variable '{name}'")
        
        return changes
    
    def _extract_global_variables(self, tree: ast.AST) -> Dict[str, str]:
        """Extract global variables from an AST.
        
        Args:
            tree: The AST to extract global variables from.
            
        Returns:
            A dictionary mapping global variable names to their string representations.
        """
        globals_dict = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and all(isinstance(target, ast.Name) for target in node.targets):
                # Check if this is a module-level assignment
                if isinstance(node.value, (ast.Constant, ast.Str, ast.Num, ast.List, ast.Dict, ast.Set)):
                    for target in node.targets:
                        globals_dict[target.id] = ast.dump(node.value)
        
        return globals_dict
    
    def _detect_import_changes(self, original_ast: ast.AST, modified_ast: ast.AST) -> List[str]:
        """Detect changes to imports.
        
        Args:
            original_ast: The original AST.
            modified_ast: The modified AST.
            
        Returns:
            A list of import changes.
        """
        changes = []
        
        # Extract imports from both ASTs
        original_imports = self._extract_imports(original_ast)
        modified_imports = self._extract_imports(modified_ast)
        
        # Check for changes to existing imports
        for module, names in original_imports.items():
            if module in modified_imports:
                # Check if the imported names have changed
                if names != modified_imports[module]:
                    # Check for added names
                    added_names = modified_imports[module] - names
                    if added_names:
                        changes.append(f"Added imports from module '{module}': {', '.join(added_names)}")
                    
                    # Check for removed names
                    removed_names = names - modified_imports[module]
                    if removed_names:
                        changes.append(f"Removed imports from module '{module}': {', '.join(removed_names)}")
            else:
                # Import was removed
                changes.append(f"Removed import of module '{module}'")
        
        # Check for added imports
        for module in modified_imports:
            if module not in original_imports:
                changes.append(f"Added import of module '{module}'")
        
        return changes
    
    def _extract_imports(self, tree: ast.AST) -> Dict[str, Set[str]]:
        """Extract imports from an AST.
        
        Args:
            tree: The AST to extract imports from.
            
        Returns:
            A dictionary mapping module names to sets of imported names.
        """
        imports = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports[name.name] = set()
            elif isinstance(node, ast.ImportFrom):
                if node.module not in imports:
                    imports[node.module] = set()
                for name in node.names:
                    imports[node.module].add(name.name)
        
        return imports
