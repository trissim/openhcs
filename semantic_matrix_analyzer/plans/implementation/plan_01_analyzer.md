# plan_01_analyzer.md
## Component: Semantic Analyzer Core

### Objective
Implement the core semantic analyzer functionality that analyzes code to extract semantic meaning, identify patterns, and build mental models. This is the foundation of the SMA system and will be used by other components.

### Plan
1. Implement the `analyze_code` method in `SemanticAnalyzer` class
   - Parse Python code using the AST module
   - Extract semantic information (classes, functions, variables, etc.)
   - Identify naming patterns and coding conventions
   - Detect code smells and potential issues
   - Return a structured analysis result

2. Implement the `analyze_error_trace` method in `SemanticAnalyzer` class
   - Parse error traces to identify error types and locations
   - Extract context information from the error
   - Identify potential root causes
   - Build a dependency chain from the error
   - Return a structured analysis result

3. Implement the `build_mental_model` method in `SemanticAnalyzer` class
   - Process analysis results to build a comprehensive mental model
   - Create relationships between components
   - Identify architectural patterns
   - Map dependencies between modules
   - Return a structured mental model

4. Implement the `generate_recommendations` method in `SemanticAnalyzer` class
   - Analyze the mental model to identify improvement opportunities
   - Generate specific, actionable recommendations
   - Prioritize recommendations based on impact
   - Provide code examples where appropriate
   - Return a list of structured recommendations

### Findings
The `SemanticAnalyzer` class in `semantic_matrix_analyzer/analyzer.py` currently has placeholder implementations for all four methods. The class is designed to take a configuration object that can be used to customize the analysis process.

The analyzer should leverage the auto-configuration system to adapt its analysis based on the codebase characteristics. It should also be able to use the configuration that has been tuned based on human feedback.

Key dependencies:
- AST module for parsing Python code
- Configuration system for customizing analysis
- Pattern matching for identifying code structures
- Natural language processing for extracting intent from comments and docstrings

### Implementation Draft
```python
"""
Semantic Matrix Analyzer core module.

This module provides the main analysis functionality for the Semantic Matrix Analyzer.
"""

import ast
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pathlib import Path

class SemanticAnalyzer:
    """
    Main analyzer class for the Semantic Matrix Analyzer.
    
    This class provides methods for analyzing code, error traces, and building
    mental models based on the analysis results.
    """
    
    def __init__(self, config=None):
        """
        Initialize the analyzer.
        
        Args:
            config: Optional configuration object or path
        """
        self.config = config
        
    def analyze_code(self, code: str) -> Dict[str, Any]:
        """
        Analyze code to extract intent.
        
        Args:
            code: Code string to analyze
            
        Returns:
            Analysis results
        """
        results = {
            "status": "success",
            "classes": [],
            "functions": [],
            "imports": [],
            "variables": [],
            "patterns": {},
            "metrics": {},
            "issues": []
        }
        
        try:
            # Parse the code into an AST
            tree = ast.parse(code)
            
            # Extract classes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = self._analyze_class(node)
                    results["classes"].append(class_info)
                elif isinstance(node, ast.FunctionDef):
                    # Only include top-level functions
                    if isinstance(node.parent, ast.Module):
                        function_info = self._analyze_function(node)
                        results["functions"].append(function_info)
                elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                    import_info = self._analyze_import(node)
                    results["imports"].extend(import_info)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            var_info = self._analyze_variable(target, node.value)
                            results["variables"].append(var_info)
            
            # Analyze patterns
            results["patterns"] = self._analyze_patterns(code, tree)
            
            # Calculate metrics
            results["metrics"] = self._calculate_metrics(tree)
            
            # Identify issues
            results["issues"] = self._identify_issues(tree, results)
            
        except SyntaxError as e:
            results["status"] = "error"
            results["error"] = f"Syntax error: {str(e)}"
        except Exception as e:
            results["status"] = "error"
            results["error"] = f"Analysis error: {str(e)}"
            
        return results
    
    def _analyze_class(self, node: ast.ClassDef) -> Dict[str, Any]:
        """Analyze a class definition."""
        methods = []
        attributes = []
        
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                methods.append(self._analyze_function(item))
            elif isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        attributes.append({
                            "name": target.id,
                            "line": target.lineno
                        })
        
        return {
            "name": node.name,
            "line": node.lineno,
            "docstring": ast.get_docstring(node),
            "methods": methods,
            "attributes": attributes,
            "bases": [base.id if isinstance(base, ast.Name) else "complex_base" for base in node.bases]
        }
    
    def _analyze_function(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """Analyze a function definition."""
        params = []
        
        for arg in node.args.args:
            param = {"name": arg.arg}
            if arg.annotation:
                if isinstance(arg.annotation, ast.Name):
                    param["type"] = arg.annotation.id
                elif isinstance(arg.annotation, ast.Attribute):
                    param["type"] = f"{arg.annotation.value.id}.{arg.annotation.attr}"
                else:
                    param["type"] = "complex_type"
            params.append(param)
        
        return_type = None
        if node.returns:
            if isinstance(node.returns, ast.Name):
                return_type = node.returns.id
            elif isinstance(node.returns, ast.Attribute):
                return_type = f"{node.returns.value.id}.{node.returns.attr}"
            else:
                return_type = "complex_type"
        
        return {
            "name": node.name,
            "line": node.lineno,
            "docstring": ast.get_docstring(node),
            "parameters": params,
            "return_type": return_type,
            "is_method": not isinstance(node.parent, ast.Module),
            "is_async": isinstance(node, ast.AsyncFunctionDef)
        }
    
    def _analyze_import(self, node: Union[ast.Import, ast.ImportFrom]) -> List[Dict[str, Any]]:
        """Analyze an import statement."""
        imports = []
        
        if isinstance(node, ast.Import):
            for name in node.names:
                imports.append({
                    "module": name.name,
                    "alias": name.asname,
                    "line": node.lineno,
                    "is_from": False
                })
        else:  # ImportFrom
            module = node.module or ""
            for name in node.names:
                imports.append({
                    "module": module,
                    "name": name.name,
                    "alias": name.asname,
                    "line": node.lineno,
                    "is_from": True
                })
        
        return imports
    
    def _analyze_variable(self, target: ast.Name, value: ast.expr) -> Dict[str, Any]:
        """Analyze a variable assignment."""
        var_type = "unknown"
        var_value = None
        
        if isinstance(value, ast.Num):
            var_type = "number"
            var_value = value.n
        elif isinstance(value, ast.Str):
            var_type = "string"
            var_value = value.s
        elif isinstance(value, ast.List):
            var_type = "list"
        elif isinstance(value, ast.Dict):
            var_type = "dict"
        elif isinstance(value, ast.Call):
            if isinstance(value.func, ast.Name):
                var_type = f"call:{value.func.id}"
            elif isinstance(value.func, ast.Attribute):
                var_type = f"call:{value.func.attr}"
        
        return {
            "name": target.id,
            "line": target.lineno,
            "type": var_type,
            "value": var_value,
            "is_constant": target.id.isupper()
        }
    
    def _analyze_patterns(self, code: str, tree: ast.AST) -> Dict[str, Any]:
        """Analyze code patterns."""
        patterns = {}
        
        # Naming patterns
        class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        function_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        variable_names = [node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)]
        
        patterns["naming"] = {
            "snake_case": sum(1 for name in function_names + variable_names if re.match(r"^[a-z][a-z0-9_]*$", name)) / max(1, len(function_names) + len(variable_names)),
            "camel_case": sum(1 for name in function_names + variable_names if re.match(r"^[a-z][a-zA-Z0-9]*$", name) and not re.match(r"^[a-z][a-z0-9_]*$", name)) / max(1, len(function_names) + len(variable_names)),
            "pascal_case": sum(1 for name in class_names if re.match(r"^[A-Z][a-zA-Z0-9]*$", name)) / max(1, len(class_names))
        }
        
        # Docstring patterns
        has_docstring = [
            node for node in ast.walk(tree) 
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)) and ast.get_docstring(node)
        ]
        all_docable = [
            node for node in ast.walk(tree) 
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module))
        ]
        patterns["documentation"] = {
            "docstring_coverage": len(has_docstring) / max(1, len(all_docable))
        }
        
        # Error handling patterns
        try_blocks = len([node for node in ast.walk(tree) if isinstance(node, ast.Try)])
        patterns["error_handling"] = {
            "try_except_count": try_blocks
        }
        
        return patterns
    
    def _calculate_metrics(self, tree: ast.AST) -> Dict[str, Any]:
        """Calculate code metrics."""
        metrics = {}
        
        # Complexity metrics
        function_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        class_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        
        metrics["complexity"] = {
            "function_count": len(function_nodes),
            "class_count": len(class_nodes),
            "average_function_length": sum(len(node.body) for node in function_nodes) / max(1, len(function_nodes)),
            "average_class_length": sum(len(node.body) for node in class_nodes) / max(1, len(class_nodes))
        }
        
        return metrics
    
    def _identify_issues(self, tree: ast.AST, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify potential issues in the code."""
        issues = []
        
        # Check for long functions
        for func in results["functions"]:
            if func.get("is_method"):
                continue
            
            func_node = next((node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == func["name"]), None)
            if func_node and len(func_node.body) > 50:
                issues.append({
                    "type": "long_function",
                    "message": f"Function '{func['name']}' is too long ({len(func_node.body)} lines)",
                    "line": func["line"],
                    "severity": "medium"
                })
        
        # Check for missing docstrings
        for func in results["functions"]:
            if not func.get("docstring") and not func["name"].startswith("_"):
                issues.append({
                    "type": "missing_docstring",
                    "message": f"Function '{func['name']}' is missing a docstring",
                    "line": func["line"],
                    "severity": "low"
                })
        
        for cls in results["classes"]:
            if not cls.get("docstring"):
                issues.append({
                    "type": "missing_docstring",
                    "message": f"Class '{cls['name']}' is missing a docstring",
                    "line": cls["line"],
                    "severity": "low"
                })
        
        # Check for unused imports
        used_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used_names.add(node.id)
        
        for imp in results["imports"]:
            if imp.get("alias") and imp["alias"] not in used_names:
                issues.append({
                    "type": "unused_import",
                    "message": f"Import '{imp['module']}' as '{imp['alias']}' is unused",
                    "line": imp["line"],
                    "severity": "low"
                })
            elif not imp.get("alias") and imp.get("name") and imp["name"] not in used_names:
                issues.append({
                    "type": "unused_import",
                    "message": f"Import '{imp['name']}' from '{imp['module']}' is unused",
                    "line": imp["line"],
                    "severity": "low"
                })
        
        return issues
```
