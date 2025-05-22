# plan_02_error_trace.md
## Component: Error Trace Analyzer

### Objective
Implement the error trace analysis functionality that parses error traces, identifies root causes, and builds mental models of the error context. This component is critical for the SMA system's ability to assist with debugging and error resolution.

### Plan
1. Complete the `analyze_error_trace` method in `SemanticAnalyzer` class
   - Parse error traces to extract error type, message, and location
   - Build a stack trace representation
   - Identify imported modules and their relationships
   - Extract variable values and types from the error context
   - Return a structured analysis result

2. Implement error pattern recognition
   - Identify common error patterns (e.g., NameError, TypeError, ImportError)
   - Extract semantic meaning from error messages
   - Map errors to potential root causes
   - Suggest potential fixes based on error type

3. Implement context building for errors
   - Extract the code context around the error
   - Identify relevant variables and their values
   - Build a dependency graph of the affected components
   - Map the error to architectural components

4. Integrate with the mental model builder
   - Ensure error trace analysis results can be used by the mental model builder
   - Add error-specific information to the mental model
   - Provide context for recommendation generation

### Findings
The `analyze_error_trace` method in `SemanticAnalyzer` class currently has a placeholder implementation. The method should parse error traces, which typically include:

1. Error type (e.g., `NameError`, `TypeError`, `ImportError`)
2. Error message
3. File path and line number
4. Stack trace showing the call hierarchy

The error trace analyzer should be able to handle different error formats and extract meaningful information from them. It should also be able to correlate the error with the codebase structure to provide context for the error.

Key dependencies:
- Regular expressions for parsing error traces
- AST module for analyzing code context
- Configuration system for customizing analysis
- Mental model builder for integrating error information

### Implementation Draft
```python
def analyze_error_trace(self, error_trace: str) -> Dict[str, Any]:
    """
    Analyze an error trace to identify root causes.
    
    Args:
        error_trace: Error trace string to analyze
        
    Returns:
        Analysis results
    """
    results = {
        "status": "success",
        "error_type": None,
        "error_message": None,
        "file_path": None,
        "line_number": None,
        "stack_trace": [],
        "context": {},
        "potential_causes": [],
        "suggested_fixes": []
    }
    
    try:
        # Parse the error trace
        lines = error_trace.strip().split('\n')
        
        # Extract the error type and message
        error_pattern = r"^([A-Za-z0-9_.]+Error|Exception):\s*(.+)$"
        for line in lines:
            match = re.search(error_pattern, line)
            if match:
                results["error_type"] = match.group(1)
                results["error_message"] = match.group(2)
                break
        
        # Extract file paths and line numbers
        file_pattern = r"File \"([^\"]+)\", line (\d+)"
        for line in lines:
            match = re.search(file_pattern, line)
            if match:
                file_path = match.group(1)
                line_number = int(match.group(2))
                
                # If this is the first occurrence, set it as the main error location
                if results["file_path"] is None:
                    results["file_path"] = file_path
                    results["line_number"] = line_number
                
                # Extract the context line if available
                context_line = None
                if i + 1 < len(lines) and not lines[i + 1].strip().startswith("File "):
                    context_line = lines[i + 1].strip()
                
                # Add to stack trace
                results["stack_trace"].append({
                    "file_path": file_path,
                    "line_number": line_number,
                    "context_line": context_line
                })
        
        # Analyze the error type and suggest potential causes and fixes
        if results["error_type"] == "NameError":
            # Extract the undefined name
            name_match = re.search(r"name '([^']+)' is not defined", results["error_message"])
            if name_match:
                undefined_name = name_match.group(1)
                results["context"]["undefined_name"] = undefined_name
                
                # Suggest potential causes
                results["potential_causes"].extend([
                    f"The variable '{undefined_name}' is used before it's defined",
                    f"There might be a typo in the variable name '{undefined_name}'",
                    f"The module containing '{undefined_name}' might not be imported"
                ])
                
                # Suggest fixes
                results["suggested_fixes"].extend([
                    f"Define the variable '{undefined_name}' before using it",
                    f"Check for typos in the variable name",
                    f"Import the module that contains '{undefined_name}'"
                ])
        
        elif results["error_type"] == "ImportError" or results["error_type"] == "ModuleNotFoundError":
            # Extract the module name
            module_match = re.search(r"No module named '([^']+)'", results["error_message"])
            if module_match:
                module_name = module_match.group(1)
                results["context"]["module_name"] = module_name
                
                # Suggest potential causes
                results["potential_causes"].extend([
                    f"The module '{module_name}' is not installed",
                    f"There might be a typo in the module name '{module_name}'",
                    f"The module '{module_name}' might be in a different location"
                ])
                
                # Suggest fixes
                results["suggested_fixes"].extend([
                    f"Install the module '{module_name}' using pip or conda",
                    f"Check for typos in the module name",
                    f"Add the module's location to the Python path"
                ])
        
        elif results["error_type"] == "TypeError":
            # Extract type information
            type_match = re.search(r"'([^']+)' object is not ([a-z]+)", results["error_message"])
            if type_match:
                obj_type = type_match.group(1)
                expected_behavior = type_match.group(2)
                results["context"]["object_type"] = obj_type
                results["context"]["expected_behavior"] = expected_behavior
                
                # Suggest potential causes
                results["potential_causes"].extend([
                    f"An object of type '{obj_type}' is being used as if it were {expected_behavior}",
                    f"There might be a type conversion missing",
                    f"The API might have changed and the object no longer supports this operation"
                ])
                
                # Suggest fixes
                results["suggested_fixes"].extend([
                    f"Check the type of the object before using it",
                    f"Convert the object to a type that supports the '{expected_behavior}' operation",
                    f"Update the code to use the correct API"
                ])
        
        elif results["error_type"] == "AttributeError":
            # Extract attribute information
            attr_match = re.search(r"'([^']+)' object has no attribute '([^']+)'", results["error_message"])
            if attr_match:
                obj_type = attr_match.group(1)
                attribute = attr_match.group(2)
                results["context"]["object_type"] = obj_type
                results["context"]["attribute"] = attribute
                
                # Suggest potential causes
                results["potential_causes"].extend([
                    f"The object of type '{obj_type}' doesn't have an attribute named '{attribute}'",
                    f"There might be a typo in the attribute name '{attribute}'",
                    f"The API might have changed and the attribute has been renamed or removed"
                ])
                
                # Suggest fixes
                results["suggested_fixes"].extend([
                    f"Check the documentation for the '{obj_type}' class to see available attributes",
                    f"Check for typos in the attribute name",
                    f"Update the code to use the correct API"
                ])
        
        # If we couldn't identify the error type, provide generic suggestions
        if not results["potential_causes"]:
            results["potential_causes"].append("The error message doesn't match any known patterns")
            results["suggested_fixes"].append("Check the error message and stack trace for clues")
        
        # Build a dependency graph if we have file information
        if results["file_path"] and results["line_number"]:
            results["context"]["dependency_graph"] = self._build_dependency_graph(
                results["file_path"], 
                results["line_number"]
            )
    
    except Exception as e:
        results["status"] = "error"
        results["error"] = f"Error trace analysis failed: {str(e)}"
    
    return results

def _build_dependency_graph(self, file_path: str, line_number: int) -> Dict[str, Any]:
    """
    Build a dependency graph for the given file and line number.
    
    Args:
        file_path: Path to the file
        line_number: Line number in the file
        
    Returns:
        Dependency graph information
    """
    graph = {
        "nodes": [],
        "edges": []
    }
    
    try:
        # Read the file
        with open(file_path, 'r') as f:
            code = f.read()
        
        # Parse the code
        tree = ast.parse(code)
        
        # Find the node at the given line number
        target_node = None
        for node in ast.walk(tree):
            if hasattr(node, 'lineno') and node.lineno == line_number:
                target_node = node
                break
        
        if target_node:
            # Add the target node
            node_id = len(graph["nodes"])
            graph["nodes"].append({
                "id": node_id,
                "type": type(target_node).__name__,
                "line": line_number
            })
            
            # Find dependencies
            if isinstance(target_node, ast.Name):
                # Find where this name is defined
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name) and target.id == target_node.id:
                                dep_id = len(graph["nodes"])
                                graph["nodes"].append({
                                    "id": dep_id,
                                    "type": "Assignment",
                                    "line": node.lineno
                                })
                                graph["edges"].append({
                                    "source": node_id,
                                    "target": dep_id,
                                    "type": "defined_by"
                                })
            
            elif isinstance(target_node, ast.Call):
                # Find the function definition
                if isinstance(target_node.func, ast.Name):
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef) and node.name == target_node.func.id:
                            dep_id = len(graph["nodes"])
                            graph["nodes"].append({
                                "id": dep_id,
                                "type": "Function",
                                "name": node.name,
                                "line": node.lineno
                            })
                            graph["edges"].append({
                                "source": node_id,
                                "target": dep_id,
                                "type": "calls"
                            })
    
    except Exception as e:
        graph["error"] = str(e)
    
    return graph
```
