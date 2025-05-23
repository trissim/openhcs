import ast
import sys
import os
import json

class ImportAnalyzer(ast.NodeVisitor):
    def __init__(self, base_path):
        self.base_path = os.path.abspath(base_path)
        self.local_dependencies = set()

    def visit_Import(self, node):
        for alias in node.names:
            self._check_module(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            # Resolve relative imports to absolute module paths
            if node.level > 0:
                # Calculate the absolute path of the current module's directory
                current_module_dir = os.path.dirname(self.current_file_path)
                
                # Construct the path based on relative levels
                # For example, '..module' from 'openhcs/sub/file.py'
                # node.level = 2 (for '..')
                # os.path.join(current_module_dir, '..', 'module')
                resolved_path_parts = [current_module_dir]
                for _ in range(node.level - 1):
                    resolved_path_parts.append("..")
                if node.module:
                    resolved_path_parts.append(node.module.replace('.', os.sep))
                
                resolved_full_path = os.path.abspath(os.path.join(*resolved_path_parts))
                
                # Convert the absolute path back to a module name relative to the base_path
                # This assumes base_path is the root of the project where 'openhcs' resides
                if resolved_full_path.startswith(self.base_path):
                    relative_to_base = os.path.relpath(resolved_full_path, self.base_path)
                    module_name = relative_to_base.replace(os.sep, '.')
                    # If it's a directory, it's a package, so append .__init__ to check
                    if os.path.isdir(resolved_full_path) and not module_name.endswith('__init__'):
                        module_name = f"{module_name}.__init__"
                else:
                    # If it resolves outside base_path, it's not a local dependency we care about
                    module_name = None # Skip this import
            else:
                # Absolute import (e.g., 'openhcs.core.pipeline')
                module_name = node.module
            
            if module_name:
                self._check_module(module_name)
        self.generic_visit(node)

    def _check_module(self, module_name):
        # Only consider modules that start with 'openhcs' as local dependencies
        if not module_name.startswith("openhcs"):
            return

        # Convert module name (e.g., 'openhcs.tui.tui_launcher')
        # to a file path relative to the project root (e.g., 'openhcs/tui/tui_launcher.py')
        module_path_parts = module_name.split('.')
        
        # Construct potential file paths
        potential_file_path = os.path.join(self.base_path, *module_path_parts)
        
        # Check if it's a package (directory with __init__.py)
        if os.path.exists(os.path.join(potential_file_path, "__init__.py")):
            self.local_dependencies.add(module_name)
            return

        # Check if it's a module (single .py file)
        if os.path.exists(potential_file_path + ".py"):
            self.local_dependencies.add(module_name)
            return

def analyze_dependencies(filepath, base_path):
    analyzer = ImportAnalyzer(base_path)
    analyzer.current_file_path = filepath # Store current file path for relative imports

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=filepath)
        analyzer.visit(tree)
        return list(analyzer.local_dependencies)
    except SyntaxError as e:
        return [f"SyntaxError: {e.msg} at line {e.lineno}"]
    except Exception as e:
        return [f"Error: {type(e).__name__}: {e}"]

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python module_dependency_analyzer.py <filepath> <base_path>")
        sys.exit(1)

    filepath = sys.argv[1]
    base_path = sys.argv[2]

    if not os.path.exists(filepath):
        print(json.dumps({"file": filepath, "dependencies": [f"FileNotFound: {filepath}"]}))
        sys.exit(1)

    dependencies = analyze_dependencies(filepath, base_path)
    print(json.dumps({"file": filepath, "dependencies": dependencies}, indent=2))