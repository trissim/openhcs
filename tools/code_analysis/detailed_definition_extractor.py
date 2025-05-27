import ast
import sys
import os
import json

class DetailedDefinitionExtractor(ast.NodeVisitor):
    def __init__(self):
        self.definitions = []
        self.current_class = None

    def visit_FunctionDef(self, node):
        self._extract_definition_info(node, "function")
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self._extract_definition_info(node, "function")
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        class_info = {
            "type": "class",
            "name": node.name,
            "start_line": node.lineno,
            "end_line": node.end_lineno,
            "definitions": [] # Nested definitions within the class
        }
        self.definitions.append(class_info)
        
        old_current_class = self.current_class
        self.current_class = class_info # Set current class for nested methods
        self.generic_visit(node)
        self.current_class = old_current_class # Restore previous current class

    def _extract_definition_info(self, node, def_type):
        name = node.name
        parameters = []
        for arg in node.args.args:
            param_name = arg.arg
            param_type = None
            if arg.annotation:
                param_type = self._get_annotation_value(arg.annotation)
            parameters.append({"name": param_name, "type": param_type})

        return_type = None
        if node.returns:
            return_type = self._get_annotation_value(node.returns)

        definition_entry = {
            "type": def_type,
            "name": name,
            "parameters": parameters,
            "return_type": return_type,
            "start_line": node.lineno,
            "end_line": node.end_lineno
        }

        if self.current_class:
            definition_entry["type"] = "method"
            definition_entry["parent_class"] = self.current_class["name"]
            self.current_class["definitions"].append(definition_entry)
        else:
            self.definitions.append(definition_entry)

    def _get_annotation_value(self, node):
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_annotation_value(node.value)}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            value = self._get_annotation_value(node.value)
            slice_value = self._get_annotation_value(node.slice)
            return f"{value}[{slice_value}]"
        elif isinstance(node, ast.Tuple):
            return ", ".join([self._get_annotation_value(el) for el in node.elts])
        else:
            return "<complex_annotation>"

def extract_detailed_definitions_from_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=filepath)

        extractor = DetailedDefinitionExtractor()
        extractor.visit(tree)
        return extractor.definitions
    except SyntaxError as e:
        return [{"type": "error", "name": "SyntaxError", "message": str(e), "line": e.lineno}]
    except Exception as e:
        return [{"type": "error", "name": type(e).__name__, "message": str(e)}]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detailed_definition_extractor.py <filepath>")
        sys.exit(1)

    filepath = sys.argv[1]
    if not os.path.exists(filepath):
        print(json.dumps([{"type": "error", "name": "FileNotFound", "message": f"File not found: {filepath}"}]))
        sys.exit(1)

    definitions = extract_detailed_definitions_from_file(filepath)
    print(json.dumps(definitions, indent=2))