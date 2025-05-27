import ast
import sys
import os
import json
from collections import Counter

class DefinitionExtractor(ast.NodeVisitor):
    def __init__(self):
        self.classes = []
        self.functions = []
        self.methods = []
        self.param_types = []
        self.return_types = []
        self.current_class_name = None

    def visit_FunctionDef(self, node):
        self._extract_function_info(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self._extract_function_info(node)
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.classes.append(node.name)
        old_class_name = self.current_class_name
        self.current_class_name = node.name
        self.generic_visit(node)
        self.current_class_name = old_class_name

    def _extract_function_info(self, node):
        name = node.name
        if self.current_class_name:
            self.methods.append(name)
        else:
            self.functions.append(name)

        for arg in node.args.args:
            if arg.annotation:
                self.param_types.append(self._get_annotation_value(arg.annotation))

        if node.returns:
            self.return_types.append(self._get_annotation_value(node.returns))

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

    def get_summary(self):
        all_names = self.functions + self.methods
        top_names = [name for name, _ in Counter(all_names).most_common(5)]
        top_param_types = [t for t, _ in Counter(self.param_types).most_common(5)]
        top_return_types = [t for t, _ in Counter(self.return_types).most_common(5)]

        return {
            "classes_count": len(self.classes),
            "functions_count": len(self.functions),
            "methods_count": len(self.methods),
            "total_definitions": len(self.classes) + len(self.functions) + len(self.methods),
            "top_names": top_names,
            "top_param_types": top_param_types,
            "top_return_types": top_return_types
        }

def extract_summary_from_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=filepath)

        extractor = DefinitionExtractor()
        extractor.visit(tree)
        return extractor.get_summary()
    except SyntaxError as e:
        # Return a default summary for files with syntax errors
        return {
            "classes_count": 0,
            "functions_count": 0,
            "methods_count": 0,
            "total_definitions": 0,
            "top_names": ["SyntaxError"],
            "top_param_types": [],
            "top_return_types": []
        }
    except Exception as e:
        # Catch any other exceptions during parsing or extraction
        return {
            "classes_count": 0,
            "functions_count": 0,
            "methods_count": 0,
            "total_definitions": 0,
            "top_names": [f"Error: {type(e).__name__}"],
            "top_param_types": [],
            "top_return_types": []
        }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_definitions.py <filepath>")
        sys.exit(1)

    filepath = sys.argv[1]
    if not os.path.exists(filepath):
        sys.exit(1) # Exit silently if file not found, as we'll handle this externally

    summary = extract_summary_from_file(filepath)
    print(json.dumps(summary))