"""
Fix contracts.json and __init__.py to use proper CamelCase module names.
This reads the Python files to get actual function names and builds correct mapping.
"""

import json
import re
from pathlib import Path


def get_function_name_from_file(py_file: Path) -> str:
    """Extract the main function name from a Python file."""
    content = py_file.read_text()
    match = re.search("@numpy.*?^def ([a-z_]+)\\(", content, re.MULTILINE | re.DOTALL)
    if match:
        func_name = match.group(1)
        if not func_name.startswith("_"):
            return func_name
    matches = re.findall("^def ([a-z_]+)\\(", content, re.MULTILINE)
    for func_name in matches:
        if not func_name.startswith("_"):
            return func_name
    return None


def snake_to_camel(snake_str: str) -> str:
    """Convert snake_case to CamelCase."""
    parts = snake_str.split("_")
    return "".join((word.capitalize() for word in parts))


def fix_contracts_json():
    """Fix contracts.json to use proper CamelCase keys based on actual function names."""
    contracts_file = Path("benchmark/cellprofiler_library/contracts.json")
    functions_dir = Path("benchmark/cellprofiler_library/functions")
    old_data = json.loads(contracts_file.read_text())
    fixed_data = {}
    for py_file in sorted(functions_dir.glob("*.py")):
        if py_file.name == "__init__.py":
            continue
        func_name = get_function_name_from_file(py_file)
        if not func_name:
            print(f"⚠️  Could not find function in {py_file.name}")
            continue
        module_name = snake_to_camel(func_name)
        old_entry = None
        for old_key, value in old_data.items():
            if old_key.lower() == module_name.lower():
                old_entry = value
                break
        if old_entry:
            old_entry["function_name"] = func_name
            fixed_data[module_name] = old_entry
        else:
            fixed_data[module_name] = {
                "function_name": func_name,
                "confidence": 0.5,
                "reasoning": "Auto-generated from existing function",
                "validated": True,
            }
            print(f"⚠️  Created new entry for {module_name} ({func_name})")
    contracts_file.write_text(json.dumps(fixed_data, indent=2))
    print(f"\n✅ Fixed {len(fixed_data)} entries in contracts.json")
    print("\nExample entries:")
    examples = list(fixed_data.items())[:5]
    for module_name, info in examples:
        print(f"  {module_name}: {info['function_name']}")


def fix_init_py():
    """Fix __init__.py to use proper CamelCase keys in CELLPROFILER_MODULES dict."""
    init_file = Path("benchmark/cellprofiler_library/__init__.py")
    contracts_file = Path("benchmark/cellprofiler_library/contracts.json")
    data = json.loads(contracts_file.read_text())
    content = init_file.read_text()
    import re

    lines = []
    lines.append("# Registry mapping CellProfiler module names to OpenHCS functions")
    lines.append("CELLPROFILER_MODULES: Dict[str, Callable] = {")
    for module_name, info in sorted(data.items()):
        func_name = info["function_name"]
        lines.append(f'    "{module_name}": {func_name},')
    lines.append("}")
    new_registry = "\n".join(lines)
    pattern = "# Registry mapping.*?^}"
    content = re.sub(pattern, new_registry, content, flags=re.MULTILINE | re.DOTALL)
    init_file.write_text(content)
    print(f"\n✅ Fixed CELLPROFILER_MODULES dict in __init__.py")


if __name__ == "__main__":
    fix_contracts_json()
    fix_init_py()
    print("\n✅ All fixed! Now you can run the converter.")
