import argparse
import os
import json
import sys
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Add the parent directory of 'tools' to sys.path to allow direct imports
# This assumes the CLI is in tools/code_analysis and helpers are in the same dir
# and the project root is two levels up.
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT) # Allows `from tools.code_analysis import ...`

from tools.code_analysis.extract_definitions import extract_summary_from_file
from tools.code_analysis.detailed_definition_extractor import extract_detailed_definitions_from_file
from tools.code_analysis.module_dependency_analyzer import analyze_dependencies

REPORTS_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "reports", "code_analysis"))
os.makedirs(REPORTS_DIR, exist_ok=True)

# --- CLI Subcommand Implementations ---

def generate_snapshot_logic(project_root_path, target_dir_path):
    """Core logic for snapshot generation - works with any Python directory."""
    header = "File Path,Classes,Functions,Methods,Total Definitions,Top Names,Top Param Types,Top Return Types"
    lines = [header]
    for root, _, files in os.walk(target_dir_path):
        for file_name in files: # Renamed 'file' to 'file_name' to avoid conflict
            if file_name.endswith(".py") and file_name != "__init__.py":
                filepath = os.path.join(root, file_name)
                summary = extract_summary_from_file(filepath)

                # Check if summary indicates an error (e.g., from SyntaxError)
                is_error = False
                if "top_names" in summary and summary["top_names"] and \
                   (summary["top_names"][0].startswith("SyntaxError") or summary["top_names"][0].startswith("Error:")):
                    is_error = True
                    error_message = summary["top_names"][0]

                if is_error:
                    lines.append(f"{os.path.relpath(filepath, project_root_path)},0,0,0,0,\"{error_message}\",\"\",\"\"")
                    continue

                classes_count = summary.get("classes_count", 0)
                functions_count = summary.get("functions_count", 0)
                methods_count = summary.get("methods_count", 0)
                total_definitions = summary.get("total_definitions", 0)
                top_names = ", ".join(summary.get("top_names", []))
                top_param_types = ", ".join(summary.get("top_param_types", []))
                top_return_types = ", ".join(summary.get("top_return_types", []))
                lines.append(f"{os.path.relpath(filepath, project_root_path)},{classes_count},{functions_count},{methods_count},{total_definitions},\"{top_names}\",\"{top_param_types}\",\"{top_return_types}\"")
    return "\n".join(lines)

def handle_snapshot(args):
    target_dir = getattr(args, 'target', None) or os.path.join(PROJECT_ROOT, "openhcs")
    target_dir_abs = os.path.abspath(target_dir)

    if not os.path.isdir(target_dir_abs):
        print(f"Error: Target directory not found at {target_dir_abs}", file=sys.stderr)
        return

    print(f"Generating high-level codebase snapshot for: {target_dir_abs}")

    # Generate output filename based on target directory
    if args.output:
        output_file = args.output
    else:
        dir_name = os.path.basename(target_dir_abs.rstrip(os.sep)) or "root"
        output_file = os.path.join(REPORTS_DIR, f"{dir_name}_codebase_snapshot.csv")

    csv_data = generate_snapshot_logic(PROJECT_ROOT, target_dir_abs)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(csv_data)
    print(f"High-level codebase snapshot generated: {output_file}")

def format_detailed_matrix_as_markdown(filepath, definitions):
    table = f"### Detailed Matrix for `{filepath}`\n\n"
    if not definitions:
        return table + "No definitions found or error in parsing.\n"

    has_errors = any(d.get("type") == "error" for d in definitions)
    if has_errors:
        table += "**Errors encountered during parsing:**\n"
        for d in definitions:
            if d.get("type") == "error":
                table += f"- {d.get('name')}: {d.get('message')} (line {d.get('line', 'N/A')})\n"
        table += "\n"
        # Optionally, one might choose to not print the table if errors are severe
        # For now, we'll print any valid definitions found alongside errors.

    headers = ["Definition Type", "Name", "Parent Class", "Parameters", "Return Type", "Lines"]
    table += "| " + " | ".join(headers) + " |\n"
    table += "| " + " | ".join(["---"] * len(headers)) + " |\n"

    def process_definition_for_table(defn, file_path_for_error_context=""):
        # Helper to format a single definition (or nested method)
        params_list = []
        for p in defn.get("parameters", []):
            p_type = p.get('type', 'Any') if p.get('type') is not None else 'Any' # Default to Any if type is None
            params_list.append(f"{p['name']}: {p_type}")

        return [
            str(defn.get("type", "unknown")).replace("|", "\\|"),
            str(defn.get("name", "N/A")).replace("|", "\\|"),
            str(defn.get("parent_class", "")).replace("|", "\\|"),
            str(", ".join(params_list)).replace("|", "\\|"),
            str(defn.get("return_type", "") if defn.get("return_type") is not None else "").replace("|", "\\|"),
            str(f"{defn.get('start_line', 'N/A')}-{defn.get('end_line', 'N/A')}").replace("|", "\\|")
        ]

    for defn in definitions:
        if defn.get("type") == "error": # Already handled above
            continue
        table += "| " + " | ".join(process_definition_for_table(defn)) + " |\n"
        if defn.get("type") == "class" and "definitions" in defn: # Nested methods
            for nested_def in defn["definitions"]:
                table += "| " + " | ".join(process_definition_for_table(nested_def)) + " |\n"
    return table + "\n"

def handle_detailed_matrix(args):
    print(f"Generating detailed definition matrix for: {', '.join(args.filepaths)}...")
    report_content = f"# Detailed Code Definition Matrix\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    for filepath in args.filepaths:
        if not os.path.exists(filepath):
            report_content += f"### Error for `{filepath}`\nFile not found.\n\n"
            continue
        definitions = extract_detailed_definitions_from_file(filepath)
        report_content += format_detailed_matrix_as_markdown(filepath, definitions)

    if args.output:
        output_filename = args.output
    elif len(args.filepaths) == 1:
        base_name = os.path.basename(args.filepaths[0])
        file_name_without_ext = os.path.splitext(base_name)[0]
        output_filename = os.path.join(REPORTS_DIR, f"detailed_code_matrix_{file_name_without_ext}.md")
    else:
        # For multiple files, use the first letter of each file, snake_cased
        initials = "_".join([os.path.basename(fp)[0] for fp in args.filepaths if fp])
        output_filename = os.path.join(REPORTS_DIR, f"detailed_code_matrix_{initials}.md")

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Detailed code matrix report generated: {output_filename}")


def format_dependencies_as_markdown(directory_path, dependencies_map):
    markdown_output = f"# Module Dependency Graph for `{directory_path}`\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    for file, deps in sorted(dependencies_map.items()):
        markdown_output += f"## `{file}`\n"
        if any("Error:" in d or "SyntaxError:" in d for d in deps):
             markdown_output += "### Errors encountered:\n"
             for dep in deps:
                 if "Error:" in dep or "SyntaxError:" in dep:
                     markdown_output += f"- {dep}\n"
        elif deps:
            markdown_output += "### Depends on:\n"
            for dep in deps:
                markdown_output += f"- `{dep}`\n"
        else:
            markdown_output += "### No local dependencies found or file has issues.\n"
        markdown_output += "\n"

    return markdown_output

def handle_dependencies(args):
    print(f"Generating module dependency graph for directory: {args.directory}...")

    target_dir_abs = os.path.abspath(args.directory)
    if not os.path.isdir(target_dir_abs):
        print(f"Error: Directory not found at {target_dir_abs}", file=sys.stderr)
        return

    dependencies_map = {}
    python_files = []
    for root, _, files in os.walk(target_dir_abs):
        for file_name in files: # Renamed 'file' to 'file_name'
            if file_name.endswith(".py"):
                python_files.append(os.path.join(root, file_name))

    for filepath in python_files:
        # analyze_dependencies expects base_path to be the project root for resolving 'openhcs.*'
        deps = analyze_dependencies(filepath, PROJECT_ROOT)
        relative_filepath = os.path.relpath(filepath, PROJECT_ROOT)
        dependencies_map[relative_filepath] = deps

    markdown_output = format_dependencies_as_markdown(args.directory, dependencies_map)

    # Ensure basename is not empty if args.directory ends with a slash
    dir_name = os.path.basename(args.directory.rstrip(os.sep)) or "root"
    output_filename = args.output or os.path.join(REPORTS_DIR, f"module_dependency_graph_{dir_name}.md")
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(markdown_output)
    print(f"Module dependency graph generated: {output_filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Generic Code Analysis CLI. Provides tools to generate snapshots, detailed matrices, and dependency graphs for any Python codebase.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands. Use <command> -h for more details.")
    subparsers.required = True

    # Snapshot command
    parser_snapshot = subparsers.add_parser("snapshot", help="Generate a high-level overview (CSV) of any Python codebase.")
    parser_snapshot.add_argument("-t", "--target", help="Target directory to analyze (defaults to 'openhcs' if it exists, otherwise current directory)")
    parser_snapshot.add_argument("-o", "--output", help="Output CSV file path. Defaults to reports/code_analysis/<dirname>_codebase_snapshot.csv")
    parser_snapshot.set_defaults(func=handle_snapshot)

    # Detailed Matrix command
    parser_detailed = subparsers.add_parser("matrix", help="Generate a detailed definition matrix (Markdown) for specified Python files.")
    parser_detailed.add_argument("filepaths", nargs="+", help="Paths to Python files to analyze.")
    parser_detailed.add_argument("-o", "--output", help="Output Markdown file path. Defaults to reports/code_analysis/detailed_code_matrix_report.md")
    parser_detailed.set_defaults(func=handle_detailed_matrix)

    # Dependencies command
    parser_deps = subparsers.add_parser("dependencies", help="Generate a module dependency graph (Markdown) for a specified directory.")
    parser_deps.add_argument("directory", help="Path to the directory to analyze (e.g., openhcs/tui).")
    parser_deps.add_argument("-o", "--output", help="Output Markdown file path. Defaults to reports/code_analysis/module_dependency_graph_<dirname>.md")
    parser_deps.set_defaults(func=handle_dependencies)


    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()