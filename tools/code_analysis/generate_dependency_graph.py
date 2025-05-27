import json
import subprocess
import sys
import os

def get_python_files_in_directory(directory):
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    return python_files

def generate_dependency_graph(directory):
    all_dependencies = {}
    python_files = get_python_files_in_directory(directory)
    
    for filepath in python_files:
        try:
            # Pass "." (current working directory, i.e., project root) as the base_path
            command = ["python", "tools/code_analysis/module_dependency_analyzer.py", filepath, "."]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            output = json.loads(result.stdout)
            
            relative_filepath = os.path.relpath(filepath, directory)
            all_dependencies[relative_filepath] = output.get("dependencies", [])
        except subprocess.CalledProcessError as e:
            relative_filepath = os.path.relpath(filepath, directory)
            all_dependencies[relative_filepath] = [f"Error: {e.stderr.strip()}"]
        except json.JSONDecodeError as e:
            relative_filepath = os.path.relpath(filepath, directory)
            all_dependencies[relative_filepath] = [f"JSON Parse Error: {e}"]
        except Exception as e:
            relative_filepath = os.path.relpath(filepath, directory)
            all_dependencies[relative_filepath] = [f"Unexpected Error: {type(e).__name__}: {e}"]
            
    return all_dependencies

def format_as_markdown(dependencies_map):
    markdown_output = "# Module Dependency Graph\n\n"
    
    for file, deps in sorted(dependencies_map.items()):
        markdown_output += f"## `{file}`\n"
        if deps:
            markdown_output += "### Depends on:\n"
            for dep in deps:
                markdown_output += f"- `{dep}`\n"
        else:
            markdown_output += "### No local dependencies found.\n"
        markdown_output += "\n"
            
    return markdown_output

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_dependency_graph.py <directory_path>")
        sys.exit(1)

    directory_path = sys.argv[1]
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found at {directory_path}")
        sys.exit(1)

    dependencies_map = generate_dependency_graph(directory_path)
    markdown_output = format_as_markdown(dependencies_map)
    
    output_filename = "reports/code_analysis/module_dependency_graph.md"
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(markdown_output)
    print(f"Module dependency graph generated and saved to {output_filename}")