import json
import subprocess
import sys
import os

def generate_matrix(filepaths):
    matrix_data = []
    for filepath in filepaths:
        try:
            # Execute detailed_definition_extractor.py for each file
            command = ["python", "tools/code_analysis/detailed_definition_extractor.py", filepath]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            definitions = json.loads(result.stdout)

            for definition in definitions:
                row = {
                    "File Path": filepath,
                    "Definition Type": definition.get("type", "unknown"),
                    "Name": definition.get("name", "N/A"),
                    "Parent Class": definition.get("parent_class", ""),
                    "Parameters": ", ".join([f"{p['name']}: {p['type']}" for p in definition.get("parameters", []) if p.get("type")]) if definition.get("parameters") else "",
                    "Return Type": definition.get("return_type", ""),
                    "Lines": f"{definition.get('start_line', 'N/A')}-{definition.get('end_line', 'N/A')}"
                }
                matrix_data.append(row)
                
                # Handle nested definitions for classes
                if definition.get("type") == "class" and "definitions" in definition:
                    for nested_def in definition["definitions"]:
                        nested_row = {
                            "File Path": filepath,
                            "Definition Type": nested_def.get("type", "unknown"),
                            "Name": nested_def.get("name", "N/A"),
                            "Parent Class": nested_def.get("parent_class", ""),
                            "Parameters": ", ".join([f"{p['name']}: {p['type']}" for p in nested_def.get("parameters", []) if p.get("type")]) if nested_def.get("parameters") else "",
                            "Return Type": nested_def.get("return_type", ""),
                            "Lines": f"{nested_def.get('start_line', 'N/A')}-{nested_def.get('end_line', 'N/A')}"
                        }
                        matrix_data.append(nested_row)

        except subprocess.CalledProcessError as e:
            matrix_data.append({
                "File Path": filepath,
                "Definition Type": "error",
                "Name": "Extraction Failed",
                "Parent Class": "",
                "Parameters": "",
                "Return Type": "",
                "Lines": f"Error: {e.stderr.strip()}"
            })
        except json.JSONDecodeError as e:
            matrix_data.append({
                "File Path": filepath,
                "Definition Type": "error",
                "Name": "JSON Parse Error",
                "Parent Class": "",
                "Parameters": "",
                "Return Type": "",
                "Lines": f"Error: {e}"
            })
        except Exception as e:
            matrix_data.append({
                "File Path": filepath,
                "Definition Type": "error",
                "Name": "Unexpected Error",
                "Parent Class": "",
                "Parameters": "",
                "Return Type": "",
                "Lines": f"Error: {type(e).__name__}: {e}"
            })
    return matrix_data

def format_as_markdown_table(data):
    if not data:
        return "No definitions found."

    headers = ["File Path", "Definition Type", "Name", "Parent Class", "Parameters", "Return Type", "Lines"]
    
    # Ensure all rows have all headers, fill missing with empty string
    for row in data:
        for header in headers:
            if header not in row:
                row[header] = ""

    # Create header row
    markdown_table = "| " + " | ".join(headers) + " |\n"
    # Create separator row
    markdown_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"

    # Create data rows
    for row in data:
        values = [str(row.get(header, "")).replace("|", "\\|") for header in headers]
        markdown_table += "| " + " | ".join(values) + " |\n"
    
    return markdown_table

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_detailed_matrix.py <filepath1> [<filepath2> ...]")
        sys.exit(1)

    filepaths = sys.argv[1:]
    matrix_data = generate_matrix(filepaths)
    markdown_output = format_as_markdown_table(matrix_data)
    
    # Write to a default file or print to stdout
    output_filename = "reports/code_analysis/detailed_code_matrix.md"
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(markdown_output)
    print(f"Detailed code matrix generated and saved to {output_filename}")