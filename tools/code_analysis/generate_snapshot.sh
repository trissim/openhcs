#!/bin/bash

output_file="reports/code_analysis/openhcs_codebase_snapshot.txt"
mkdir -p "$(dirname "$output_file")"
echo "File Path,Classes,Functions,Methods,Total Definitions,Top Names,Top Param Types,Top Return Types" > "$output_file"

find openhcs/ -name "*.py" | while read -r filepath; do
    filename=$(basename "$filepath")
    if [[ "$filename" == "__init__.py" ]]; then
        continue
    fi

    summary_json=$(python tools/code_analysis/extract_definitions.py "$filepath")
    
    # Check if summary_json is empty or invalid JSON
    if [[ -z "$summary_json" || "$summary_json" == "null" ]]; then
        classes_count=0
        functions_count=0
        methods_count=0
        total_definitions=0
        top_names=""
        top_param_types=""
        top_return_types=""
    else
        classes_count=$(echo "$summary_json" | jq -r '.classes_count // 0')
        functions_count=$(echo "$summary_json" | jq -r '.functions_count // 0')
        methods_count=$(echo "$summary_json" | jq -r '.methods_count // 0')
        total_definitions=$(echo "$summary_json" | jq -r '.total_definitions // 0')
        top_names=$(echo "$summary_json" | jq -r '.top_names | join(", ")')
        top_param_types=$(echo "$summary_json" | jq -r '.top_param_types | join(", ")')
        top_return_types=$(echo "$summary_json" | jq -r '.top_return_types | join(", ")')
    fi

    echo "$filepath,$classes_count,$functions_count,$methods_count,$total_definitions,\"$top_names\",\"$top_param_types\",\"$top_return_types\"" >> "$output_file"
done

echo "Codebase snapshot generated: $output_file"