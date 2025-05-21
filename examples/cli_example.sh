#!/bin/bash
# Example of using the Structural Intent Analysis CLI

# Set up the environment
echo "Setting up the environment..."
cd "$(dirname "$0")/.."

# Show version information
echo "Showing version information..."
python -m semantic_matrix_analyzer.cli version

# Analyze a simple Python file
echo -e "\nAnalyzing a simple Python file..."
python -m semantic_matrix_analyzer.cli intent examples/intent_analysis_example.py --python --format markdown

# Analyze a directory with minimal configuration
echo -e "\nAnalyzing a directory with minimal configuration..."
python -m semantic_matrix_analyzer.cli intent examples --minimal --no-structure --format text

# Analyze a directory with comprehensive configuration
echo -e "\nAnalyzing a directory with comprehensive configuration..."
python -m semantic_matrix_analyzer.cli intent examples --comprehensive --format json --output intent_analysis_report.json

# Show the report
if [ -f intent_analysis_report.json ]; then
    echo -e "\nReport saved to intent_analysis_report.json"
    echo "First 10 lines of the report:"
    head -n 10 intent_analysis_report.json
    rm intent_analysis_report.json
fi

echo -e "\nDone!"
