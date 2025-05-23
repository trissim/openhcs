#!/bin/bash

# Generic Code Analysis Quick Start
# Usage: ./tools/quick_start.sh [target_directory]

TARGET_DIR=${1:-"openhcs"}

if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory '$TARGET_DIR' not found."
    echo "Usage: $0 [target_directory]"
    echo "Example: $0 src"
    echo "Example: $0 my_project"
    exit 1
fi

echo "🔍 Running comprehensive code analysis on: $TARGET_DIR"
echo "📊 This will generate:"
echo "  - High-level codebase snapshot (CSV)"
echo "  - Detailed code matrix (Markdown)"
echo "  - Module dependency graph (Markdown)"
echo ""

# Create reports directory
mkdir -p reports/code_analysis

# Get directory name for output files
DIR_NAME=$(basename "$TARGET_DIR")

echo "1/3 📈 Generating codebase snapshot..."
python tools/code_analysis/code_analyzer_cli.py snapshot --target "$TARGET_DIR"

echo "2/3 🔬 Generating detailed matrix..."
python tools/code_analysis/code_analyzer_cli.py matrix "$TARGET_DIR"/*.py -o "reports/code_analysis/${DIR_NAME}_detailed_matrix.md" 2>/dev/null || echo "   (No Python files in root of $TARGET_DIR, skipping detailed matrix)"

echo "3/3 🕸️  Generating dependency graph..."
python tools/code_analysis/code_analyzer_cli.py dependencies "$TARGET_DIR"

echo ""
echo "✅ Analysis complete! Reports generated in reports/code_analysis/"
echo "📁 Files created:"
echo "   - ${DIR_NAME}_codebase_snapshot.csv"
echo "   - ${DIR_NAME}_detailed_matrix.md (if applicable)"
echo "   - module_dependency_graph_${DIR_NAME}.md"
echo ""
echo "🎯 Next steps:"
echo "   - Review the CSV for quantitative metrics"
echo "   - Read the Markdown files for insights"
echo "   - Use this data to make informed architectural decisions"