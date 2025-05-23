# Code Analysis Tools - Technical Documentation

## Overview

This directory contains a suite of Python-based code analysis tools designed to provide comprehensive insights into any Python codebase.

## Tools Available

### 1. `code_analyzer_cli.py` - Main CLI Interface

**Purpose**: Unified command-line interface for all analysis operations.

**Commands**:
- `snapshot` - Generate high-level codebase overview
- `matrix` - Create detailed analysis of specific files
- `dependencies` - Map module relationships

**Usage Examples**:
```bash
# Analyze entire project
python code_analyzer_cli.py snapshot --target src/ -o output.csv

# Detailed analysis of specific files
python code_analyzer_cli.py matrix file1.py file2.py -o output.md

# Map dependencies for a directory
python code_analyzer_cli.py dependencies src/ -o output.md
```

### 2. `extract_definitions.py` - Summary Extractor

**Purpose**: Extract high-level metrics from Python files.

**Output**: JSON with counts of classes, functions, methods, and type annotations.

**Direct Usage**:
```bash
python extract_definitions.py path/to/file.py
```

### 3. `detailed_definition_extractor.py` - Deep Analysis

**Purpose**: Extract detailed information about all code definitions.

**Output**: JSON with complete function signatures, parameters, return types, and line numbers.

**Direct Usage**:
```bash
python detailed_definition_extractor.py path/to/file.py
```

### 4. `module_dependency_analyzer.py` - Dependency Mapper

**Purpose**: Analyze import relationships between modules.

**Output**: List of local dependencies for each file.

### 5. `call_graph_analyzer.py` - Call Graph Analysis

**Purpose**: Analyze function call relationships in Python code.

**Features**:
- Identifies entry points (functions not called by any other function)
- Identifies leaf functions (functions that don't call any other function)
- Detects circular dependencies
- Lists most called functions and functions with most calls

**Usage**:
```bash
python call_graph_analyzer.py path/to/analyze -o output.md
```

### 6. `semantic_role_analyzer.py` - Semantic Role Analysis

**Purpose**: Analyze the semantic roles of functions in Python code.

**Features**:
- Classifies functions as GETTER, SETTER, FACTORY, MUTATOR, VALIDATOR, INITIALIZER, FINALIZER, EVENT_HANDLER, or UTILITY
- Identifies state mutations in functions
- Provides insights into code organization and design patterns

**Usage**:
```bash
python semantic_role_analyzer.py path/to/analyze -o output.md
```

### 7. `interface_classifier.py` - Interface Analysis

**Purpose**: Analyze and classify interfaces in Python code.

**Features**:
- Identifies abstract base classes and interfaces
- Detects duck typing patterns
- Analyzes interface coverage and implementation
- Identifies protocol adherence

**Usage**:
```bash
python interface_classifier.py path/to/analyze -o output.md
```

### 8. `meta_analyzer.py` - Meta Analysis

**Purpose**: Run multiple analysis tools in sequence.

**Commands**:
- `comprehensive` - Run all analysis tools
- `architecture` - Run call graph, interface, and dependency analysis
- `semantics` - Run semantic role analysis
- `imports` - Run import analysis and optionally fix issues
- `quality` - Run code quality analysis

**Usage**:
```bash
# Run comprehensive analysis
python meta_analyzer.py comprehensive path/to/analyze

# Run architecture analysis
python meta_analyzer.py architecture path/to/analyze

# Run imports analysis and fix issues
python meta_analyzer.py imports path/to/analyze --fix
```

## Output Formats

### CSV Files (Quantitative Data)
- File path, class count, function count, method count
- Top function names, parameter types, return types
- Suitable for data analysis and metrics

### Markdown Files (Human-Readable)
- Detailed tables with function signatures
- Dependency graphs
- Error reporting
- Suitable for documentation and review

### JSON (Raw Data)
- Complete structured data
- Suitable for programmatic processing

## Error Handling

All tools gracefully handle:
- Syntax errors in Python files
- Missing files or directories
- Import resolution failures
- Encoding issues

Errors are reported in the output without stopping analysis of other files.

## Integration with AI Agents

These tools are designed to be easily discoverable and usable by AI agents:

1. **Self-documenting**: Clear help text and examples
2. **Predictable outputs**: Consistent file naming and locations
3. **Error resilient**: Won't crash on problematic files
4. **Modular**: Each tool can be used independently

## Extending the Tools

To add new analysis capabilities:

1. Create a new extractor class inheriting from `ast.NodeVisitor`
2. Add the extraction logic to `code_analyzer_cli.py`
3. Update the argument parser with new commands
4. Follow the existing error handling patterns

## Dependencies

- Python 3.6+
- Standard library (ast, os, sys, json, argparse)
- NetworkX (for call graph visualization)
- Matplotlib (for graph visualization)

```bash
pip install networkx matplotlib
```