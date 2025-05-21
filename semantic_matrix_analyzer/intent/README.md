# Structural Intent Analysis

The Structural Intent Analysis system is a tool for extracting intent from code structure. It analyzes method/class/variable names, type hints, and structural patterns to provide insights about code intent.

## Installation

```bash
# Install from source
git clone https://github.com/yourusername/semantic_matrix_analyzer.git
cd semantic_matrix_analyzer
pip install -e .
```

## Usage

### Command-Line Interface

The Structural Intent Analysis system provides a command-line interface (CLI) for analyzing codebases:

```bash
# Show help
sma-intent --help

# Analyze a Python file
sma-intent path/to/file.py --python

# Analyze a Java file
sma-intent path/to/file.java --java

# Analyze a directory
sma-intent path/to/directory --python

# Use a specific configuration
sma-intent path/to/directory --config path/to/config.json

# Use a predefined configuration
sma-intent path/to/directory --microservices
sma-intent path/to/directory --minimal
sma-intent path/to/directory --comprehensive

# Disable specific analyzers
sma-intent path/to/directory --no-names
sma-intent path/to/directory --no-types
sma-intent path/to/directory --no-structure

# Set confidence threshold
sma-intent path/to/directory --min-confidence 0.5

# Limit results
sma-intent path/to/directory --max-results 50

# Save report to file
sma-intent path/to/directory --output report.txt

# Change output format
sma-intent path/to/directory --format text
sma-intent path/to/directory --format markdown
sma-intent path/to/directory --format json

# Enable verbose logging
sma-intent path/to/directory --verbose
```

You can also use the main CLI:

```bash
# Show help
sma --help

# Show version information
sma version

# Analyze a codebase
sma intent path/to/directory --python
```

### Python API

The Structural Intent Analysis system can also be used as a Python library:

```python
from pathlib import Path
from semantic_matrix_analyzer.intent.config.configuration import Configuration
from semantic_matrix_analyzer.intent.analyzers.intent_analyzer import ConfigurableIntentAnalyzer
from semantic_matrix_analyzer.cross_file.dependency_graph import DependencyGraph, DependencyExtractor

# Create a configuration
config = Configuration()

# Create an analyzer
analyzer = ConfigurableIntentAnalyzer(config)

# Find files to analyze
file_paths = [Path("path/to/file.py")]

# Build dependency graph
dependency_graph = DependencyGraph()
dependency_extractor = DependencyExtractor()

for file_path in file_paths:
    nodes, edges = dependency_extractor.extract_dependencies(file_path)
    
    for node in nodes:
        dependency_graph.add_node(node)
    
    for edge in edges:
        dependency_graph.add_edge(edge)

# Analyze codebase
report = analyzer.analyze_codebase(file_paths, dependency_graph)

# Format report
formatted_report = analyzer.format_report(report, "markdown")

# Print report
print(formatted_report)
```

## Configuration

The Structural Intent Analysis system is highly configurable. You can customize the analysis by providing a configuration file or by using the Python API.

### Configuration File

You can provide a configuration file in JSON, YAML, or Python format:

```json
{
    "name_analysis": {
        "enabled": true,
        "tokenization": {
            "separators": ["_", "-", " "],
            "normalize_tokens": true
        },
        "semantic_extraction": {
            "action_verbs": {
                "get": "Retrieve or access",
                "set": "Modify or update",
                "create": "Create or instantiate",
                "delete": "Remove or destroy",
                "update": "Modify or change"
            }
        }
    },
    "type_analysis": {
        "enabled": true,
        "type_mappings": {
            "str": ["String", "Textual data", "entity"],
            "int": ["Integer", "Numeric data", "entity"],
            "float": ["Float", "Numeric data with decimal precision", "entity"],
            "bool": ["Boolean", "True/False condition", "state"]
        }
    },
    "structural_analysis": {
        "enabled": true,
        "patterns": {
            "layered_architecture": {
                "enabled": true,
                "layer_names": ["presentation", "ui", "application", "service", "domain", "model", "data", "persistence", "infrastructure"],
                "confidence": 0.7
            }
        }
    },
    "integration": {
        "combine_intents": true,
        "build_hierarchy": true,
        "report_format": "text",
        "min_confidence": 0.3,
        "max_results": 100
    }
}
```

### Python API

You can also configure the analysis using the Python API:

```python
from semantic_matrix_analyzer.intent.config.configuration import Configuration

# Create a configuration
config = Configuration()

# Set configuration options
config.set("name_analysis.enabled", True)
config.set("name_analysis.tokenization.separators", ["_", "-", " "])
config.set("name_analysis.semantic_extraction.action_verbs.get", "Retrieve or access")
config.set("type_analysis.enabled", True)
config.set("type_analysis.type_mappings.str", ["String", "Textual data", "entity"])
config.set("structural_analysis.enabled", True)
config.set("structural_analysis.patterns.layered_architecture.enabled", True)
config.set("integration.combine_intents", True)
config.set("integration.build_hierarchy", True)
config.set("integration.report_format", "text")
config.set("integration.min_confidence", 0.3)
config.set("integration.max_results", 100)
```

## Examples

See the `examples` directory for more examples of using the Structural Intent Analysis system:

- `intent_analysis_example.py`: Demonstrates how to use the complete system
- `name_analysis_example.py`: Demonstrates how to use the name analyzer
- `type_analysis_example.py`: Demonstrates how to use the type hint analyzer
- `structural_analysis_example.py`: Demonstrates how to use the structural analyzer
- `cli_example.sh`: Demonstrates how to use the CLI
