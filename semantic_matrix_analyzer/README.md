# Semantic Matrix Analyzer

## Core Principle: Separation of Judgment from Execution through Intent Extraction

The Semantic Matrix Analyzer (SMA) is a specialized tool designed for AI agents to assist humans in architecting, implementing, and debugging code with maximum accuracy and minimum cognitive load. It achieves this through the principle of **separating judgment from execution through intent extraction**.

## Key Benefits

- **Reduced Cognitive Load**: Focus on high-level design while agents handle semantic details
- **Improved Code Quality**: Ensure consistency between intent and implementation
- **Accelerated Development**: Spend less time debugging and more time creating
- **Better Understanding**: Quickly grasp unfamiliar codebases through agent-assisted analysis
- **Error Trace Processing**: Identify root causes through semantic analysis

## Technical Features

- **AST-based Analysis**: Analyzes Python code structure using the built-in `ast` module
- **Intent Extraction**: AI extracts intents from code, error traces, and conversations
- **Dynamic Configuration**: Adjusts analysis parameters based on context and feedback
- **Error Trace Processing**: Identifies root causes through semantic analysis of error stacks
- **Mental Model Building**: Constructs comprehensive understanding of code relationships
- **Pattern Detection**: Detects specific code patterns using string matching, regex, and AST traversal
- **Matrix Generation**: Creates a matrix correlating components with intents
- **Visualization**: Generates heatmap visualizations of the semantic matrix
- **Reporting**: Produces comprehensive Markdown reports with findings and recommendations

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install numpy matplotlib
```

## Usage

### Declaration of Principle

**This tool is now established as the principal methodology for AI-assisted development.** All coding projects will benefit from:

1. Separation of judgment from execution through intent extraction
2. Reduction of cognitive load through AI-assisted semantic analysis
3. Improved code quality through intent-driven development
4. Accelerated development through focused human-agent collaboration

### For Humans: Focus on Creativity, Not Implementation

The Semantic Matrix Analyzer is designed to reduce your cognitive load:

1. **Focus on high-level objectives** and let the AI agent handle semantic analysis
2. **Describe your concerns** about code quality, error traces, or implementation challenges
3. **Receive structured insights** about code intent and behavior
4. **Make informed decisions** with less mental effort

You never need to run any commands yourself or understand the technical details. The AI agent uses SMA on your behalf.

### For AI Agents: Intent-Driven Development

AI agents should use SMA as their primary approach for assisting with code:

1. **Configure the analyzer** for the specific context:
   ```python
   from semantic_matrix_analyzer.config_manager import ConfigManager

   config = ConfigManager()
   config.set_weight("error_detection", 0.9)
   config.add_key("error_indicators", "exception")
   ```

2. **Analyze code or error traces** to extract intent:
   ```python
   from semantic_matrix_analyzer.analyzer import SemanticAnalyzer

   analyzer = SemanticAnalyzer()
   results = analyzer.analyze_code(code_snippet)
   ```

3. **Build a comprehensive mental model** based on the analysis:
   ```python
   mental_model = analyzer.build_mental_model(results)
   ```

4. **Provide targeted assistance** based on the extracted intent:
   ```python
   recommendations = analyzer.generate_recommendations(mental_model)
   ```

See [Agent-Assisted Development](docs/agent_assisted_development.md) for a detailed guide on this approach.

### Technical Reference (For AI Agents)

The following command-line options are provided for AI agents to use when analyzing codebases. Users should not need to use these commands directly.

#### `semantic_matrix_analyzer.py analyze` Command

- `--config`: Path to configuration file (JSON)
- `--project-dir`: Path to project directory (default: current directory)
- `--components`: Comma-separated list of components to analyze
- `--intents`: Comma-separated list of intents to analyze
- `--output-dir`: Directory to save output files (default: current directory)
- `--output-prefix`: Prefix for output files (default: "semantic_matrix")
- `--format`: Output format for visualization (png, svg, pdf) (default: png)

#### `semantic_matrix_analyzer.py generate-config` Command

- `--output`: Output file path (default: "semantic_matrix_config.json")

#### `extract_intents.py` Command

- `--input`: Path to conversation text file
- `--text`: Conversation text (alternative to --input)
- `--output`: Path to output JSON file
- `--append`: Append to existing file instead of overwriting

### Configuration File

The configuration file is a JSON file with the following structure:

```json
{
  "project_dir": ".",
  "components": {
    "Component1": "path/to/component1.py",
    "Component2": "path/to/component2.py"
  },
  "intents": [
    "Intent1",
    "Intent2"
  ],
  "output": {
    "directory": ".",
    "prefix": "semantic_matrix",
    "format": "png"
  }
}
```

- `project_dir`: Root directory of the project
- `components`: Dictionary mapping component names to file paths (relative to project_dir)
  - If a value is `null`, the tool will try to infer the file path
- `intents`: List of intents to analyze
- `output`: Output configuration
  - `directory`: Directory to save output files
  - `prefix`: Prefix for output files
  - `format`: Output format for visualization (png, svg, pdf)

## Output Files

The tool generates the following output files:

- `{prefix}.{format}`: Visualization of the semantic matrix
- `{prefix}_report.md`: Comprehensive report with findings and recommendations
- `{prefix}_data.json`: Analysis data in JSON format for further processing

## Error Trace Processing

### Core Principles of SMA Error Trace Processing

1. **Comprehensive Mental Model**: Use SMA extensively to build a complete mental model of the codebase
2. **Assumption Identification**: Leverage SMA to identify and resolve any assumptions in reasoning that could lead to fatal errors
3. **Ambiguity Reporting**: Report to senior developers when encountering ambiguities that need clarification
4. **Guided Resolution**: Wait for guidance before proceeding when faced with critical uncertainties
5. **Error Trace Analysis First**: Always analyze error traces with SMA before attempting fixes
6. **Semantic Understanding**: Focus on understanding the semantic relationships between components in the error stack
7. **Root Cause Identification**: Use SMA to identify the root cause by analyzing intent and dependencies
8. **Targeted Fixes**: Make precise, targeted fixes based on comprehensive analysis

### Example: Error Trace Analysis

```python
# Configure for error analysis
config.set_weight("error_context_analysis", 0.9)
config.add_key("error_indicators", "NameError")

# Analyze error trace
analysis = analyzer.analyze_error_trace("""
NameError: name 'MemoryType' is not defined
File "constants.py", line 48
""")

# Build mental model
model = analyzer.build_mental_model(analysis)
# The model reveals:
# - MemoryType is defined later in the file (line 64)
# - ENFORCED_BACKEND_MAPPING is using MemoryType before its definition
# - This violates the semantic ordering requirement

# Generate recommendations
recommendations = analyzer.generate_recommendations(model)
# Recommendations include:
# - Move ENFORCED_BACKEND_MAPPING after MemoryType definition
# - Or remove ENFORCED_BACKEND_MAPPING if not essential
```

### Behind the Scenes (What the AI Does)

1. The AI extracts intents from the conversation:
   - Error handling practices
   - Specific exception catching
   - Exception logging

2. The AI runs the analysis:
   ```bash
   python extract_intents.py --text "..." --output conversations/error_handling.json
   python semantic_matrix_analyzer.py analyze --project-dir ./user_codebase --config error_handling_config.json
   ```

3. The AI interprets the results:
   - `./output/semantic_matrix.png`: Visualization of the semantic matrix
   - `./output/semantic_matrix_report.md`: Comprehensive report
   - `./output/semantic_matrix_data.json`: Analysis data in JSON format

## Extending the Tool

### Creating Custom Intent Plugins

You can create custom intent plugins by subclassing the `IntentPlugin` class:

```python
# plugins/my_custom_plugin.py
from semantic_matrix_analyzer import IntentPlugin, Intent

class MyCustomPlugin(IntentPlugin):
    @staticmethod
    def get_intents() -> List[Intent]:
        intents = []

        # Create custom intents
        my_intent = Intent(
            name="My Custom Intent",
            description="Description of my custom intent"
        )

        # Add patterns to the intent
        my_intent.add_string_pattern(
            name="pattern1",
            description="Description of pattern1",
            pattern="pattern1",
            weight=1.0
        )

        intents.append(my_intent)
        return intents
```

### Defining Intents in Conversations

You can define intents in conversations using a specific format that can be extracted by the `extract_intents.py` script:

```
Intent: Clean Code
Pattern: meaningful name
Type: string
Description: Using meaningful variable and function names

Pattern: def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)\s*:\s*(?:\s*\"\"\"[^\"]*\"\"\"\s*)?(?:[^\n]*\n){1,10}\s*return
Type: regex
Description: Writing small, focused functions
```

### Adding New Analysis Rules

You can add new analysis rules by modifying the `analyze_component` method in the `SemanticMatrixBuilder` class. For example, to add a rule that checks for direct file I/O:

```python
# Check for direct file I/O
if "open(" in source_code or "with open" in source_code:
    analysis.issues.append({
        "type": "direct_file_io",
        "message": f"Component '{component}' uses direct file I/O instead of VFS"
    })
```

### Adding New Pattern Types

You can add new pattern types by extending the `IntentPattern` class and updating the `IntentDetector._check_pattern` method:

```python
# Add a new pattern type
def _check_new_pattern_type(self, pattern: Any, analysis: ComponentAnalysis) -> float:
    # Implement your pattern checking logic here
    return 0.0  # Return a score between 0.0 and 1.0

# Update the _check_pattern method
def _check_pattern(self, pattern: IntentPattern, analysis: ComponentAnalysis) -> float:
    if pattern.pattern_type == "new_type":
        return self._check_new_pattern_type(pattern.pattern, analysis)
    # ... existing pattern types ...
```

### Custom Component Path Resolution

You can customize how component paths are resolved by modifying the `get_file_path_for_component` method in the `SemanticMatrixBuilder` class.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
