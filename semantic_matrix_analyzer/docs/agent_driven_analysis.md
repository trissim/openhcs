# Agent-Driven Analysis

This document describes the agent-driven approach to code analysis implemented in the Semantic Matrix Analyzer. This approach leverages the agent's judgment to reduce human cognitive load by intelligently selecting which files are worth analyzing.

The agent-driven analyzer is now fully configurable, allowing you to customize the selection algorithm, thresholds, weights, and even provide custom functions for calculating relevance, information value, and effort.

## Motivation

Static analysis of large codebases faces several challenges:

1. **Scale**: Analyzing every file in a large codebase is computationally expensive
2. **Relevance**: Not all files are equally relevant to the user's concerns
3. **Cognitive Load**: Users shouldn't need to manually select which files to analyze
4. **Efficiency**: Analysis resources should be focused on high-value targets

The agent-driven approach addresses these challenges by using an intelligent agent to select which files to analyze based on relevance, information value, and effort required.

## Formal Model

The agent-driven approach is formalized as follows:

### Selection Function

The agent applies a selection function **S(f)** to determine which files to analyze:

**S(f) = R(f) × I(f) / E(f) > T**

Where:
- **R(f)**: Relevance function for file f (0.0 to 1.0)
- **I(f)**: Information value function for file f (0.0 to 1.0)
- **E(f)**: Effort required to analyze file f (positive value)
- **T**: User-specified threshold for analysis (default: 0.5)

### Relevance Function

The relevance function **R(f)** considers:

- File's relationship to user's stated concerns
- Architectural centrality (how many other files depend on it)
- Historical usefulness (files that were useful in past analyses)
- Recent changes

### Information Value Function

The information value function **I(f)** considers:

- Complexity metrics (cyclomatic complexity, etc.)
- Pattern density (number of potential patterns per LOC)
- Semantic richness (how much "intent" is expressed)
- Dependency count (files with many dependents have higher value)

### Effort Function

The effort function **E(f)** considers:

- File size
- Language complexity
- Parse time
- Analysis difficulty

## Implementation

The agent-driven approach is implemented in the `AgentDrivenAnalyzer` class, which provides:

1. **File Selection**: Selects which files to analyze based on the selection function
2. **Metric Collection**: Collects metrics for files to inform selection
3. **Feedback Integration**: Updates the model based on user feedback
4. **Analysis Execution**: Analyzes selected files using the Semantic Matrix Analyzer
5. **Configuration Management**: Loads and saves configuration from JSON files

## Configuration

The agent-driven analyzer is fully configurable through the `AgentConfig` class. You can customize:

### Selection Thresholds

- `selection_threshold`: Minimum score for a file to be selected (default: 0.5)
- `min_relevance_score`: Minimum relevance score (default: 0.1)
- `min_information_value`: Minimum information value score (default: 0.1)
- `max_effort_multiplier`: Maximum effort multiplier (default: 3.0)

### Relevance Weights

- `explicit_mention_weight`: Weight for explicitly mentioned files (default: 1.0)
- `component_match_weight`: Weight for files matching mentioned components (default: 0.8)
- `central_file_weight`: Weight for files central to the codebase (default: 0.7)
- `historical_usefulness_weight`: Weight for files that were useful in the past (default: 0.6)

### Information Value Weights

- `complexity_weight`: Weight for file complexity (default: 0.7)
- `dependency_weight`: Weight for file dependencies (default: 0.6)
- `change_frequency_weight`: Weight for file change frequency (default: 0.5)

### Effort Multipliers

- `large_file_multiplier`: Effort multiplier for large files (default: 1.5)
- `very_large_file_multiplier`: Effort multiplier for very large files (default: 2.0)
- `complex_file_multiplier`: Effort multiplier for complex files (default: 1.5)
- `very_complex_file_multiplier`: Effort multiplier for very complex files (default: 2.0)
- `many_dependencies_multiplier`: Effort multiplier for files with many dependencies (default: 1.5)

### Thresholds

- `large_file_threshold`: Threshold for large files (default: 500 lines)
- `very_large_file_threshold`: Threshold for very large files (default: 1000 lines)
- `many_dependencies_threshold`: Threshold for many dependencies (default: 10)
- `high_change_frequency`: Threshold for high change frequency (default: 5.0 changes per month)
- `medium_change_frequency`: Threshold for medium change frequency (default: 2.0 changes per month)

### File Selection

- `file_extensions`: Set of file extensions to analyze (default: {".py"})
- `max_files`: Maximum number of files to analyze (default: None, meaning no limit)

### Custom Functions

You can provide custom functions for calculating relevance, information value, and effort:

```python
def custom_relevance_function(analyzer, file_path):
    # Custom logic for calculating relevance
    return relevance_score

def custom_information_value_function(analyzer, file_path):
    # Custom logic for calculating information value
    return information_value_score

def custom_effort_function(analyzer, file_path):
    # Custom logic for calculating effort
    return effort_score
```

These functions can be specified in the configuration file:

```json
{
  "custom_functions": {
    "calculate_relevance": "module.submodule:custom_relevance_function",
    "calculate_information_value": "module.submodule:custom_information_value_function",
    "calculate_effort": "module.submodule:custom_effort_function"
  }
}
```

## Usage

### Basic Usage

```python
from semantic_matrix_analyzer.agent import AgentDrivenAnalyzer, UserIntent

# Define user intent
user_intent = UserIntent(
    primary_concerns=["error handling", "thread safety"],
    component_mentions=["database", "network"]
)

# Create analyzer with default configuration
analyzer = AgentDrivenAnalyzer(codebase_path="path/to/codebase", user_intent=user_intent)

# Collect file metrics
analyzer.collect_file_metrics()

# Select files for analysis
selected_files = analyzer.select_files_for_analysis()

# Analyze selected files
results = analyzer.analyze_selected_files(selected_files)

# Update model based on user feedback
for selection in selected_files:
    file_path = selection.file_path
    was_useful = True  # This would come from user feedback
    analyzer.update_from_feedback(file_path, was_useful)
```

### Using Configuration

```python
from semantic_matrix_analyzer.agent import AgentDrivenAnalyzer, UserIntent
from semantic_matrix_analyzer.agent.config import AgentConfig

# Define user intent
user_intent = UserIntent(
    primary_concerns=["error handling", "thread safety"],
    component_mentions=["database", "network"]
)

# Create a custom configuration
config = AgentConfig()
config.selection_threshold = 0.4  # Lower threshold to select more files
config.file_extensions = {".py", ".js", ".ts"}  # Analyze Python and JavaScript/TypeScript files
config.max_files = 50  # Limit to 50 files

# Create analyzer with custom configuration
analyzer = AgentDrivenAnalyzer(
    codebase_path="path/to/codebase",
    user_intent=user_intent,
    config=config
)

# Or load configuration from a file
analyzer = AgentDrivenAnalyzer(
    codebase_path="path/to/codebase",
    user_intent=user_intent,
    config_path="path/to/config.json"
)

# Save configuration for future use
analyzer.save_config("path/to/save/config.json")
```

### Using Custom Functions

```python
def my_relevance_function(analyzer, file_path):
    # Custom logic for calculating relevance
    relevance = 0.1  # Start with minimum relevance

    # Prioritize files that match user concerns
    for concern in analyzer.user_intent.primary_concerns:
        if concern.lower() in file_path.name.lower():
            relevance = max(relevance, 0.9)

    return relevance

# Create a custom configuration with custom functions
config = AgentConfig()
config.calculate_relevance_func = my_relevance_function

# Create analyzer with custom configuration
analyzer = AgentDrivenAnalyzer(
    codebase_path="path/to/codebase",
    user_intent=user_intent,
    config=config
)
```

### Command-Line Interface

The agent-driven approach is also available through the command-line interface:

```bash
# Basic usage
python -m semantic_matrix_analyzer analyze \
    --project-dir path/to/codebase \
    --agent-driven \
    --concerns "error handling,thread safety" \
    --components "database,network"

# Using configuration
python -m semantic_matrix_analyzer analyze \
    --project-dir path/to/codebase \
    --agent-driven \
    --config path/to/config.json

# Generate a default configuration
python -m semantic_matrix_analyzer generate-agent-config \
    --output path/to/config.json
```

## Benefits

The agent-driven approach provides several benefits:

1. **Reduced Cognitive Load**: Users don't need to manually select files
2. **Improved Efficiency**: Analysis resources are focused on high-value targets
3. **Adaptive Analysis**: The system learns from past analyses to improve future selections
4. **Scalability**: Makes analysis of large codebases tractable

## Feedback Loop

The agent-driven approach includes a feedback loop that allows the system to learn from user feedback:

1. Agent selects files for analysis
2. User provides feedback on which files were useful
3. Agent updates its model based on feedback
4. Agent uses updated model for future selections

This feedback loop creates a virtuous cycle where the system becomes more accurate over time.

## Limitations

The agent-driven approach has some limitations:

1. **Cold Start**: Initial selections may not be optimal without historical data
2. **Dependency on Metrics**: Quality of selection depends on quality of metrics
3. **Missed Connections**: May miss issues that span multiple files if some are filtered out

## Future Work

Future improvements to the agent-driven approach include:

1. **Improved Metrics**: More sophisticated metrics for relevance and information value
2. **Cross-File Analysis**: Better handling of issues that span multiple files
3. **Learning Models**: Machine learning models for file selection
4. **User Preference Learning**: Learning user preferences from feedback
5. **Explanation Generation**: Explaining why files were selected for analysis
