# Semantic Matrix Analyzer

A tool for analyzing codebases using AST to create semantically dense matrices that correlate intent with AST correctness.

## Overview

The Semantic Matrix Analyzer (SMA) is designed to be universally applicable to any codebase while reducing cognitive load for humans and increasing correctness through AI agents bounded by semantic and AST correctness.

Key features:

1. **Universal Applicability**: Works with any codebase, regardless of language, framework, or size
2. **Cognitive Load Reduction**: Users express concerns in natural language without understanding technical details
3. **Bounded Correctness**: AI agents are constrained by semantic and AST correctness
4. **Clarity First**: Clear communication and documentation take precedence over all other concerns
5. **Agent-Driven Analysis**: Leverages agent judgment to select which files are worth analyzing

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/semantic_matrix_analyzer.git
cd semantic_matrix_analyzer

# Install the package
pip install -e .
```

## Usage

### Basic Analysis

```bash
# Analyze a codebase
semantic-matrix-analyzer analyze --project-dir path/to/codebase

# Analyze specific components
semantic-matrix-analyzer analyze --project-dir path/to/codebase --components "component1,component2"

# Analyze for specific intents
semantic-matrix-analyzer analyze --project-dir path/to/codebase --intents "clean_code,error_handling"
```

### Agent-Driven Analysis

```bash
# Use agent-driven analysis
semantic-matrix-analyzer analyze --project-dir path/to/codebase --agent-driven

# Specify concerns for agent-driven analysis
semantic-matrix-analyzer analyze --project-dir path/to/codebase --agent-driven --concerns "error handling,thread safety"

# Use a configuration file for agent-driven analysis
semantic-matrix-analyzer analyze --project-dir path/to/codebase --agent-driven --config path/to/config.json

# Generate a default agent configuration
semantic-matrix-analyzer generate-agent-config --output path/to/config.json
```

### Extracting Intents from Conversations

```bash
# Extract intents from a conversation file
extract-intents --input path/to/conversation.txt --output path/to/intents.json

# Extract intents from text
extract-intents --text "Intent: Clean Code..." --output path/to/intents.json
```

## Components

### Core Components

- **Language Parsing**: Parses code into AST for different programming languages
- **Pattern Detection**: Detects patterns in code using string, regex, and AST-based matching
- **Intent Registry**: Manages intents and their associated patterns
- **Semantic Matrix**: Correlates components with intents to create a semantic matrix
- **Plugin System**: Allows extending the analyzer with new capabilities

### Agent-Driven Analysis

- **File Selection**: Intelligently selects which files to analyze based on relevance, information value, and effort required
- **Metric Collection**: Collects metrics for files to inform selection
- **Feedback Integration**: Updates the model based on user feedback
- **Configuration System**: Allows customizing the agent's behavior through configuration files

## Development

### Project Structure

```
semantic_matrix_analyzer/
├── agent/                 # Agent-driven analysis
├── conversation/          # Conversation intent extraction
├── core/                  # Core functionality
├── language/              # Language-specific parsing
├── patterns/              # Pattern detection
├── plugins/               # Plugin system
├── cli.py                 # Command-line interface
├── extract_intents.py     # Intent extraction script
└── __main__.py            # Entry point
```

### Adding New Patterns

You can add new patterns by creating a plugin:

```python
from semantic_matrix_analyzer.patterns import Intent
from semantic_matrix_analyzer.plugins import IntentPlugin

class MyIntentPlugin(IntentPlugin):
    @property
    def name(self) -> str:
        return "my_intent_plugin"

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def description(self) -> str:
        return "My custom intent plugin"

    @staticmethod
    def get_intents() -> List[Intent]:
        intents = []

        # Create an intent
        clean_code = Intent(
            name="Clean Code",
            description="Clean, maintainable code"
        )

        # Add patterns to the intent
        clean_code.add_string_pattern(
            name="descriptive_names",
            description="Using descriptive variable names",
            pattern="descriptive",
            weight=1.0
        )

        intents.append(clean_code)
        return intents
```

### Adding New Languages

You can add support for new languages by implementing a language parser:

```python
from semantic_matrix_analyzer.language import LanguageParser, language_registry

class JavaScriptParser(LanguageParser):
    @classmethod
    def get_supported_extensions(cls) -> Set[str]:
        return {".js", ".jsx"}

    def parse_file(self, file_path: Path) -> Any:
        # Implement JavaScript parsing
        pass

    # Implement other required methods...

# Register the parser
language_registry.register_parser(JavaScriptParser)
```

### Customizing Agent-Driven Analysis

You can customize the agent-driven analysis by creating a configuration file:

```json
{
  "selection_threshold": 0.5,
  "min_relevance_score": 0.1,
  "min_information_value": 0.1,
  "max_effort_multiplier": 3.0,

  "explicit_mention_weight": 1.0,
  "component_match_weight": 0.8,
  "central_file_weight": 0.7,
  "historical_usefulness_weight": 0.6,

  "complexity_weight": 0.7,
  "dependency_weight": 0.6,
  "change_frequency_weight": 0.5,

  "file_extensions": [".py", ".pyx", ".pyi"],
  "max_files": 50,

  "custom_functions": {
    "calculate_relevance": "module.submodule:custom_relevance_function",
    "calculate_information_value": "module.submodule:custom_information_value_function",
    "calculate_effort": "module.submodule:custom_effort_function"
  }
}
```

You can also provide custom functions for calculating relevance, information value, and effort:

```python
def custom_relevance_function(analyzer, file_path):
    # Custom logic for calculating relevance
    relevance = 0.1  # Start with minimum relevance

    # Prioritize files that match user concerns
    for concern in analyzer.user_intent.primary_concerns:
        if concern.lower() in file_path.name.lower():
            relevance = max(relevance, 0.9)

    return relevance
```

## Architecture

The Semantic Matrix Analyzer is designed with a modular architecture:

1. **Language Parsing Layer**: Handles parsing code into AST for different languages
2. **Pattern Detection Layer**: Detects patterns in code using various matching strategies
3. **Intent Management Layer**: Manages intents and their associated patterns
4. **Matrix Generation Layer**: Creates semantic matrices correlating components with intents
5. **Agent-Driven Analysis Layer**: Intelligently selects files for analysis
6. **Plugin System Layer**: Allows extending the analyzer with new capabilities

## Future Improvements

- **Multi-Language Support**: Add support for more programming languages
- **Enhanced Pattern Detection**: Implement more sophisticated pattern detection algorithms
- **Semantic Grounding**: Ground all recommendations in actual code patterns
- **Conversation Memory**: Store and utilize conversation history
- **AST Verification**: Verify suggestions against the AST
- **Large Codebase Support**: Optimize for very large codebases
- **VCS Integration**: Integrate with version control systems
- **CI/CD Integration**: Support continuous integration workflows

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
