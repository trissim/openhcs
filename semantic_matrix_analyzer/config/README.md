# Semantic Matrix Analyzer Configuration

This directory contains configuration files for the Semantic Matrix Analyzer.

## Configuration Structure

The configuration is organized into the following sections:

### Analysis

Controls how the analyzer processes code:

- **weights**: Importance factors for different analysis aspects
- **patterns**: Regular expressions and rules for code structure
- **tokens**: Special markers and tags to identify in code
- **keys**: Indicator words for intent and error detection

### Visualization

Controls how results are displayed:

- **colors**: Color scheme for confidence levels
- **thresholds**: Numeric thresholds for confidence levels

### Output

Controls output formatting:

- **formats**: Available output formats
- **default_format**: Default output format
- **verbosity**: Detail level of output

## Using the Configuration CLI

The Semantic Matrix Analyzer includes a command-line interface for managing configuration:

```bash
# View the entire configuration
python -m semantic_matrix_analyzer.config_cli view

# View a specific section
python -m semantic_matrix_analyzer.config_cli view --section analysis.weights

# Update a weight
python -m semantic_matrix_analyzer.config_cli update --weight name_similarity=0.8

# Update a pattern
python -m semantic_matrix_analyzer.config_cli update --pattern code_structure.max_function_length=120

# Add a token
python -m semantic_matrix_analyzer.config_cli update --add-token special_markers=IMPORTANT

# Remove a token
python -m semantic_matrix_analyzer.config_cli update --remove-token special_markers=NOTE

# Add a key
python -m semantic_matrix_analyzer.config_cli update --add-key intent_indicators=purpose

# Remove a key
python -m semantic_matrix_analyzer.config_cli update --remove-key intent_indicators=aim

# Reset to defaults (requires confirmation)
python -m semantic_matrix_analyzer.config_cli reset --confirm
```

## Examples

### Adjusting Weights for Error Analysis

To optimize for error trace analysis, increase the weights for error indicators:

```bash
python -m semantic_matrix_analyzer.config_cli update --weight error_detection=0.9
python -m semantic_matrix_analyzer.config_cli update --add-key error_indicators=exception
python -m semantic_matrix_analyzer.config_cli update --add-key error_indicators=failure
```

### Customizing for Python Projects

To customize for Python-specific analysis:

```bash
python -m semantic_matrix_analyzer.config_cli update --pattern naming_conventions.function="^[a-z][a-z0-9_]*$"
python -m semantic_matrix_analyzer.config_cli update --add-token docstring_tags=type
python -m semantic_matrix_analyzer.config_cli update --add-token docstring_tags=rtype
```

## Configuration File Format

The configuration is stored in YAML format. You can directly edit the file if preferred:

```yaml
analysis:
  weights:
    name_similarity: 0.7
    type_compatibility: 0.8
    # ...
  patterns:
    naming_conventions:
      class: "^[A-Z][a-zA-Z0-9]*$"
      # ...
  # ...
```

## Custom Configuration Files

You can specify a custom configuration file:

```bash
python -m semantic_matrix_analyzer.config_cli -c /path/to/custom_config.yaml view
```

This allows you to maintain different configurations for different projects or analysis types.
