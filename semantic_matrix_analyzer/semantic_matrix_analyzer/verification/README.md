# Enhanced AST Verification

The Enhanced AST Verification system is a component of the Semantic Matrix Analyzer that provides functionality for verifying code suggestions against the AST, ensuring syntactic and semantic correctness, detecting potential side effects, and providing confidence scores for each suggestion.

## Overview

The Enhanced AST Verification system consists of several key components:

1. **Suggestion Verification**: Verifies code suggestions against the AST
2. **Code Change Simulation**: Simulates code changes before applying them
3. **Side Effect Detection**: Detects potential side effects of code changes
4. **Verification Reporting**: Provides comprehensive reporting of verification results

## Usage

### Basic Usage

```python
from semantic_matrix_analyzer.verification import (
    CodeSuggestion, SuggestionVerifier, SideEffectDetector, VerificationReporter
)

# Create a code suggestion
suggestion = CodeSuggestion(
    file_path=Path("example.py"),
    start_line=1,
    end_line=10,
    original_code="def example():\n    return 42",
    suggested_code="def example():\n    return 43",
    description="Change return value",
    confidence=0.9
)

# Create verifiers
side_effect_detector = SideEffectDetector()
verifier = SuggestionVerifier(side_effect_detector)
reporter = VerificationReporter()

# Verify the suggestion
verification_result = verifier.verify_suggestion(suggestion)

# Store the verification result with the suggestion
suggestion.verification_result = verification_result

# Generate a report
report = reporter.generate_report(suggestion, verification_result)

# Format the report as text
text_report = reporter.format_report(report, format="text")
print(text_report)
```

### Simulating Code Changes

```python
from semantic_matrix_analyzer.verification import CodeChangeSimulator

# Create a code change simulator
simulator = CodeChangeSimulator()

# Simulate the change
simulation_result = simulator.simulate_change(suggestion)

if simulation_result["success"]:
    print("Simulation successful")
    print("Analysis results:", simulation_result["analysis"])
else:
    print("Simulation failed:", simulation_result["error"])
```

## Components

### CodeSuggestion

Represents a suggestion for changing code:

- `file_path`: The path to the file containing the code
- `start_line`: The line number where the suggestion starts
- `end_line`: The line number where the suggestion ends
- `original_code`: The original code
- `suggested_code`: The suggested code
- `description`: A description of the suggestion
- `confidence`: The confidence score for the suggestion (0.0 to 1.0)
- `verification_result`: The result of verifying the suggestion

### VerificationResult

Represents the result of verifying a suggestion:

- `is_valid`: Whether the suggestion is valid
- `syntax_valid`: Whether the suggestion is syntactically valid
- `semantic_valid`: Whether the suggestion is semantically valid
- `side_effects`: A list of potential side effects
- `confidence`: The confidence score for the verification (0.0 to 1.0)
- `error_message`: An error message if the suggestion is invalid

### SuggestionVerifier

Verifies code suggestions against the AST:

- `verify_suggestion(suggestion)`: Verify a code suggestion against the AST
- `_check_syntax(code)`: Check if the code is syntactically valid
- `_check_semantics(suggestion)`: Check if the code is semantically valid
- `_check_side_effects(suggestion)`: Check for potential side effects
- `_calculate_confidence(suggestion, syntax_valid, semantic_valid, side_effects)`: Calculate the confidence score

### CodeChangeSimulator

Simulates code changes to detect potential issues:

- `simulate_change(suggestion)`: Simulate a code change and return the results
- `_create_temp_file(file_path)`: Create a temporary copy of the file
- `_apply_suggestion(suggestion, temp_file)`: Apply the suggestion to the temporary file
- `_analyze_ast(tree)`: Analyze the AST for potential issues

### SideEffectDetector

Detects potential side effects of code changes:

- `detect_side_effects(suggestion)`: Detect potential side effects of a code change
- `_detect_function_signature_changes(original_ast, modified_ast)`: Detect changes to function signatures
- `_detect_class_interface_changes(original_ast, modified_ast)`: Detect changes to class interfaces
- `_detect_global_variable_changes(original_ast, modified_ast)`: Detect changes to global variables
- `_detect_import_changes(original_ast, modified_ast)`: Detect changes to imports

### VerificationReporter

Reports verification results:

- `generate_report(suggestion, verification_result)`: Generate a report of verification results
- `format_report(report, format)`: Format a report in the specified format
- `summarize_verification_result(verification_result)`: Summarize a verification result

## Benefits

The Enhanced AST Verification system provides several benefits:

1. **Improved Correctness**: All suggestions are verified against the AST before being proposed
2. **Reduced Cognitive Load**: Users don't need to mentally verify the correctness of suggestions
3. **Side Effect Detection**: Potential unintended consequences of changes are identified
4. **Confidence Scoring**: Users can prioritize high-confidence suggestions
5. **Feedback Loop**: The system learns from verification results to improve future suggestions
