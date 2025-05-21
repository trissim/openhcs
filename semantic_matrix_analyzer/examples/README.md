# Semantic Matrix Analyzer Examples

This directory contains examples of how to use the Semantic Matrix Analyzer tool.

## Clean Code Example

This example demonstrates how to analyze code for clean code principles.

### Files

- `clean_code_conversation.txt`: A conversation between a user and an AI about clean code principles
- `clean_code_config.json`: Configuration file for the analysis
- `sample_code/good_code.py`: Example of code that follows clean code principles
- `sample_code/bad_code.py`: Example of code that violates clean code principles

### Running the Example

1. Extract intents from the conversation:

```bash
cd semantic_matrix_analyzer
python -m semantic_matrix_analyzer.extract_intents --input examples/clean_code_conversation.txt --output examples/clean_code_intents.json
```

2. Run the analysis:

```bash
python -m semantic_matrix_analyzer.semantic_matrix_analyzer analyze --config examples/clean_code_config.json
```

3. View the results:

```bash
cat examples/output/clean_code_analysis_report.md
```

## Expected Results

The analysis should show that:

- `good_code.py` has high alignment with clean code principles
- `bad_code.py` has low alignment with clean code principles

The report will provide details on which specific principles are followed or violated in each file.

## Creating Your Own Examples

To create your own examples:

1. Create a conversation file with intents and patterns
2. Extract intents from the conversation
3. Create a configuration file pointing to your code
4. Run the analysis
5. Review the results

See the main README for more detailed instructions.
