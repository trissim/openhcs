# plan_07_meta_tool_async_extension.md
## Component: Meta Tool Async/Await Analysis Extension

### Objective
Extend the meta tool to better analyze async/await patterns in the codebase. Create new analysis components specifically for identifying coroutine issues, improper awaiting, and async function annotations.

### Findings
Based on analysis of the codebase and the meta tool:

1. **Current Meta Tool Limitations**:
   - The meta tool doesn't specifically analyze async/await patterns
   - No specific checks for coroutines that are not properly awaited
   - No specific checks for async functions without proper return type annotations

2. **Async/Await Analysis Needs**:
   - Need to identify coroutines that are not properly awaited
   - Need to identify async functions without proper return type annotations
   - Need to verify compliance with prompt_toolkit async patterns

3. **Affected Components**:
   - `tools/code_analysis/meta_analyzer.py`: Main meta tool
   - `tools/code_analysis/code_analyzer_cli.py`: CLI interface for code analysis
   - `tools/code_analysis/interface_classifier.py`: Interface analysis

### Plan
1. **Create Async Pattern Analyzer**:
   - Create a new component for analyzing async/await patterns
   - Include checks for coroutines that are not properly awaited
   - Include checks for async functions without proper return type annotations

2. **Update Meta Analyzer**:
   - Integrate the async pattern analyzer into the meta tool
   - Add a new command for async pattern analysis
   - Include async pattern analysis in comprehensive analysis

3. **Create Async Interface Analyzer**:
   - Extend the interface classifier to analyze async interfaces
   - Include checks for compliance with prompt_toolkit async patterns
   - Verify proper implementation of async interfaces

4. **Update Code Analyzer CLI**:
   - Add a new command for async pattern analysis
   - Include async pattern analysis in comprehensive analysis
   - Add options for controlling async pattern analysis

5. **Create Async Pattern Report Generator**:
   - Create a new component for generating reports on async patterns
   - Include visualizations of async call graphs
   - Highlight potential coroutine issues

6. **Static Analysis**:
   - Use the extended meta tool to analyze itself
   - Apply static analysis to verify the new components
   - Verify that the new components correctly identify async pattern issues

### Implementation Draft
```python
# In tools/code_analysis/meta_analyzer.py

def analyze_async_patterns(path: str, output_dir: str) -> int:
    """
    Analyze async/await patterns in a file or directory.

    Args:
        path: Path to the file or directory to analyze
        output_dir: Directory to write the output to

    Returns:
        0 on success, non-zero on failure
    """
    # Run async pattern analyzer
    async_analyzer_tool = os.path.join(TOOLS_DIR, "async_analyzer.py")
    if not os.path.exists(async_analyzer_tool):
        print(f"Warning: Async analyzer tool not found at {async_analyzer_tool}")
        return 1

    # Run the async analyzer
    if run_tool(async_analyzer_tool, [path, "--output-dir", output_dir]) != 0:
        print("Warning: Async pattern analysis failed")
        return 1

    return 0

def analyze_comprehensive(path: str, output_dir: str = None) -> int:
    """
    Run a comprehensive analysis on a file or directory.

    Args:
        path: Path to the file or directory to analyze
        output_dir: Directory to write the output to

    Returns:
        0 on success, non-zero on failure
    """
    # ... existing code ...

    # Run async pattern analysis
    if analyze_async_patterns(path, output_dir) != 0:
        print("Warning: Async pattern analysis failed")

    # ... rest of the method ...

# In tools/code_analysis/code_analyzer_cli.py

def main():
    """Main entry point for the code analyzer CLI."""
    parser = argparse.ArgumentParser(description="Code Analyzer CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # ... existing commands ...

    # Async Patterns command
    parser_async = subparsers.add_parser("async-patterns", help="Analyze async/await patterns in a file or directory.")
    parser_async.add_argument("path", help="Path to the file or directory to analyze.")
    parser_async.add_argument("-o", "--output", help="Output Markdown file path. Defaults to reports/code_analysis/async_patterns_<name>.md")
    parser_async.add_argument("--output-dir", help="Output directory for the report. Defaults to reports/code_analysis/")
    parser_async.set_defaults(func=handle_async_patterns)

    # ... rest of the method ...

# In tools/code_analysis/async_analyzer.py

def main():
    """Main entry point for the async analyzer."""
    parser = argparse.ArgumentParser(description="Async Pattern Analyzer")
    parser.add_argument("path", help="Path to the file or directory to analyze.")
    parser.add_argument("-o", "--output", help="Output Markdown file path. Defaults to reports/code_analysis/async_patterns_<name>.md")
    parser.add_argument("--output-dir", help="Output directory for the report. Defaults to reports/code_analysis/")

    args = parser.parse_args()

    # Handle the command
    handle_async_patterns(args)

if __name__ == "__main__":
    main()
```
