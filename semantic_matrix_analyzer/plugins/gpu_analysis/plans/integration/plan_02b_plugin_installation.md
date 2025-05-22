# Plan 02b: Plugin Registration Mechanism - Installation and CLI Integration

## Objective

Create an installation mechanism for the GPU Analysis Plugin and integrate it with SMA's CLI to implement the unimplemented CLI commands.

## Background

In addition to the core registration mechanism covered in Plan 02a, the GPU Analysis Plugin needs an installation script to deploy it into SMA's plugin directory. It also needs to integrate with SMA's CLI to implement the unimplemented CLI commands for analyzing code, processing error traces, and extracting intents.

## Current State

SMA's CLI in `semantic_matrix_analyzer/sma_cli.py` has several unimplemented command handlers:

```python
def handle_analyze_command(args: argparse.Namespace) -> None:
    """Handle the analyze command."""
    print_header("CODE ANALYSIS")
    print("Analyzing code for intent extraction...")
    # Implementation would go here
    print(color_text("Not yet implemented", "YELLOW"))

def handle_error_trace_command(args: argparse.Namespace) -> None:
    """Handle the error-trace command."""
    print_header("ERROR TRACE PROCESSING")
    print("Processing error trace...")
    # Implementation would go here
    print(color_text("Not yet implemented", "YELLOW"))

def handle_extract_intent_command(args: argparse.Namespace) -> None:
    """Handle the extract-intent command."""
    print_header("INTENT EXTRACTION")
    print("Extracting intent from conversation...")
    # Implementation would go here
    print(color_text("Not yet implemented", "YELLOW"))
```

Additionally, the `generate_project_snapshot` function has placeholders for semantic analysis:

```python
# Placeholder for semantic analysis
if focus in ["semantics", "all"] and depth >= 2:
    snapshot["semantics"] = {
        "status": "placeholder",
        "message": "Semantic analysis would analyze code patterns, naming conventions, and code quality."
    }
```

## Implementation Plan

### 1. Create an Installation Script

Create a script to install the GPU Analysis Plugin into SMA's plugin directory:

```python
# Create a new file: brain/gpu_analysis/install_plugin.py

#!/usr/bin/env python3
"""
Installation script for the GPU Analysis Plugin.

This script installs the GPU Analysis Plugin into SMA's plugin directory.
"""

import os
import sys
import shutil
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("install_plugin")

def install_plugin(sma_dir: Path = None, plugin_name: str = "gpu_analysis_plugin"):
    """
    Install the GPU Analysis Plugin into SMA's plugin directory.

    Args:
        sma_dir: Path to the SMA directory
        plugin_name: Name of the plugin file (without .py extension)

    Returns:
        True if installation was successful, False otherwise
    """
    try:
        # Determine SMA directory
        if sma_dir is None:
            # Try to find SMA directory
            sma_dir = Path("/home/ts/code/projects/openhcs/semantic_matrix_analyzer")
            if not sma_dir.exists():
                logger.error(f"SMA directory not found: {sma_dir}")
                return False

        # Determine plugin directories
        src_dir = Path(__file__).parent
        dst_dir = sma_dir / "semantic_matrix_analyzer" / "plugins"

        if not dst_dir.exists():
            logger.error(f"SMA plugin directory not found: {dst_dir}")
            return False

        # Copy plugin files
        src_file = src_dir / f"{plugin_name}.py"
        dst_file = dst_dir / f"{plugin_name}.py"

        if not src_file.exists():
            logger.error(f"Plugin file not found: {src_file}")
            return False

        # Copy the plugin file
        shutil.copy2(src_file, dst_file)
        logger.info(f"Copied {src_file} to {dst_file}")

        # Copy required modules
        required_modules = [
            "ast_adapter.py",
            "ast_tensor.py",
            "analyzers/semantic_analyzer.py",
            "analyzers/complexity_analyzer.py",
            "analyzers/dependency_analyzer.py",
            "analyzers/__init__.py",
            "pattern_matcher.py",
            "config_manager.py",
            "batch_processor.py",
            "utils/gpu_utils.py",
            "utils/__init__.py",
            "logging_integration.py",
            "error_handling.py",
            "config_integration.py",
        ]

        # Create required directories
        (dst_dir / "analyzers").mkdir(exist_ok=True)
        (dst_dir / "utils").mkdir(exist_ok=True)

        for module in required_modules:
            src_module = src_dir / module
            dst_module = dst_dir / module

            if not src_module.exists():
                logger.warning(f"Required module not found: {src_module}")
                continue

            # Create parent directory if it doesn't exist
            dst_module.parent.mkdir(exist_ok=True)

            # Copy the module
            shutil.copy2(src_module, dst_module)
            logger.info(f"Copied {src_module} to {dst_module}")

        logger.info("GPU Analysis Plugin installed successfully")
        return True

    except Exception as e:
        logger.error(f"Error installing GPU Analysis Plugin: {e}")
        return False

if __name__ == "__main__":
    # Parse command line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Install the GPU Analysis Plugin")
    parser.add_argument("--sma-dir", type=str, help="Path to the SMA directory")
    args = parser.parse_args()

    sma_dir = Path(args.sma_dir) if args.sma_dir else None

    # Install the plugin
    success = install_plugin(sma_dir)

    sys.exit(0 if success else 1)
```

### 2. Create CLI Integration Functions

Create functions to integrate with SMA's CLI commands:

```python
# Add to gpu_analysis_plugin.py:

def register_cli_commands(plugin: 'GPUAnalysisPlugin', context: 'PluginContext') -> None:
    """
    Register the plugin's CLI commands with SMA.

    This function registers the plugin's CLI commands with SMA,
    allowing them to be used to implement unimplemented SMA CLI commands.

    Args:
        plugin: The GPU Analysis Plugin instance
        context: The plugin context
    """
    try:
        import importlib

        # Try to import SMA CLI module
        try:
            sma_cli = importlib.import_module("semantic_matrix_analyzer.sma_cli")
        except ImportError:
            context.log("error", "Failed to import SMA CLI module")
            return

        # Register analyze command handler
        if hasattr(sma_cli, "handle_analyze_command"):
            # Store original method for fallback
            if not hasattr(sma_cli, "_original_handle_analyze_command"):
                sma_cli._original_handle_analyze_command = sma_cli.handle_analyze_command

            # Replace with GPU-accelerated method
            def gpu_handle_analyze_command(args: Any) -> None:
                """GPU-accelerated analyze command handler."""
                try:
                    # Print header
                    if hasattr(sma_cli, "print_header"):
                        sma_cli.print_header("CODE ANALYSIS")
                    print("Analyzing code for intent extraction using GPU acceleration...")

                    # Extract code or file path from args
                    code = None
                    file_path = None

                    if hasattr(args, 'code_file') and args.code_file:
                        file_path = args.code_file
                        print(f"Analyzing file: {file_path}")
                    elif hasattr(args, 'project_dir') and args.project_dir:
                        print(f"Analyzing project directory: {args.project_dir}")
                        # Analyze project directory
                        results = plugin.analyze_project(args.project_dir)

                        # Print results
                        print(f"Analyzed {len(results)} files using GPU acceleration")

                        # Format output
                        if hasattr(args, 'format') and args.format == "json":
                            import json
                            print(json.dumps(results, indent=2))
                        else:
                            # Print summary
                            print(f"Average complexity: {sum(r['complexity']['cyclomatic_complexity'] for r in results.values()) / len(results):.2f}")
                            print(f"Total pattern matches: {sum(len(r.get('pattern_matches', [])) for r in results.values())}")
                        return

                    # Analyze code or file
                    if file_path:
                        results = plugin.analyze_file(file_path)
                    elif code:
                        results = plugin.analyze_code(code)
                    else:
                        print("No code or file specified")
                        return

                    # Format output
                    if hasattr(args, 'format') and args.format == "json":
                        import json
                        print(json.dumps(results, indent=2))
                    else:
                        # Print results
                        print(f"Analysis completed using GPU acceleration")
                        print(f"Complexity: {results.get('complexity', {})}")
                        print(f"Patterns: {len(results.get('pattern_matches', []))}")
                except Exception as e:
                    context.log("error", f"GPU analysis failed, falling back to original method: {e}")
                    sma_cli._original_handle_analyze_command(args)

            # Apply the replacement
            sma_cli.handle_analyze_command = gpu_handle_analyze_command
            context.log("info", "Registered GPU-accelerated handle_analyze_command with SMA CLI")

        # Register error trace command handler
        if hasattr(sma_cli, "handle_error_trace_command"):
            # Store original method for fallback
            if not hasattr(sma_cli, "_original_handle_error_trace_command"):
                sma_cli._original_handle_error_trace_command = sma_cli.handle_error_trace_command

            # Replace with GPU-accelerated method
            def gpu_handle_error_trace_command(args: Any) -> None:
                """GPU-accelerated error trace command handler."""
                try:
                    # Print header
                    if hasattr(sma_cli, "print_header"):
                        sma_cli.print_header("ERROR TRACE PROCESSING")
                    print("Processing error trace using GPU acceleration...")

                    # Get error trace
                    error_trace = ""
                    if hasattr(args, 'input_file') and args.input_file:
                        print(f"Reading error trace from file: {args.input_file}")
                        with open(args.input_file, "r") as f:
                            error_trace = f.read()
                    elif hasattr(args, 'text') and args.text:
                        error_trace = args.text
                    else:
                        print("No error trace specified")
                        return

                    # Analyze error trace
                    results = plugin.analyze_error_trace(error_trace)

                    # Format output
                    if hasattr(args, 'format') and args.format == "json":
                        import json
                        print(json.dumps(results, indent=2))
                    else:
                        # Print results
                        print(f"Error trace analysis completed using GPU acceleration")
                        print(f"Snippets analyzed: {results.get('snippets_analyzed', 0)}")
                        print(f"Error type: {results.get('error_type', 'Unknown')}")

                        # Print suggested fixes
                        if 'suggested_fixes' in results:
                            print("\nSuggested fixes:")
                            for fix in results['suggested_fixes']:
                                print(f"- {fix}")
                except Exception as e:
                    context.log("error", f"GPU error trace analysis failed, falling back to original method: {e}")
                    sma_cli._original_handle_error_trace_command(args)

            # Apply the replacement
            sma_cli.handle_error_trace_command = gpu_handle_error_trace_command
            context.log("info", "Registered GPU-accelerated handle_error_trace_command with SMA CLI")

        # Register extract intent command handler
        if hasattr(sma_cli, "handle_extract_intent_command"):
            # Store original method for fallback
            if not hasattr(sma_cli, "_original_handle_extract_intent_command"):
                sma_cli._original_handle_extract_intent_command = sma_cli.handle_extract_intent_command

            # Replace with GPU-accelerated method
            def gpu_handle_extract_intent_command(args: Any) -> None:
                """GPU-accelerated extract intent command handler."""
                try:
                    # Print header
                    if hasattr(sma_cli, "print_header"):
                        sma_cli.print_header("INTENT EXTRACTION")
                    print("Extracting intent from conversation using GPU acceleration...")

                    # Get conversation text
                    conversation_text = ""
                    if hasattr(args, 'input_file') and args.input_file:
                        print(f"Reading conversation from file: {args.input_file}")
                        with open(args.input_file, "r") as f:
                            conversation_text = f.read()
                    elif hasattr(args, 'text') and args.text:
                        conversation_text = args.text
                    else:
                        print("No conversation specified")
                        return

                    # Extract intents
                    intents = plugin.extract_intents_from_conversation(conversation_text)

                    # Save intents to file if specified
                    if hasattr(args, 'output_file') and args.output_file:
                        with open(args.output_file, "w") as f:
                            import json
                            json.dump(intents, f, indent=2)
                        print(f"Intents saved to {args.output_file}")

                    # Format output
                    if hasattr(args, 'format') and args.format == "json":
                        import json
                        print(json.dumps(intents, indent=2))
                    else:
                        # Print intents
                        print(f"Intent extraction completed using GPU acceleration")
                        print(f"Intents extracted: {len(intents)}")
                        for intent in intents:
                            print(f"\nIntent: {intent['name']}")
                            print(f"Description: {intent['description']}")
                            print(f"Patterns: {len(intent['patterns'])}")
                            for pattern in intent['patterns']:
                                print(f"  - {pattern['name']}: {pattern['pattern']} ({pattern['pattern_type']})")
                except Exception as e:
                    context.log("error", f"GPU intent extraction failed, falling back to original method: {e}")
                    sma_cli._original_handle_extract_intent_command(args)

            # Apply the replacement
            sma_cli.handle_extract_intent_command = gpu_handle_extract_intent_command
            context.log("info", "Registered GPU-accelerated handle_extract_intent_command with SMA CLI")

        # Register semantic analysis in generate_project_snapshot
        if hasattr(sma_cli, "generate_project_snapshot"):
            # Store original method for fallback
            if not hasattr(sma_cli, "_original_generate_project_snapshot"):
                sma_cli._original_generate_project_snapshot = sma_cli.generate_project_snapshot

            # Replace with GPU-accelerated method
            def gpu_generate_project_snapshot(
                project_dir: str,
                depth: int = 3,
                focus: str = "all",
                analyzer: Any = None,
                output_format: str = "markdown"
            ) -> Any:
                """GPU-accelerated project snapshot generation."""
                try:
                    # Call original method to get basic snapshot
                    snapshot = sma_cli._original_generate_project_snapshot(
                        project_dir=project_dir,
                        depth=depth,
                        focus=focus,
                        analyzer=analyzer,
                        output_format="json"  # Always get JSON for processing
                    )

                    # Add GPU-accelerated semantic analysis
                    if focus in ["semantics", "all"] and depth >= 2:
                        # Generate semantic snapshot
                        semantic_snapshot = plugin.generate_semantic_snapshot(project_dir, depth)

                        # Update snapshot with semantic analysis
                        snapshot["semantics"] = semantic_snapshot

                    # Format output
                    if output_format == "json":
                        return snapshot
                    elif output_format == "yaml":
                        import yaml
                        return yaml.dump(snapshot, default_flow_style=False)
                    else:
                        # Format as markdown
                        if hasattr(sma_cli, "format_snapshot_as_markdown"):
                            return sma_cli.format_snapshot_as_markdown(snapshot)
                        else:
                            import json
                            return json.dumps(snapshot, indent=2)
                except Exception as e:
                    context.log("error", f"GPU snapshot generation failed, falling back to original method: {e}")
                    return sma_cli._original_generate_project_snapshot(
                        project_dir=project_dir,
                        depth=depth,
                        focus=focus,
                        analyzer=analyzer,
                        output_format=output_format
                    )

            # Apply the replacement
            sma_cli.generate_project_snapshot = gpu_generate_project_snapshot
            context.log("info", "Registered GPU-accelerated generate_project_snapshot with SMA CLI")

        context.log("info", "Successfully registered CLI commands with SMA")
    except Exception as e:
        context.log("error", f"Error registering CLI commands with SMA: {e}")
```

### 3. Update Plugin Initialization

Update the plugin's `initialize` method to register CLI commands:

```python
def initialize(self, context: PluginContext) -> None:
    """Initialize the plugin with the given context.

    Args:
        context: The plugin context.
    """
    self.context = context

    # Log initialization using SMA's logging system
    context.log("info", f"GPU Analysis Plugin initialized with device: {self.device}")
    if self.device == "cuda":
        context.log("info", f"CUDA device: {torch.cuda.get_device_name(0)}")
        context.log("info", f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Register plugin methods with SMA
    register_plugin_methods_with_sma(self, context)

    # Register CLI commands
    register_cli_commands(self, context)
```

### 4. Add Project Analysis Method

Add a method to analyze an entire project directory:

```python
def analyze_project(self, project_dir: Union[str, Path],
                   analysis_types: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
    """
    Analyze a project directory using GPU acceleration.

    Args:
        project_dir: Path to the project directory
        analysis_types: Types of analysis to perform (if None, perform all)

    Returns:
        Dictionary of analysis results by file path
    """
    try:
        if self.context:
            self.context.log("debug", f"Analyzing project directory: {project_dir}")

        # Find Python files
        python_files = []
        for root, _, files in os.walk(project_dir):
            for file in files:
                if file.endswith(".py"):
                    python_files.append(os.path.join(root, file))

        # Analyze files in batch
        results = {}
        for file_path in python_files:
            try:
                result = self.analyze_file(file_path, analysis_types)
                results[file_path] = result
            except Exception as e:
                if self.context:
                    self.context.log("error", f"Error analyzing file {file_path}: {e}")
                # Continue with next file

        return results
    except Exception as e:
        if self.context:
            self.context.log("error", f"Error analyzing project directory {project_dir}: {e}")
        raise RuntimeError(f"GPU analysis failed for project directory {project_dir}: {e}") from e
```

## Implementation Focus

The implementation should focus on:

1. **Installation Mechanism**: Creating a script to install the GPU Analysis Plugin into SMA's plugin directory.

2. **CLI Integration**: Implementing registration of the plugin's CLI commands with SMA to enable implementation of unimplemented SMA CLI commands.

3. **Project Analysis**: Implementing functionality to analyze entire project directories.

4. **Advanced Analysis**: Implementing functionality to process error traces, extract intents from conversations, and generate semantic snapshots.

## Success Criteria

1. The GPU Analysis Plugin can be installed into SMA's plugin directory using the installation script.

2. The plugin's CLI commands are registered with SMA.

3. The plugin can analyze entire project directories.

4. The plugin can process error traces and extract intents from conversations.

5. The plugin can generate semantic snapshots of projects.

Note: Testing will be deferred until the complete architecture is implemented and stable. The focus is on velocity and architectural completion rather than incremental validation.

## References

1. SMA CLI: `semantic_matrix_analyzer/sma_cli.py`

2. GPU Analysis Plugin: `brain/gpu_analysis/plugin.py`

3. SMA Plugin Manager: `semantic_matrix_analyzer/semantic_matrix_analyzer/plugins/__init__.py`
