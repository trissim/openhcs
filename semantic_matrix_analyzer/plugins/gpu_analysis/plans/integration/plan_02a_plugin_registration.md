# Plan 02a: Plugin Registration Mechanism - Core Registration

## Objective

Refactor the GPU Analysis Plugin registration mechanism to properly integrate with SMA's plugin discovery and loading system, enabling the implementation of unimplemented SMA methods.

## Background

SMA uses a plugin manager to discover, load, and manage plugins. The current GPU Analysis Plugin registration mechanism doesn't align with SMA's expectations, which will prevent proper integration. Additionally, the plugin needs to be registered in a way that allows it to implement the unimplemented SMA methods identified in Plan 01.

## Current State

The current GPU Analysis Plugin registration in `brain/gpu_analysis/plugin.py` is implemented as:

```python
def register_plugin(sma_registry):
    """
    Register the GPU analysis plugin with SMA's registry.

    Args:
        sma_registry: SMA registry
    """
    # Create plugin
    plugin = GPUAnalysisPlugin()

    # Register plugin
    sma_registry.register_plugin("gpu_analysis", plugin)

    # Register language parsers
    from gpu_analysis.ast_adapter import register_gpu_parser
    register_gpu_parser(sma_registry.language_registry)

    # Log registration
    logger.info("GPU Analysis Plugin registered with SMA")

    return plugin
```

SMA's plugin manager in `semantic_matrix_analyzer/semantic_matrix_analyzer/plugins/__init__.py` works differently:

```python
class PluginManager:
    """Manages the discovery, loading, and lifecycle of plugins."""

    def __init__(self):
        """Initialize the plugin manager."""
        self.plugins: Dict[str, SMAPlugin] = {}
        self.intent_plugins: List[IntentPlugin] = []

    def discover_plugins(self, plugin_dirs: List[Path]) -> None:
        """Discover available plugins.

        Args:
            plugin_dirs: List of directories to search for plugins.
        """
        import importlib.util
        import sys

        for plugin_dir in plugin_dirs:
            if not plugin_dir.exists() or not plugin_dir.is_dir():
                continue

            for plugin_file in plugin_dir.glob("*_plugin.py"):
                try:
                    # Import the plugin module
                    module_name = plugin_file.stem
                    spec = importlib.util.spec_from_file_location(module_name, plugin_file)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        sys.modules[module_name] = module
                        spec.loader.exec_module(module)

                        # Find plugin classes in the module
                        for attr_name in dir(module):
                            attr = getattr(module, attr_name)
                            if (isinstance(attr, type) and
                                issubclass(attr, SMAPlugin) and
                                attr is not SMAPlugin and
                                attr is not IntentPlugin):
                                self.load_plugin(attr)
                except Exception as e:
                    import logging
                    logger = logging.getLogger("sma")
                    logger.error(f"Error loading plugin {plugin_file}: {e}")

    def load_plugin(self, plugin_class: Type[SMAPlugin]) -> Optional[SMAPlugin]:
        """Load a plugin.

        Args:
            plugin_class: The plugin class to load.

        Returns:
            The loaded plugin, or None if loading failed.
        """
        try:
            plugin = plugin_class()
            self.plugins[plugin.name] = plugin

            if isinstance(plugin, IntentPlugin):
                self.intent_plugins.append(plugin)

            return plugin
        except Exception as e:
            import logging
            logger = logging.getLogger("sma")
            logger.error(f"Error instantiating plugin {plugin_class.__name__}: {e}")
            return None
```

## Unimplemented SMA Methods

The following SMA methods are currently unimplemented and will be implemented using GPU acceleration:

1. `SemanticMatrixBuilder.analyze_component` in the core module:
   ```python
   # TODO: Extract dependencies
   # TODO: Analyze component
   # TODO: Detect patterns
   # TODO: Calculate intent alignments
   ```

2. CLI command handlers in `sma_cli.py`:
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

3. Semantic analysis placeholders in `generate_project_snapshot`:
   ```python
   # Placeholder for semantic analysis
   if focus in ["semantics", "all"] and depth >= 2:
       snapshot["semantics"] = {
           "status": "placeholder",
           "message": "Semantic analysis would analyze code patterns, naming conventions, and code quality."
       }
   ```

## Implementation Plan

### 1. Create a Proper Plugin Module

Rename `plugin.py` to `gpu_analysis_plugin.py` to match SMA's plugin discovery pattern:

```python
# Rename from:
# brain/gpu_analysis/plugin.py
# To:
# brain/gpu_analysis/gpu_analysis_plugin.py
```

### 2. Update Plugin Registration Function

Replace the current `register_plugin` function with a proper plugin discovery mechanism:

```python
# Remove the current register_plugin function:
def register_plugin(sma_registry):
    """
    Register the GPU analysis plugin with SMA's registry.

    Args:
        sma_registry: SMA registry
    """
    # Create plugin
    plugin = GPUAnalysisPlugin()

    # Register plugin
    sma_registry.register_plugin("gpu_analysis", plugin)

    # Register language parsers
    from gpu_analysis.ast_adapter import register_gpu_parser
    register_gpu_parser(sma_registry.language_registry)

    # Log registration
    logger.info("GPU Analysis Plugin registered with SMA")

    return plugin

# Add a new function for manual registration (for testing and direct use):
def register_with_sma():
    """
    Register the GPU analysis plugin with SMA's plugin manager.

    This function is for manual registration when automatic discovery is not possible.
    It should be called after SMA has been initialized.

    Returns:
        The registered plugin instance, or None if registration failed.
    """
    try:
        from semantic_matrix_analyzer.semantic_matrix_analyzer.plugins import plugin_manager
        from semantic_matrix_analyzer.semantic_matrix_analyzer.language import language_registry

        # Load the plugin using SMA's plugin manager
        plugin = plugin_manager.load_plugin(GPUAnalysisPlugin)

        if plugin:
            # Register language parsers
            from gpu_analysis.ast_adapter import register_gpu_parser
            register_gpu_parser(language_registry)

            # Initialize the plugin
            context = PluginContext()
            plugin.initialize(context)

            logger.info("GPU Analysis Plugin registered with SMA")
            return plugin
        else:
            logger.error("Failed to register GPU Analysis Plugin with SMA")
            return None
    except Exception as e:
        logger.error(f"Error registering GPU Analysis Plugin with SMA: {e}")
        return None
```

### 3. Create a Plugin Entry Point

Create an entry point for SMA's plugin discovery mechanism:

```python
# Add to the end of the file:

# This is the entry point for SMA's plugin discovery mechanism
# SMA will look for plugin classes in the module
GPUAnalysisPluginClass = GPUAnalysisPlugin
```

### 4. Register Plugin Methods with SMA

Add code to register the plugin's methods with SMA's core components:

```python
def register_plugin_methods_with_sma(plugin: GPUAnalysisPlugin, context: PluginContext) -> None:
    """
    Register the plugin's methods with SMA's core components.

    This function registers the plugin's methods with SMA's core components,
    allowing them to be used to implement unimplemented SMA methods.

    Args:
        plugin: The GPU Analysis Plugin instance
        context: The plugin context
    """
    try:
        from semantic_matrix_analyzer.semantic_matrix_analyzer.core import semantic_matrix_builder
        from semantic_matrix_analyzer.semantic_matrix_analyzer.cli import sma_cli

        # Register analyze_component method with SemanticMatrixBuilder
        if hasattr(semantic_matrix_builder, 'SemanticMatrixBuilder'):
            builder = semantic_matrix_builder.SemanticMatrixBuilder

            # Store original method for fallback
            if not hasattr(builder, '_original_analyze_component'):
                builder._original_analyze_component = builder.analyze_component

            # Replace with GPU-accelerated method
            def gpu_analyze_component(self, component: str, file_path: Path) -> Any:
                """GPU-accelerated component analysis."""
                try:
                    return plugin.analyze_component(component, file_path)
                except Exception as e:
                    context.log("error", f"GPU analysis failed, falling back to original method: {e}")
                    return self._original_analyze_component(component, file_path)

            # Apply the replacement
            builder.analyze_component = gpu_analyze_component
            context.log("info", "Registered GPU-accelerated analyze_component method with SemanticMatrixBuilder")

        # Register CLI command handlers
        if hasattr(sma_cli, 'handle_analyze_command'):
            # Store original method for fallback
            if not hasattr(sma_cli, '_original_handle_analyze_command'):
                sma_cli._original_handle_analyze_command = sma_cli.handle_analyze_command

            # Replace with GPU-accelerated method
            def gpu_handle_analyze_command(args: Any) -> None:
                """GPU-accelerated analyze command handler."""
                try:
                    # Extract code or file path from args
                    code = None
                    file_path = None

                    if hasattr(args, 'code_file') and args.code_file:
                        file_path = args.code_file
                    elif hasattr(args, 'project_dir') and args.project_dir:
                        # Analyze project directory
                        results = plugin.analyze_project(args.project_dir)

                        # Print results
                        print(f"Analyzed {len(results)} files using GPU acceleration")
                        return

                    # Analyze code or file
                    if file_path:
                        results = plugin.analyze_file(file_path)
                    elif code:
                        results = plugin.analyze_code(code)
                    else:
                        print("No code or file specified")
                        return

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

        # Register other CLI command handlers similarly
        # ...

        context.log("info", "Successfully registered plugin methods with SMA")
    except Exception as e:
        context.log("error", f"Error registering plugin methods with SMA: {e}")
```

### 5. Update Plugin Initialization

Update the plugin's `initialize` method to register its methods with SMA:

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
```

## Implementation Focus

The implementation should focus on:

1. **Plugin Discovery**: Ensuring the GPU Analysis Plugin can be discovered and loaded by SMA's plugin manager.

2. **Plugin Initialization**: Implementing proper initialization through SMA's plugin manager.

3. **Method Registration**: Implementing registration of the plugin's methods with SMA's core components to enable implementation of unimplemented SMA methods.

4. **API Integration**: Ensuring the plugin can be used through SMA's plugin API.

## Success Criteria

1. The GPU Analysis Plugin can be discovered and loaded by SMA's plugin manager.

2. The plugin is correctly initialized by SMA's plugin manager.

3. The plugin's methods are registered with SMA's core components.

4. The plugin can be used through SMA's plugin API.

Note: Testing will be deferred until the complete architecture is implemented and stable. The focus is on velocity and architectural completion rather than incremental validation.

## References

1. SMA Plugin Manager: `semantic_matrix_analyzer/semantic_matrix_analyzer/plugins/__init__.py`

2. GPU Analysis Plugin: `brain/gpu_analysis/plugin.py`

3. SMA Core Module: `semantic_matrix_analyzer/semantic_matrix_analyzer/core/__init__.py`

4. SMA CLI: `semantic_matrix_analyzer/sma_cli.py`
