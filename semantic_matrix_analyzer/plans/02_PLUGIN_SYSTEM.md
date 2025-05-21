# Plan 02: Plugin System Enhancement

## Objective

Create a robust, extensible plugin architecture for the Semantic Matrix Analyzer that allows for easy addition of new capabilities, pattern types, and integrations without modifying the core codebase.

## Rationale

A powerful plugin system is essential for:

1. **Universal Applicability**: Different projects have unique requirements and patterns
2. **Reduced Cognitive Load**: Users can install plugins rather than learning how to create custom patterns
3. **Community Contributions**: Enables experts to share knowledge through plugins
4. **Framework-Specific Analysis**: Allows specialized analysis for popular frameworks
5. **Future-Proofing**: New languages, frameworks, and patterns can be added without core changes

## Implementation Details

### 1. Plugin Interface

Define a comprehensive plugin interface that allows plugins to extend various aspects of the system:

```python
class SMAPlugin:
    """Base class for all SMA plugins."""
    
    @property
    def name(self) -> str:
        """Get the name of the plugin."""
        raise NotImplementedError
    
    @property
    def version(self) -> str:
        """Get the version of the plugin."""
        raise NotImplementedError
    
    @property
    def description(self) -> str:
        """Get the description of the plugin."""
        raise NotImplementedError
    
    def initialize(self, context: 'PluginContext') -> None:
        """Initialize the plugin with the given context."""
        pass
    
    def shutdown(self) -> None:
        """Perform cleanup when the plugin is being unloaded."""
        pass
```

### 2. Plugin Types

Create specialized interfaces for different types of plugins:

#### 2.1 Intent Plugin (Already Implemented)

```python
class IntentPlugin(SMAPlugin):
    """Plugin for defining intents and patterns."""
    
    @staticmethod
    def get_intents() -> List[Intent]:
        """Get the intents defined by this plugin."""
        return []
```

#### 2.2 Language Plugin

```python
class LanguagePlugin(SMAPlugin):
    """Plugin for adding support for a programming language."""
    
    def get_parser(self) -> LanguageParser:
        """Get the language parser provided by this plugin."""
        raise NotImplementedError
```

#### 2.3 Framework Plugin

```python
class FrameworkPlugin(SMAPlugin):
    """Plugin for adding support for a specific framework."""
    
    @property
    def framework_name(self) -> str:
        """Get the name of the framework."""
        raise NotImplementedError
    
    def detect_framework(self, project_dir: Path) -> bool:
        """Detect if the framework is used in the project."""
        raise NotImplementedError
    
    def get_intents(self) -> List[Intent]:
        """Get framework-specific intents."""
        return []
```

#### 2.4 Integration Plugin

```python
class IntegrationPlugin(SMAPlugin):
    """Plugin for integrating with external systems."""
    
    @property
    def integration_type(self) -> str:
        """Get the type of integration (e.g., 'vcs', 'ci', 'issue-tracker')."""
        raise NotImplementedError
    
    def initialize_integration(self, config: Dict[str, Any]) -> None:
        """Initialize the integration with the given configuration."""
        raise NotImplementedError
```

### 3. Plugin Discovery and Loading

Implement a robust system for discovering and loading plugins:

```python
class PluginManager:
    """Manages the discovery, loading, and lifecycle of plugins."""
    
    def __init__(self):
        self.plugins = {}
        self.intent_plugins = []
        self.language_plugins = []
        self.framework_plugins = []
        self.integration_plugins = []
    
    def discover_plugins(self) -> None:
        """Discover available plugins."""
        # Look in standard plugin directories
        # Look in user-specified plugin directories
        # Look for installed Python packages with SMA plugin entry points
        pass
    
    def load_plugin(self, plugin_path: Path) -> Optional[SMAPlugin]:
        """Load a plugin from the given path."""
        # Import the plugin module
        # Find plugin classes
        # Instantiate the plugin
        # Register the plugin
        pass
    
    def initialize_plugins(self, context: 'PluginContext') -> None:
        """Initialize all loaded plugins."""
        for plugin in self.plugins.values():
            plugin.initialize(context)
    
    def get_plugin_by_name(self, name: str) -> Optional[SMAPlugin]:
        """Get a plugin by name."""
        return self.plugins.get(name)
    
    def get_intent_plugins(self) -> List[IntentPlugin]:
        """Get all intent plugins."""
        return self.intent_plugins
    
    def get_language_plugins(self) -> List[LanguagePlugin]:
        """Get all language plugins."""
        return self.language_plugins
    
    def get_framework_plugins(self) -> List[FrameworkPlugin]:
        """Get all framework plugins."""
        return self.framework_plugins
    
    def get_integration_plugins(self) -> List[IntegrationPlugin]:
        """Get all integration plugins."""
        return self.integration_plugins
```

### 4. Plugin Context

Create a context object that provides plugins with access to the SMA system:

```python
class PluginContext:
    """Context provided to plugins for interacting with the SMA system."""
    
    def __init__(self, sma_instance: 'SemanticMatrixAnalyzer'):
        self.sma = sma_instance
    
    def register_intent(self, intent: Intent) -> None:
        """Register an intent with the SMA system."""
        self.sma.register_intent(intent)
    
    def register_language_parser(self, parser: LanguageParser) -> None:
        """Register a language parser with the SMA system."""
        self.sma.register_language_parser(parser)
    
    def log(self, level: str, message: str) -> None:
        """Log a message through the SMA logging system."""
        self.sma.log(level, message)
```

### 5. Plugin Configuration

Implement a configuration system for plugins:

```python
class PluginConfig:
    """Configuration for a plugin."""
    
    def __init__(self, plugin_name: str, config_data: Dict[str, Any]):
        self.plugin_name = plugin_name
        self.config_data = config_data
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self.config_data.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value."""
        self.config_data[key] = value
    
    def save(self) -> None:
        """Save the configuration."""
        # Save to a configuration file
        pass
```

### 6. Plugin Distribution

Create a system for packaging and distributing plugins:

1. Define a standard plugin package structure
2. Create tools for building plugin packages
3. Implement a plugin repository system
4. Add commands for installing plugins from the repository

## Success Criteria

1. At least 5 different types of plugins supported
2. Plugin discovery from multiple sources (directories, Python packages)
3. Plugin configuration system with persistence
4. Documentation for creating and distributing plugins
5. At least 3 example plugins (one of each type)

## Dependencies

- None (this is a foundational plan)

## Timeline

- Research and design: 1 week
- Plugin interface implementation: 1 week
- Plugin types implementation: 2 weeks
- Plugin discovery and loading: 1 week
- Plugin context and configuration: 1 week
- Plugin distribution system: 2 weeks
- Documentation and examples: 1 week

Total: 9 weeks
