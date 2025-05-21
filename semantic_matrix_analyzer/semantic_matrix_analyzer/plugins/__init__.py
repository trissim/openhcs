"""
Plugin system for Semantic Matrix Analyzer.

This module provides a plugin system for extending the functionality of the analyzer.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type

from semantic_matrix_analyzer.patterns import Intent


class SMAPlugin(ABC):
    """Base class for all SMA plugins.

    All plugins must inherit from this class and implement its abstract methods.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the plugin."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Get the version of the plugin."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Get the description of the plugin."""
        pass

    def initialize(self, context: 'PluginContext') -> None:
        """Initialize the plugin with the given context.

        Args:
            context: The plugin context.
        """
        pass

    def shutdown(self) -> None:
        """Perform cleanup when the plugin is being unloaded."""
        pass


class IntentPlugin(SMAPlugin):
    """Plugin for defining intents and patterns.

    Intent plugins provide intents and patterns for detecting code quality issues.
    """

    @staticmethod
    def get_intents() -> List[Intent]:
        """Get the intents defined by this plugin.

        Returns:
            A list of intents defined by this plugin.
        """
        return []


@dataclass
class PluginContext:
    """Context provided to plugins for interacting with the SMA system."""

    def register_intent(self, intent: Intent) -> None:
        """Register an intent with the SMA system.

        Args:
            intent: The intent to register.
        """
        from semantic_matrix_analyzer.core import intent_registry
        intent_registry.register_intent(intent)

    def log(self, level: str, message: str) -> None:
        """Log a message through the SMA logging system.

        Args:
            level: The log level ("debug", "info", "warning", "error").
            message: The message to log.
        """
        import logging
        logger = logging.getLogger("sma")

        if level == "debug":
            logger.debug(message)
        elif level == "info":
            logger.info(message)
        elif level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
        else:
            logger.info(message)


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

    def initialize_plugins(self) -> None:
        """Initialize all loaded plugins."""
        context = PluginContext()

        for plugin in self.plugins.values():
            try:
                plugin.initialize(context)
            except Exception as e:
                import logging
                logger = logging.getLogger("sma")
                logger.error(f"Error initializing plugin {plugin.name}: {e}")

    def shutdown_plugins(self) -> None:
        """Shutdown all loaded plugins."""
        for plugin in self.plugins.values():
            try:
                plugin.shutdown()
            except Exception as e:
                import logging
                logger = logging.getLogger("sma")
                logger.error(f"Error shutting down plugin {plugin.name}: {e}")

    def get_plugin_by_name(self, name: str) -> Optional[SMAPlugin]:
        """Get a plugin by name.

        Args:
            name: The name of the plugin.

        Returns:
            The plugin, or None if not found.
        """
        return self.plugins.get(name)

    def get_intent_plugins(self) -> List[IntentPlugin]:
        """Get all intent plugins.

        Returns:
            A list of all intent plugins.
        """
        return self.intent_plugins.copy()


# Global plugin manager
plugin_manager = PluginManager()
