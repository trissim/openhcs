"""
Generic intent plugin for Semantic Matrix Analyzer.

This plugin provides generic software engineering intents.
"""

from typing import List

from semantic_matrix_analyzer.patterns import Intent
from semantic_matrix_analyzer.plugins import IntentPlugin


class GenericIntentPlugin(IntentPlugin):
    """Generic intent plugin with common software engineering intents."""
    
    @property
    def name(self) -> str:
        """Get the name of the plugin."""
        return "generic_intent_plugin"
    
    @property
    def version(self) -> str:
        """Get the version of the plugin."""
        return "0.1.0"
    
    @property
    def description(self) -> str:
        """Get the description of the plugin."""
        return "Generic intent plugin with common software engineering intents."
    
    @staticmethod
    def get_intents() -> List[Intent]:
        """Get generic software engineering intents."""
        intents = []
        
        # Immutability intent
        immutability = Intent(
            name="Immutability",
            description="Using immutable data structures and patterns"
        )
        immutability.add_string_pattern(
            name="immutable_pattern",
            description="Reference to immutability",
            pattern="immutable",
            weight=1.0
        )
        immutability.add_string_pattern(
            name="frozen_pattern",
            description="Using frozen dataclasses",
            pattern="frozen",
            weight=0.8
        )
        immutability.add_regex_pattern(
            name="frozen_dataclass_pattern",
            description="Defining frozen dataclasses",
            pattern=r"@dataclass\s*\(\s*frozen\s*=\s*True\s*\)",
            weight=1.0
        )
        immutability.add_string_pattern(
            name="readonly_pattern",
            description="Using readonly properties",
            pattern="readonly",
            weight=0.7
        )
        immutability.add_string_pattern(
            name="const_pattern",
            description="Using const variables",
            pattern="const ",
            weight=0.7
        )
        immutability.add_regex_pattern(
            name="final_pattern",
            description="Using final variables",
            pattern=r"final\s+[a-zA-Z_][a-zA-Z0-9_]*",
            weight=0.7
        )
        intents.append(immutability)
        
        # Dependency Injection intent
        di = Intent(
            name="Dependency Injection",
            description="Using dependency injection patterns"
        )
        di.add_string_pattern(
            name="inject_pattern",
            description="Reference to injection",
            pattern="inject",
            weight=1.0
        )
        di.add_string_pattern(
            name="dependency_injection_pattern",
            description="Reference to dependency injection",
            pattern="dependency injection",
            weight=1.0
        )
        di.add_string_pattern(
            name="di_pattern",
            description="Reference to DI",
            pattern="DI",
            weight=0.5
        )
        di.add_regex_pattern(
            name="constructor_injection_pattern",
            description="Constructor injection pattern",
            pattern=r"def\s+__init__\s*\(\s*self\s*,\s*[^)]*\)\s*:",
            weight=0.5
        )
        intents.append(di)
        
        # Factory Pattern intent
        factory = Intent(
            name="Factory Pattern",
            description="Using factory patterns for object creation"
        )
        factory.add_string_pattern(
            name="factory_pattern",
            description="Reference to factory pattern",
            pattern="factory",
            weight=1.0
        )
        factory.add_string_pattern(
            name="create_pattern",
            description="Methods for creating objects",
            pattern="create_",
            weight=0.7
        )
        factory.add_regex_pattern(
            name="factory_method_pattern",
            description="Factory method pattern",
            pattern=r"@classmethod\s+def\s+create",
            weight=0.8
        )
        intents.append(factory)
        
        # Singleton Pattern intent
        singleton = Intent(
            name="Singleton Pattern",
            description="Using singleton pattern for single instances"
        )
        singleton.add_string_pattern(
            name="singleton_pattern",
            description="Reference to singleton pattern",
            pattern="singleton",
            weight=1.0
        )
        singleton.add_regex_pattern(
            name="instance_check_pattern",
            description="Checking for existing instance",
            pattern=r"if\s+cls\._instance\s+is\s+None",
            weight=0.8
        )
        singleton.add_regex_pattern(
            name="private_constructor_pattern",
            description="Private constructor pattern",
            pattern=r"_[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*None",
            weight=0.6
        )
        intents.append(singleton)
        
        # Observer Pattern intent
        observer = Intent(
            name="Observer Pattern",
            description="Using observer pattern for event handling"
        )
        observer.add_string_pattern(
            name="observer_pattern",
            description="Reference to observer pattern",
            pattern="observer",
            weight=1.0
        )
        observer.add_string_pattern(
            name="subscribe_pattern",
            description="Methods for subscribing to events",
            pattern="subscribe",
            weight=0.8
        )
        observer.add_string_pattern(
            name="notify_pattern",
            description="Methods for notifying observers",
            pattern="notify",
            weight=0.8
        )
        observer.add_string_pattern(
            name="listener_pattern",
            description="Reference to listeners",
            pattern="listener",
            weight=0.7
        )
        observer.add_string_pattern(
            name="event_pattern",
            description="Reference to events",
            pattern="event",
            weight=0.6
        )
        intents.append(observer)
        
        # Strategy Pattern intent
        strategy = Intent(
            name="Strategy Pattern",
            description="Using strategy pattern for algorithm selection"
        )
        strategy.add_string_pattern(
            name="strategy_pattern",
            description="Reference to strategy pattern",
            pattern="strategy",
            weight=1.0
        )
        strategy.add_regex_pattern(
            name="abstract_method_pattern",
            description="Abstract method pattern",
            pattern=r"@abstractmethod",
            weight=0.7
        )
        intents.append(strategy)
        
        return intents
