# Creating Custom Plugins for Semantic Matrix Analyzer

This guide explains how to create custom plugins for the Semantic Matrix Analyzer tool.

## Overview

Plugins allow you to extend the Semantic Matrix Analyzer with custom intents and patterns. This is useful for:

- Adding domain-specific intents
- Supporting specific frameworks or libraries
- Implementing custom pattern detection logic
- Integrating with other tools or data sources

## Basic Plugin Structure

A plugin is a Python class that inherits from `IntentPlugin` and implements the `get_intents` method:

```python
from semantic_matrix_analyzer import IntentPlugin, Intent

class MyCustomPlugin(IntentPlugin):
    @staticmethod
    def get_intents() -> List[Intent]:
        """Get intents defined by this plugin."""
        intents = []
        
        # Create and configure intents
        my_intent = Intent(
            name="My Custom Intent",
            description="Description of my custom intent"
        )
        
        # Add patterns to the intent
        my_intent.add_string_pattern(
            name="pattern1",
            description="Description of pattern1",
            pattern="pattern1",
            weight=1.0
        )
        
        intents.append(my_intent)
        return intents
```

## Plugin Installation

Plugins should be placed in the `plugins` directory with a filename ending in `_plugin.py`. The Semantic Matrix Analyzer will automatically discover and load these plugins.

## Pattern Types

Plugins can define several types of patterns:

### String Patterns

String patterns check for the presence of a specific string in the code:

```python
intent.add_string_pattern(
    name="repository_pattern",
    description="Using repository pattern",
    pattern="Repository",
    weight=1.0,
    is_negative=False
)
```

### Regex Patterns

Regex patterns use regular expressions to match more complex patterns:

```python
intent.add_regex_pattern(
    name="constructor_injection_pattern",
    description="Constructor injection pattern",
    pattern=r"def\s+__init__\s*\(\s*self\s*,\s*[^)]*\)\s*:",
    weight=0.8,
    is_negative=False
)
```

### AST Patterns

AST patterns check for specific AST node types and conditions:

```python
def is_immutable_class(node):
    """Check if a class is immutable."""
    if not isinstance(node, ast.ClassDef):
        return False
    
    # Check for @dataclass(frozen=True)
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name) and decorator.func.id == 'dataclass':
            for keyword in decorator.keywords:
                if keyword.arg == 'frozen' and isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
                    return True
    
    return False

intent.add_ast_pattern(
    name="immutable_class_pattern",
    description="Using immutable classes",
    node_type=ast.ClassDef,
    condition=is_immutable_class,
    weight=1.0,
    is_negative=False
)
```

### Annotation Patterns

Annotation patterns check for special annotations in comments or docstrings:

```python
intent.add_annotation_pattern(
    name="intent_annotation_pattern",
    description="Using intent annotations",
    annotation="clean_code",
    weight=1.0,
    is_negative=False
)
```

## Negative Patterns

Negative patterns reduce the intent alignment score when they are found:

```python
intent.add_string_pattern(
    name="global_state_pattern",
    description="Using global state",
    pattern="global ",
    weight=1.0,
    is_negative=True
)
```

## Pattern Weights

Patterns can have different weights to indicate their importance:

```python
# Critical pattern (high weight)
intent.add_string_pattern(
    name="critical_pattern",
    description="Critical pattern",
    pattern="critical_pattern",
    weight=2.0
)

# Normal pattern (default weight)
intent.add_string_pattern(
    name="normal_pattern",
    description="Normal pattern",
    pattern="normal_pattern",
    weight=1.0
)

# Minor pattern (low weight)
intent.add_string_pattern(
    name="minor_pattern",
    description="Minor pattern",
    pattern="minor_pattern",
    weight=0.5
)
```

## Example: Framework-Specific Plugin

Here's an example of a plugin for a specific framework:

```python
class DjangoPlugin(IntentPlugin):
    @staticmethod
    def get_intents() -> List[Intent]:
        intents = []
        
        # Django models intent
        models_intent = Intent(
            name="Django Models",
            description="Using Django models correctly"
        )
        
        models_intent.add_string_pattern(
            name="model_class_pattern",
            description="Defining Django models",
            pattern="models.Model",
            weight=1.0
        )
        
        models_intent.add_regex_pattern(
            name="field_definition_pattern",
            description="Defining model fields",
            pattern=r"[a-zA-Z_]+\s*=\s*models\.[A-Za-z]+Field\(",
            weight=0.8
        )
        
        models_intent.add_string_pattern(
            name="raw_sql_pattern",
            description="Using raw SQL instead of ORM",
            pattern="raw",
            weight=1.0,
            is_negative=True
        )
        
        intents.append(models_intent)
        
        # Django views intent
        views_intent = Intent(
            name="Django Views",
            description="Using Django views correctly"
        )
        
        views_intent.add_string_pattern(
            name="class_based_view_pattern",
            description="Using class-based views",
            pattern="View",
            weight=1.0
        )
        
        views_intent.add_regex_pattern(
            name="function_based_view_pattern",
            description="Using function-based views",
            pattern=r"def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(\s*request\s*,",
            weight=0.8
        )
        
        intents.append(views_intent)
        
        return intents
```

## Advanced: Custom Pattern Types

You can create custom pattern types by extending the `IntentDetector` class:

1. Add a new pattern type to `IntentPattern`
2. Add a new method to `IntentDetector` to check for this pattern type
3. Update the `_check_pattern` method to call your new method

For example, to add a pattern type that checks for specific import patterns:

```python
# Add a new method to IntentDetector
def _check_import_pattern(self, pattern: str, analysis: ComponentAnalysis) -> float:
    """Check if a specific import pattern is present."""
    if not analysis.ast_node:
        return 0.0
    
    for node in ast.walk(analysis.ast_node):
        if isinstance(node, ast.Import):
            for name in node.names:
                if pattern in name.name:
                    return 1.0
        elif isinstance(node, ast.ImportFrom):
            if pattern in node.module:
                return 1.0
            for name in node.names:
                if pattern in name.name:
                    return 1.0
    
    return 0.0

# Update _check_pattern method
def _check_pattern(self, pattern: IntentPattern, analysis: ComponentAnalysis) -> float:
    """Check if a pattern is present in a component analysis."""
    if pattern.pattern_type == "import":
        return self._check_import_pattern(pattern.pattern, analysis)
    elif pattern.pattern_type == "string":
        return self._check_string_pattern(pattern.pattern, analysis)
    # ... other pattern types ...
```

Then you can use your new pattern type in your plugin:

```python
intent.add_pattern(IntentPattern(
    name="framework_import_pattern",
    description="Importing the framework",
    pattern_type="import",
    pattern="django",
    weight=1.0
))
```

## Testing Your Plugin

To test your plugin:

1. Place it in the `plugins` directory
2. Run the Semantic Matrix Analyzer with the `--intents` option including your custom intents
3. Check the report to see if your patterns are being detected correctly

## Best Practices

1. **Be Specific**: Define clear, specific patterns that accurately represent your intents
2. **Use Negative Patterns**: Define both positive and negative patterns for each intent
3. **Weight Appropriately**: Give higher weights to more important patterns
4. **Document**: Include clear descriptions for all intents and patterns
5. **Test**: Test your plugin on both good and bad code examples
6. **Combine Pattern Types**: Use a combination of string, regex, and AST patterns for more accurate detection
