#!/usr/bin/env python3
"""
Example of using the Type Hint Analyzer to extract intent from type hints.
"""

import ast
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Add the semantic_matrix_analyzer directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "semantic_matrix_analyzer"))

from intent.config.configuration import Configuration
from intent.analyzers.type_analyzer import TypeHintExtractor, ConfigurableTypeHintAnalyzer


def setup_logging():
    """Set up logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def analyze_type_hints():
    """Analyze some example type hints."""
    # Create a configuration
    config = Configuration()
    
    # Create a type hint analyzer
    analyzer = ConfigurableTypeHintAnalyzer(config)
    
    # Example parameter type hints
    parameter_types = [
        ("user_id", "int"),
        ("name", "str"),
        ("is_active", "bool"),
        ("items", "List[str]"),
        ("user", "Optional[User]"),
        ("data", "Dict[str, Any]"),
        ("callback", "Callable[[int], bool]"),
        ("values", "Union[int, float, str]"),
        ("config", "Optional[Dict[str, Any]]"),
        ("path", "Path")
    ]
    
    # Example return type hints
    return_types = [
        "None",
        "bool",
        "str",
        "int",
        "List[User]",
        "Optional[Dict[str, Any]]",
        "Tuple[int, str, bool]",
        "Union[str, None]",
        "Iterator[str]",
        "Any"
    ]
    
    # Analyze parameter type hints
    print("\nParameter Type Hints:")
    print("-" * 80)
    for param_name, type_hint in parameter_types:
        intent = analyzer.analyze_parameter_type(param_name, type_hint)
        print(f"{param_name}: {type_hint:30} -> {intent.name} ({intent.type.value}, {intent.confidence:.2f})")
        print(f"{' ':38}    {intent.description}")
    
    # Analyze return type hints
    print("\nReturn Type Hints:")
    print("-" * 80)
    for type_hint in return_types:
        intent = analyzer.analyze_return_type(type_hint)
        print(f"{type_hint:30} -> {intent.name} ({intent.type.value}, {intent.confidence:.2f})")
        print(f"{' ':30}    {intent.description}")


def analyze_code_with_type_hints():
    """Analyze type hints in a code snippet."""
    # Example code with type hints
    code = """
class User:
    def __init__(self, id: int, name: str, email: str):
        self.id = id
        self.name = name
        self.email = email

class UserRepository:
    def get_user(self, id: int) -> Optional[User]:
        # Get a user by ID
        pass
    
    def save_user(self, user: User) -> bool:
        # Save a user
        pass
    
    def delete_user(self, id: int) -> bool:
        # Delete a user
        pass
    
    def get_all_users(self) -> List[User]:
        # Get all users
        pass

def process_data(data: Dict[str, Any], callback: Callable[[Dict[str, Any]], bool] = None) -> Tuple[bool, Optional[str]]:
    # Process data
    pass
"""
    
    # Parse the code
    tree = ast.parse(code)
    
    # Create a type hint extractor
    extractor = TypeHintExtractor()
    
    # Create a type hint analyzer
    config = Configuration()
    analyzer = ConfigurableTypeHintAnalyzer(config)
    
    # Extract and analyze type hints
    print("\nType Hints in Code:")
    print("-" * 80)
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Get function name
            function_name = node.name
            
            # Extract type hints
            type_hints = extractor.extract_type_hints_from_ast(node)
            
            if type_hints:
                print(f"\nFunction: {function_name}")
                
                # Analyze parameter type hints
                for param_name, type_hint in type_hints.items():
                    if param_name != "return":
                        intent = analyzer.analyze_parameter_type(param_name, type_hint)
                        print(f"  Parameter: {param_name}: {type_hint:20} -> {intent.name} ({intent.type.value}, {intent.confidence:.2f})")
                        print(f"  {' ':33}    {intent.description}")
                
                # Analyze return type hint
                if "return" in type_hints:
                    intent = analyzer.analyze_return_type(type_hints["return"])
                    print(f"  Return: {type_hints['return']:20} -> {intent.name} ({intent.type.value}, {intent.confidence:.2f})")
                    print(f"  {' ':33}    {intent.description}")


def main():
    """Main entry point for the example."""
    setup_logging()
    analyze_type_hints()
    analyze_code_with_type_hints()


if __name__ == "__main__":
    main()
