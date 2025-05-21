#!/usr/bin/env python3
"""
Example of using the Name Analyzer to extract intent from names.
"""

import logging
import sys
from pathlib import Path

# Add the semantic_matrix_analyzer directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "semantic_matrix_analyzer"))

from intent.config.configuration import Configuration
from intent.analyzers.name_analyzer import ConfigurableNameAnalyzer


def setup_logging():
    """Set up logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def analyze_names():
    """Analyze some example names."""
    # Create a configuration
    config = Configuration()

    # Create a name analyzer
    analyzer = ConfigurableNameAnalyzer(config)

    # Example class names
    class_names = [
        "UserRepository",
        "OrderService",
        "PaymentProcessor",
        "CustomerFactory",
        "EventListener",
        "DataValidator",
        "ConfigManager",
        "ApiClient",
        "DatabaseConnection",
        "AuthenticationProvider"
    ]

    # Example method names
    method_names = [
        "get_user",
        "create_order",
        "process_payment",
        "validate_data",
        "handle_event",
        "is_valid",
        "has_permission",
        "can_access",
        "should_retry",
        "calculate_total"
    ]

    # Example variable names
    variable_names = [
        "user_id",
        "order_status",
        "payment_amount",
        "is_active",
        "has_children",
        "max_retries",
        "current_user",
        "total_items",
        "error_message",
        "config_options"
    ]

    # Analyze class names
    print("\nClass Names:")
    print("-" * 80)
    for name in class_names:
        intent = analyzer.analyze_class_name(name)
        print(f"{name:20} -> {intent.name} ({intent.type.value}, {intent.confidence:.2f})")
        print(f"{' ':20}    {intent.description}")

    # Analyze method names
    print("\nMethod Names:")
    print("-" * 80)
    for name in method_names:
        intent = analyzer.analyze_method_name(name)
        print(f"{name:20} -> {intent.name} ({intent.type.value}, {intent.confidence:.2f})")
        print(f"{' ':20}    {intent.description}")

    # Analyze variable names
    print("\nVariable Names:")
    print("-" * 80)
    for name in variable_names:
        intent = analyzer.analyze_variable_name(name)
        print(f"{name:20} -> {intent.name} ({intent.type.value}, {intent.confidence:.2f})")
        print(f"{' ':20}    {intent.description}")


def main():
    """Main entry point for the example."""
    setup_logging()
    analyze_names()


if __name__ == "__main__":
    main()
