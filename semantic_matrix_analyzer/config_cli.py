#!/usr/bin/env python3
"""
Command-line interface for managing Semantic Matrix Analyzer configuration.

This module provides a CLI for viewing and modifying the configuration
settings used by the Semantic Matrix Analyzer.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from semantic_matrix_analyzer.config_manager import ConfigManager

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Manage Semantic Matrix Analyzer configuration"
    )
    
    # Config file option
    parser.add_argument(
        "--config", "-c",
        help="Path to configuration file",
        type=str
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # View command
    view_parser = subparsers.add_parser("view", help="View configuration")
    view_parser.add_argument(
        "--section", "-s",
        help="Section to view (e.g., 'analysis.weights')",
        type=str
    )
    view_parser.add_argument(
        "--format", "-f",
        help="Output format (json or yaml)",
        choices=["json", "yaml"],
        default="json"
    )
    
    # Update command
    update_parser = subparsers.add_parser("update", help="Update configuration")
    update_parser.add_argument(
        "--weight", "-w",
        help="Update weight (format: name=value)",
        action="append"
    )
    update_parser.add_argument(
        "--pattern", "-p",
        help="Update pattern (format: category.name=value)",
        action="append"
    )
    update_parser.add_argument(
        "--add-token", "-at",
        help="Add token (format: category=token)",
        action="append"
    )
    update_parser.add_argument(
        "--remove-token", "-rt",
        help="Remove token (format: category=token)",
        action="append"
    )
    update_parser.add_argument(
        "--add-key", "-ak",
        help="Add key (format: category=key)",
        action="append"
    )
    update_parser.add_argument(
        "--remove-key", "-rk",
        help="Remove key (format: category=key)",
        action="append"
    )
    
    # Reset command
    reset_parser = subparsers.add_parser("reset", help="Reset configuration to defaults")
    reset_parser.add_argument(
        "--confirm",
        help="Confirm reset",
        action="store_true"
    )
    
    return parser.parse_args()

def get_nested_value(config: Dict[str, Any], section_path: str) -> Any:
    """
    Get a nested value from the configuration.
    
    Args:
        config: Configuration dictionary
        section_path: Dot-separated path to the section
        
    Returns:
        Value at the specified path
    """
    if not section_path:
        return config
        
    parts = section_path.split(".")
    current = config
    
    for part in parts:
        if part not in current:
            print(f"Error: Section '{part}' not found in configuration")
            return None
        current = current[part]
        
    return current

def view_config(config_manager: ConfigManager, args: argparse.Namespace) -> None:
    """
    View configuration.
    
    Args:
        config_manager: Configuration manager
        args: Command-line arguments
    """
    config = config_manager.get_config()
    section = get_nested_value(config, args.section) if args.section else config
    
    if section is None:
        return
        
    if args.format == "json":
        print(json.dumps(section, indent=2))
    else:
        import yaml
        print(yaml.dump(section, default_flow_style=False))

def update_config(config_manager: ConfigManager, args: argparse.Namespace) -> None:
    """
    Update configuration.
    
    Args:
        config_manager: Configuration manager
        args: Command-line arguments
    """
    # Update weights
    if args.weight:
        for weight_arg in args.weight:
            try:
                name, value = weight_arg.split("=")
                config_manager.set_weight(name, float(value))
                print(f"Updated weight '{name}' to {value}")
            except ValueError:
                print(f"Error: Invalid weight format '{weight_arg}'. Use name=value")
    
    # Update patterns
    if args.pattern:
        for pattern_arg in args.pattern:
            try:
                path, value = pattern_arg.split("=")
                category, name = path.split(".")
                
                # Try to convert value to appropriate type
                try:
                    # Try as number
                    if "." in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    # Keep as string
                    pass
                    
                config_manager.set_pattern(category, name, value)
                print(f"Updated pattern '{category}.{name}' to {value}")
            except ValueError:
                print(f"Error: Invalid pattern format '{pattern_arg}'. Use category.name=value")
    
    # Add tokens
    if args.add_token:
        for token_arg in args.add_token:
            try:
                category, token = token_arg.split("=")
                config_manager.add_token(category, token)
                print(f"Added token '{token}' to category '{category}'")
            except ValueError:
                print(f"Error: Invalid token format '{token_arg}'. Use category=token")
    
    # Remove tokens
    if args.remove_token:
        for token_arg in args.remove_token:
            try:
                category, token = token_arg.split("=")
                config_manager.remove_token(category, token)
                print(f"Removed token '{token}' from category '{category}'")
            except ValueError:
                print(f"Error: Invalid token format '{token_arg}'. Use category=token")
    
    # Add keys
    if args.add_key:
        for key_arg in args.add_key:
            try:
                category, key = key_arg.split("=")
                config_manager.add_key(category, key)
                print(f"Added key '{key}' to category '{category}'")
            except ValueError:
                print(f"Error: Invalid key format '{key_arg}'. Use category=key")
    
    # Remove keys
    if args.remove_key:
        for key_arg in args.remove_key:
            try:
                category, key = key_arg.split("=")
                config_manager.remove_key(category, key)
                print(f"Removed key '{key}' from category '{category}'")
            except ValueError:
                print(f"Error: Invalid key format '{key_arg}'. Use category=key")

def reset_config(config_manager: ConfigManager, args: argparse.Namespace) -> None:
    """
    Reset configuration to defaults.
    
    Args:
        config_manager: Configuration manager
        args: Command-line arguments
    """
    if not args.confirm:
        print("Warning: This will reset all configuration to defaults.")
        print("Use --confirm to proceed.")
        return
        
    config_manager.reset_to_defaults()
    print("Configuration reset to defaults.")

def main():
    """Main entry point."""
    args = parse_args()
    
    # Create config manager
    config_manager = ConfigManager(args.config)
    
    # Execute command
    if args.command == "view":
        view_config(config_manager, args)
    elif args.command == "update":
        update_config(config_manager, args)
    elif args.command == "reset":
        reset_config(config_manager, args)
    else:
        print("Error: No command specified")
        print("Use one of: view, update, reset")
        sys.exit(1)

if __name__ == "__main__":
    main()
