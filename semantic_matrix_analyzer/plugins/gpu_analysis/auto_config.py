"""
Auto-Configuration Module for GPU Analysis Plugin.

This module provides functionality for generating auto-configuration
based on project analysis, enabling adaptive analysis and continuous improvement.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union
import logging
import os
from pathlib import Path
import torch

# Configure logging
logger = logging.getLogger(__name__)

def generate_auto_config(project_dir: Union[str, Path], context: Optional[Any] = None) -> Dict[str, Any]:
    """
    Generate auto-configuration for a project.

    This function analyzes a project and generates an initial configuration
    based on the project's characteristics, which can then be refined through
    human feedback.

    Args:
        project_dir: Path to the project directory
        context: Optional plugin context for logging

    Returns:
        Auto-generated configuration dictionary
    """
    try:
        # Initialize logger
        def log(level: str, message: str) -> None:
            """Log a message using the context logger if available."""
            if context and hasattr(context, 'log'):
                context.log(level, message)
            else:
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

        log("info", f"Generating auto-configuration for project: {project_dir}")

        # Convert project_dir to Path
        project_dir = Path(project_dir)

        # Start with default configuration
        config = {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "batch_size": 32,
            "precision": "float32",
            "cache_size": 1024,
            "analyzers": {
                "complexity": {
                    "enabled": True,
                    "weights": {
                        "cyclomatic": 1.0,
                        "cognitive": 1.0,
                        "halstead": 0.5
                    }
                },
                "dependency": {
                    "enabled": True,
                    "max_depth": 3
                },
                "semantic": {
                    "enabled": True,
                    "embedding_model": "default",
                    "similarity_threshold": 0.7
                },
                "pattern": {
                    "enabled": True,
                    "confidence_threshold": 0.6
                }
            },
            "patterns": [],
            "intents": []
        }

        # Analyze project to customize configuration
        # Count Python files
        python_files = []
        for root, _, files in os.walk(project_dir):
            for file in files:
                if file.endswith(".py"):
                    python_files.append(os.path.join(root, file))

        # Adjust batch size based on project size
        num_files = len(python_files)
        if num_files < 10:
            config["batch_size"] = 16
        elif num_files < 50:
            config["batch_size"] = 32
        elif num_files < 200:
            config["batch_size"] = 64
        else:
            config["batch_size"] = 128

        # Adjust cache size based on project size
        if num_files < 10:
            config["cache_size"] = 256
        elif num_files < 50:
            config["cache_size"] = 512
        elif num_files < 200:
            config["cache_size"] = 1024
        else:
            config["cache_size"] = 2048

        # Extract common patterns from project
        patterns = extract_patterns_from_project(project_dir, context)
        config["patterns"] = patterns

        # Extract intents from project documentation
        intents = extract_intents_from_project(project_dir, context)
        config["intents"] = intents

        log("info", f"Auto-configuration generated for project: {project_dir}")
        return config
    except Exception as e:
        logger.error(f"Error generating auto-configuration: {e}")
        # Return default configuration as fallback
        return {
            "device": "cpu",
            "batch_size": 32,
            "precision": "float32",
            "cache_size": 1024,
            "analyzers": {
                "complexity": {"enabled": True},
                "dependency": {"enabled": True},
                "semantic": {"enabled": True},
                "pattern": {"enabled": True}
            },
            "patterns": [],
            "intents": []
        }

def extract_patterns_from_project(project_dir: Union[str, Path], context: Optional[Any] = None) -> List[Dict[str, Any]]:
    """
    Extract patterns from a project.

    This function analyzes a project and extracts common patterns that can be
    used for pattern matching in the GPU Analysis Plugin.

    Args:
        project_dir: Path to the project directory
        context: Optional plugin context for logging

    Returns:
        List of pattern dictionaries
    """
    try:
        # Initialize logger
        def log(level: str, message: str) -> None:
            """Log a message using the context logger if available."""
            if context and hasattr(context, 'log'):
                context.log(level, message)
            else:
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

        log("info", f"Extracting patterns from project: {project_dir}")

        # Convert project_dir to Path
        project_dir = Path(project_dir)

        # Find Python files
        python_files = []
        for root, _, files in os.walk(project_dir):
            for file in files:
                if file.endswith(".py"):
                    python_files.append(Path(root) / file)

        # Extract patterns from each file
        patterns = []
        for file_path in python_files[:10]:  # Limit to first 10 files for performance
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    code = f.read()

                # Extract patterns from code
                file_patterns = extract_patterns_from_code(code, str(file_path))
                patterns.extend(file_patterns)
            except Exception as e:
                log("warning", f"Error extracting patterns from {file_path}: {e}")

        log("info", f"Extracted {len(patterns)} patterns from project")
        return patterns
    except Exception as e:
        logger.error(f"Error extracting patterns from project: {e}")
        return []

def extract_patterns_from_code(code: str, file_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Extract patterns from code.

    This function analyzes code and extracts common patterns that can be
    used for pattern matching in the GPU Analysis Plugin.

    Args:
        code: Code to extract patterns from
        file_path: Optional file path for context

    Returns:
        List of pattern dictionaries
    """
    try:
        # Parse code into AST
        import ast
        try:
            code_ast = ast.parse(code)
        except Exception as e:
            logger.warning(f"Error parsing code into AST: {e}")
            return []

        # Extract patterns
        patterns = []

        # Extract function patterns
        for node in ast.walk(code_ast):
            if isinstance(node, ast.FunctionDef):
                # Extract function pattern
                function_name = node.name
                args = [arg.arg for arg in node.args.args]

                # Create pattern
                pattern = {
                    "name": f"function_{function_name}",
                    "description": f"Function {function_name} with arguments {', '.join(args)}",
                    "type": "ast",
                    "pattern": ast.unparse(node),
                    "weight": 1.0,
                    "source": file_path
                }

                patterns.append(pattern)
            elif isinstance(node, ast.ClassDef):
                # Extract class pattern
                class_name = node.name

                # Create pattern
                pattern = {
                    "name": f"class_{class_name}",
                    "description": f"Class {class_name}",
                    "type": "ast",
                    "pattern": ast.unparse(node),
                    "weight": 1.0,
                    "source": file_path
                }

                patterns.append(pattern)

        return patterns
    except Exception as e:
        logger.error(f"Error extracting patterns from code: {e}")
        return []

def extract_intents_from_project(project_dir: Union[str, Path], context: Optional[Any] = None) -> List[Dict[str, Any]]:
    """
    Extract intents from a project.

    This function analyzes a project and extracts intents from documentation
    that can be used for intent alignment in the GPU Analysis Plugin.

    Args:
        project_dir: Path to the project directory
        context: Optional plugin context for logging

    Returns:
        List of intent dictionaries
    """
    try:
        # Initialize logger
        def log(level: str, message: str) -> None:
            """Log a message using the context logger if available."""
            if context and hasattr(context, 'log'):
                context.log(level, message)
            else:
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

        log("info", f"Extracting intents from project: {project_dir}")

        # Convert project_dir to Path
        project_dir = Path(project_dir)

        # Find documentation files
        doc_files = []
        for root, _, files in os.walk(project_dir):
            for file in files:
                if file.endswith((".md", ".rst", ".txt")):
                    doc_files.append(Path(root) / file)

        # Extract intents from each file
        intents = []
        for file_path in doc_files[:10]:  # Limit to first 10 files for performance
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()

                # Extract intents from text
                file_intents = extract_intents_from_text(text, str(file_path))
                intents.extend(file_intents)
            except Exception as e:
                log("warning", f"Error extracting intents from {file_path}: {e}")

        log("info", f"Extracted {len(intents)} intents from project")
        return intents
    except Exception as e:
        logger.error(f"Error extracting intents from project: {e}")
        return []

def extract_intents_from_text(text: str, source: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Extract intents from text.

    This function analyzes text and extracts intents that can be
    used for intent alignment in the GPU Analysis Plugin.

    Args:
        text: Text to extract intents from
        source: Optional source for context

    Returns:
        List of intent dictionaries
    """
    try:
        # Extract intents
        intents = []

        # Extract intents from headings
        import re
        heading_pattern = re.compile(r'#+\s+(.*)')
        headings = heading_pattern.findall(text)

        for heading in headings:
            # Create intent
            intent = {
                "name": f"intent_{heading.lower().replace(' ', '_')}",
                "description": heading,
                "patterns": [],
                "keywords": heading.lower().split(),
                "weight": 1.0,
                "source": source
            }

            intents.append(intent)

        # Extract intents from code blocks
        code_block_pattern = re.compile(r'```(?:python)?\s+(.*?)\s+```', re.DOTALL)
        code_blocks = code_block_pattern.findall(text)

        for i, code_block in enumerate(code_blocks):
            # Create intent
            intent = {
                "name": f"intent_code_block_{i}",
                "description": f"Code block {i}",
                "patterns": [],
                "keywords": [],
                "weight": 1.0,
                "source": source
            }

            intents.append(intent)

        return intents
    except Exception as e:
        logger.error(f"Error extracting intents from text: {e}")
        return []
