#!/usr/bin/env python3
"""
Extract Intents from Conversations

This script extracts intents from conversation text and saves them to a JSON file
that can be used by the semantic_matrix_analyzer.py tool.

Usage:
    python extract_intents.py --input conversation.txt --output conversations/my_intents.json
    python extract_intents.py --text "Intent: Clean Code..." --output conversations/my_intents.json
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

from semantic_matrix_analyzer.conversation import ConversationIntentExtractor


def setup_logging(verbose: bool = False) -> None:
    """Set up logging.

    Args:
        verbose: Whether to enable verbose logging.
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

logger = logging.getLogger("extract_intents")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Extract intents from conversations")

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", type=str, help="Path to conversation text file")
    input_group.add_argument("--text", type=str, help="Conversation text")

    # Output options
    parser.add_argument("--output", type=str, required=True, help="Path to output JSON file")

    # Other options
    parser.add_argument("--append", action="store_true", help="Append to existing file instead of overwriting")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    return parser.parse_args()


def read_conversation_text(file_path: str) -> Optional[str]:
    """Read conversation text from a file.

    Args:
        file_path: Path to the conversation file.

    Returns:
        The conversation text, or None if the file could not be read.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading conversation file {file_path}: {e}")
        return None


def main() -> None:
    """Main entry point for the script."""
    args = parse_args()
    setup_logging(args.verbose)

    # Get conversation text
    if args.input:
        conversation_text = read_conversation_text(args.input)
    else:
        conversation_text = args.text

    if not conversation_text:
        logger.error("No conversation text provided")
        return

    # Extract intents from conversation
    intents = ConversationIntentExtractor.extract_intents_from_text(conversation_text)

    if not intents:
        logger.warning("No intents found in conversation")
        return

    logger.info(f"Extracted {len(intents)} intents from conversation")

    # Save intents to file
    output_path = Path(args.output)
    ConversationIntentExtractor.save_intents_to_file(intents, output_path, args.append)

    logger.info(f"Saved {len(intents)} intents to file: {output_path}")
    logger.info(f"You can now use these intents with semantic_matrix_analyzer.py")


if __name__ == "__main__":
    main()
