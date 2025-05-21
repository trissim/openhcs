#!/usr/bin/env python3
"""
Command-line interface for conversation memory.

This module provides a command-line interface for interacting with the conversation memory system.
"""

import sys
from semantic_matrix_analyzer.conversation.memory.cli import main

if __name__ == "__main__":
    sys.exit(main())
