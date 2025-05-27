#!/usr/bin/env python3
"""
Setup script for semantic_matrix_analyzer package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="semantic_matrix_analyzer",
    version="0.1.0",
    author="OpenHCS Team",
    author_email="your.email@example.com",
    description="A tool for analyzing Python codebases using AST to create semantically dense matrices",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/semantic_matrix_analyzer",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "matplotlib",
        "pyyaml",
    ],
    extras_require={
        "tensorflow": ["tensorflow>=2.12.0,<2.16.0"],
        "gpu": [
            "torch",
            "torchvision",
            "jaxlib",
            "cupy",
            "tensorflow>=2.12.0,<2.16.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sma-cli=semantic_matrix_analyzer.sma_cli:main",
            "semantic-matrix-analyzer=semantic_matrix_analyzer.cli:main",
            "extract-intents=semantic_matrix_analyzer.extract_intents:main",
            "conversation-memory=semantic_matrix_analyzer.conversation.memory_cli:main",
        ],
    },
)
