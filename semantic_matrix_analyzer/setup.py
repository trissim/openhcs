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
        "nltk",
        "spacy",
        "scikit-learn",
        "networkx",
    ],
    entry_points={
        "console_scripts": [
            "semantic-matrix-analyzer=semantic_matrix_analyzer.cli:main",
            "extract-intents=semantic_matrix_analyzer.extract_intents:main",
            "sma=semantic_matrix_analyzer.cli:main",
            "sma-intent=semantic_matrix_analyzer.intent.cli:main",
        ],
    },
)
