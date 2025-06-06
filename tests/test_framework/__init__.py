"""
Test framework package for OpenHCS.

This package contains the mathematical test framework and related utilities
for comprehensive testing of the OpenHCS codebase.
"""

from .mathematical_test_framework import test_framework, FrameworkMetrics, FrameworkResult

# For backward compatibility, provide aliases with the old names
TestMetrics = FrameworkMetrics
TestResult = FrameworkResult

__all__ = ['test_framework', 'FrameworkMetrics', 'FrameworkResult', 'TestMetrics', 'TestResult']
