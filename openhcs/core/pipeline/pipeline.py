"""
Pipeline module for OpenHCS.

This module provides the core pipeline components for OpenHCS:
- StepResult: Immutable container for step execution results
- PipelineCompiler: Compiles a pipeline into an executable form
- PipelineExecutor: Executes a compiled pipeline

Doctrinal Clauses:
- Clause 17 — VFS Exclusivity (FileManager is the only component that uses VirtualPath)
- Clause 17-B — Path Format Discipline
- Clause 66 — Immutability After Construction
- Clause 88 — No Inferred Capabilities
- Clause 106-A — Declared Memory Types
- Clause 251 — Declarative Memory Conversion
- Clause 262 — Pre-Runtime Conversion
- Clause 281 — Context-Bound Identifiers
- Clause 297 — Immutable Result Enforcement
- Clause 504 — Pipeline Preparation Modifications
- Clause 524 — Step = Declaration = ID = Runtime Authority
"""

import logging

from openhcs.core.pipeline.compiler import PipelineCompiler
from openhcs.core.pipeline.executor import PipelineExecutor
from openhcs.core.steps.step_result import StepResult

logger = logging.getLogger(__name__)

# Re-export key components
__all__ = [
    'StepResult',
    'PipelineCompiler',
    'PipelineExecutor'
]
