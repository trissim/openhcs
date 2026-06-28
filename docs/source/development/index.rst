Development
===========

This section provides information for developers who want to contribute to OpenHCS.

Development Methodologies
=========================

Systematic approaches for OpenHCS development workflows.

.. toctree::
   :maxdepth: 1

   systematic_refactoring_framework
   architectural_refactoring_patterns
   refactoring_principles
   respecting_codebase_architecture
   literal_includes_audit_methodology
   compositional_commit_message_generation

Development Guides
==================

Practical guides for specific development tasks.

.. toctree::
   :maxdepth: 1

   ui-patterns
   pipeline_debugging_guide
   placeholder_inheritance_debugging
   parameter_analysis_audit
   unified_parameter_analyzer_migration
   placeholder_refresh_threading
   scope_hierarchy_live_context
   lazy_dataclass_utils
   pyclesperanto_simple_implementation
   window_manager_usage
   compact_window_patterns
   mcp_development
   runtime_system_assembly_rules

Testing and CI
==============

Continuous integration and testing strategies.

.. toctree::
   :maxdepth: 1

   git_worktree_testing
   omero_testing

**CPU-Only Testing**: OpenHCS supports CPU-only mode for CI environments without GPU dependencies. See :doc:`../user_guide/cpu_only_mode` for configuration details.

**CI Configuration Example**:

.. code-block:: yaml

   # .github/workflows/tests.yml
   name: Tests
   on: [push, pull_request]

   jobs:
     test:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v3
         - name: Set up Python
           uses: actions/setup-python@v3
           with:
             python-version: '3.9'
         - name: Install dependencies
           run: |
             pip install -e .
             pip install pytest
         - name: Run tests in CPU-only mode
           env:
             OPENHCS_CPU_ONLY: 1
           run: pytest tests/

**Key Benefits**:
- No GPU dependencies required for CI
- Faster test execution in cloud environments
- Consistent test results across different hardware
- Full pipeline logic validation without GPU acceleration
