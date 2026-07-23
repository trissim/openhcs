Python Object Reference Documentation Methodology
=================================================

**Perpetual documentation accuracy through Python object references and implementation-priming prose.**

*Status: CANONICAL*
*Applies to: All OpenHCS documentation with code examples*

Overview
--------

Traditional documentation suffers from the "documentation drift" problem - code examples become outdated as implementation evolves. OpenHCS eliminates this through systematic use of Python object references (`:py:meth:`, `:py:func:`, `:py:class:`) combined with clear, concise prose that primes readers with implementation context.

This methodology ensures documentation remains perpetually accurate by making Sphinx automatically validate object references, while providing readers with the mental framework to understand code before they see it.

Core Principles
---------------

Python Object References as Single Source of Truth
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Principle**: Documentation uses Python object references that Sphinx automatically validates against the actual codebase.

**Implementation**:

.. code-block:: rst

   The core entry point :py:meth:`~openhcs.core.lazy_config.LazyDataclassFactory.make_lazy_with_field_level_auto_hierarchy`
   works by creating a temporary lazy dataclass instance, then asking that instance to resolve the specific field value.

**Rationale**: When implementation changes, Sphinx build fails if referenced objects don't exist. No manual synchronization required, with automatic validation.

Implementation-Priming Prose
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Principle**: Prose explains how methods work conceptually before readers encounter the actual code, building intuition for easy code comprehension.

**Implementation Pattern**:

.. code-block:: rst

   :py:meth:`~openhcs.core.lazy_placeholder._resolve_field_with_composition_awareness` works like a smart field finder.
   Given a dataclass instance and a field name, it first checks if the field exists directly on the instance
   (using `dataclasses.fields()`). If found, it gets the value using `getattr()`. If not found, it loops through
   all fields looking for nested dataclasses, then recursively searches inside each one.

**Rationale**: Readers understand the implementation strategy before seeing code, making the actual implementation immediately comprehensible.

Architectural Accuracy Over Syntactic Accuracy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Principle**: Documentation should reflect architectural patterns and design decisions, with Python object references automatically tracking method evolution.

**Example**: Documentation describing lazy structure preservation functionality uses `:py:func:`~openhcs.core.lazy_config.rebuild_lazy_config_with_new_global_reference`` - if method is renamed, Sphinx build fails until reference is updated.

**Implementation Strategy**:

- Use Python object references that track refactoring automatically
- Sphinx validates object existence during build
- Preserve architectural explanations with validated references
- Build failures force documentation updates during refactoring

Logical Method Decomposition with Implementation Context
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Principle**: Large methods that serve multiple conceptual purposes should be documented with clear prose explaining each logical phase, using Python object references for validation.

**Documentation Approach**:

.. code-block:: rst

   Auto-Discovery Phase
   ~~~~~~~~~~~~~~~~~~~~

   The simplified contextvars system uses explicit context management to determine configuration
   hierarchy, eliminating the need for complex field path detection and auto-discovery mechanisms.

   Context Detection Phase
   ~~~~~~~~~~~~~~~~~~~~~~~

   :py:func:`~openhcs.core.lazy_config._get_current_config` checks if we're running in a PyQt application
   context where thread-local storage should be available. If context is missing in a GUI environment,
   it logs an architecture warning since this indicates a context management bug.

   Hierarchy Building Phase
   ~~~~~~~~~~~~~~~~~~~~~~~~

   The system constructs a resolution hierarchy by combining the current field path with discovered parent
   relationships. :py:meth:`~openhcs.core.lazy_config.LazyDataclassFactory._create_field_level_hierarchy_provider`
   builds a chain where each level can inherit from the next, creating the step → pipeline → global resolution flow.

**Rationale**: Prose explains the conceptual purpose of each phase, while Python object references ensure the described functionality actually exists and can be validated by Sphinx.

Audit Methodology
------------------

Phase 1: Python Object Reference Identification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Catalog all code references in documentation and convert to validated Python object references.

**Process**:

1. **Scan for code references**: Find all manual code examples, method names, and class references
2. **Classify references**:

   - **Implementation references**: Method signatures, class definitions → `:py:meth:`, `:py:class:`
   - **Function references**: Standalone functions → `:py:func:`
   - **Attribute references**: Class attributes, constants → `:py:attr:`
   - **Conceptual examples**: Pseudo-code illustrating patterns (keep as-is)

3. **Create mapping document**: Current reference, target Python object, validation status

**Tools**:

.. code-block:: bash

   # Find all code blocks in documentation
   find docs/ -name "*.rst" -exec grep -l "code-block:: python" {} \;

   # Find manual method references that should be Python objects
   grep -r "def [a-zA-Z_]" docs/source/architecture/ | grep -v ":py:"

   # Find class references that should be Python objects
   grep -r "class [A-Z]" docs/source/architecture/ | grep -v ":py:"

Phase 2: Python Object Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Verify each Python object reference points to actual, existing implementation.

**Process**:

1. **Object existence check**: Sphinx automatically validates Python object references during build
2. **Import verification**: Confirm referenced modules can be imported
3. **Signature validation**: Verify method signatures match expectations
4. **Phantom reference detection**: Sphinx build fails for non-existent objects

**Validation Benefits**:

- **Automatic validation**: Sphinx validates all `:py:` references during build
- **Import checking**: References fail if modules can't be imported
- **Refactoring safety**: Build fails when referenced objects are renamed/moved
- **No manual verification**: Sphinx handles all validation automatically

**Build Integration**:

.. code-block:: bash

   # Sphinx automatically validates Python object references
   sphinx-build -b html docs/source docs/build -W --keep-going

   # -W treats warnings as errors (including invalid references)
   # --keep-going shows all invalid references at once

Phase 3: Systematic Replacement with Implementation Context
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Replace code examples with Python object references and implementation-priming prose.

**Process**:

1. **High priority first**: Core implementation methods, class definitions
2. **Add implementation context**: Write clear prose explaining how each method works
3. **Batch replacement**: Group related examples for efficient processing
4. **Build verification**: Test documentation build after each batch

**Replacement Template**:

.. code-block:: rst

   {Clear prose explaining what the method does and how it works conceptually}

   :py:meth:`~module.path.ClassName.method_name` {additional context about the implementation approach}.

   {Brief explanation of why this approach was chosen or what problem it solves}

**Example Implementation**:

.. code-block:: rst

   The service uses :py:func:`~openhcs.core.lazy_placeholder._resolve_field_with_composition_awareness`
   to find field values. This function first checks if the field exists directly on the dataclass
   (like `num_workers` on `PipelineConfig`). If not found, it recursively searches through nested
   dataclasses (like looking for `output_dir_suffix` inside `materialization_defaults`).

Phase 4: Perpetual Maintenance with Automatic Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Maintain accuracy as codebase evolves with automatic validation.

**Process**:

1. **Build integration**: Documentation build fails if Python object references are invalid
2. **Refactoring protocol**: Sphinx automatically detects renamed/moved methods and fails build
3. **Review integration**: Code reviews catch documentation build failures from invalid references
4. **Automated verification**: CI enforces that all Python object references are valid

**Automatic Maintenance Benefits**:

- **Zero manual tracking**: Sphinx handles all reference validation
- **Immediate feedback**: Build fails instantly when references become invalid
- **Refactoring safety**: Impossible to forget updating documentation during refactoring
- **Cross-reference accuracy**: All internal links automatically validated

OpenHCS-Specific Implementation Guidelines
------------------------------------------

Fail-Loud Documentation with Implementation Context
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Principle**: Documentation build should fail immediately when Python object references become invalid, while providing clear implementation context.

**Implementation**: Use Sphinx's strict mode with Python object references and implementation-priming prose.

.. code-block:: rst

   # Fail-loud with implementation context
   :py:meth:`~openhcs.core.lazy_config.LazyDataclassFactory._create_lazy_dataclass_unified`
   works like a dataclass compiler. It takes a regular dataclass definition and generates a new
   class with the same fields and interface, but replaces the field access behavior.

Mathematical Simplification Applied to Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Principle**: Eliminate duplicate explanations by referencing single implementation source with consistent prose patterns.

**Before**:

.. code-block:: rst

   # Multiple explanations of the same concept
   Configuration Resolution (in config.rst)
   Lazy Field Resolution (in lazy_config.rst)
   Thread-Local Context (in context.rst)

**After**:

.. code-block:: rst

   # Single implementation with multiple documentation perspectives
   :py:func:`~openhcs.core.lazy_config._resolve_value_from_sources` implements the core
   resolution logic by trying each source in the fallback chain until one returns a non-None value.
   (Referenced consistently across config.rst, lazy_config.rst, and context.rst)

Architectural Coherence with Implementation Context
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Principle**: Document architectural patterns and design decisions with clear prose that explains implementation approach.

**Focus Areas**:

- **Why** code is structured a certain way (architectural rationale)
- **How** methods work conceptually (implementation approach)
- **When** to use specific approaches (usage context)
- **What** trade-offs were made (design decisions)

**Implementation Context Pattern**:

.. code-block:: rst

   # Explain HOW the method works before referencing it
   :py:meth:`~openhcs.core.lazy_placeholder._resolve_field_with_composition_awareness`
   works like a smart field finder. Given a dataclass instance and a field name, it first
   checks if the field exists directly on the instance. If not found, it loops through all
   fields looking for nested dataclasses, then recursively searches inside each one.

**Avoid**:

- Python object references without implementation context
- Architectural explanations without concrete method references
- Implementation details without architectural rationale

Breadth-First Documentation Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Principle**: Organize documentation from architectural concepts to implementation details.

**Structure**:

1. **Architectural overview**: Why the system exists, what problems it solves
2. **Core patterns**: Key design patterns and their rationale
3. **Implementation examples**: Literal includes showing actual code
4. **Usage patterns**: How to use the implemented functionality
5. **Integration details**: How components work together

Advanced Techniques
--------------------

Phantom Method Resolution
~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Documentation references methods that were renamed or refactored during development.

**Solution**: Update Python object references to point to actual implementation with clear implementation context.

**Process**:

1. **Sphinx validation**: Build fails automatically for non-existent method references
2. **Functionality analysis**: Determine what the phantom method was supposed to do
3. **Implementation mapping**: Find actual code that performs the same functionality
4. **Reference update**: Update Python object reference to actual method

**Example**:

.. code-block:: rst

   # Old documentation with phantom method reference
   :py:meth:`~openhcs.core.lazy_config._preserve_lazy_structure_if_needed`

   # Updated with actual implementation and context
   :py:func:`~openhcs.core.lazy_config.rebuild_lazy_config_with_new_global_reference`
   preserves lazy structure by creating a new lazy config instance that maintains the same
   field resolution behavior but uses an updated global reference for context resolution.

Logical Method Decomposition with Implementation Context
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Large architectural methods serve multiple conceptual purposes but cannot be split for architectural reasons.

**Solution**: Document logical phases with clear prose explaining each phase, using Python object references for validation.

**Technique**:

.. code-block:: rst

   Complex Provider Function
   ~~~~~~~~~~~~~~~~~~~~~~~~~

   :py:meth:`~openhcs.core.lazy_config.LazyDataclassFactory.make_lazy_simple`
   creates lazy dataclasses using the simplified contextvars system:

   **Simplified Approach**

   The new system uses explicit context management through Python's contextvars module,
   eliminating complex auto-discovery and context detection mechanisms while maintaining
   full inheritance functionality.

   The system constructs a resolution hierarchy by combining the current field path with
   discovered parent relationships using :py:func:`~openhcs.core.lazy_config._create_hierarchy_chain`.

   **Phase 4: Field Resolution**

   .. literalinclude:: ../../../openhcs/core/lazy_config.py
      :lines: 440-461
      :caption: Field-level inheritance resolution

Architectural Pattern Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Principle**: Document the architectural reasoning behind implementation choices.

**Template**:

.. code-block:: rst

   Pattern: {Pattern Name}
   ~~~~~~~~~~~~~~~~~~~~~~~

   **Problem**: {What architectural problem does this solve?}

   **Solution**: {How does the implementation address the problem?}

   **Implementation**:

   .. literalinclude:: ../../../{source_file}
      :lines: {start}-{end}
      :caption: {Pattern implementation}

   **Rationale**: {Why this approach over alternatives?}

   **Trade-offs**: {What are the costs and benefits?}

Cross-Reference Accuracy
~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Documentation cross-references become stale when methods are renamed or moved.

**Solution**: Use literal includes for cross-references to maintain accuracy.

**Implementation**:

.. code-block:: rst

   # Instead of manual cross-reference
   See the `_create_lazy_dataclass` method for implementation details.

   # Use literal include reference
   The lazy dataclass creation process:

   .. literalinclude:: ../../../openhcs/core/lazy_config.py
      :lines: 253-299
      :caption: Core lazy dataclass creation (_create_lazy_dataclass_unified)

Integration with OpenHCS Development Workflow
----------------------------------------------

Code Review Protocol
~~~~~~~~~~~~~~~~~~~~~

**Requirement**: All code changes that affect documented functionality must update corresponding literal includes.

**Process**:

1. **Identify affected documentation**: Which docs reference the changed code?
2. **Verify line numbers**: Do literal includes still point to correct functionality?
3. **Update captions**: Do descriptions still accurately reflect the code?
4. **Test documentation build**: Ensure all literal includes resolve correctly

Refactoring Safety Net
~~~~~~~~~~~~~~~~~~~~~~

**Principle**: Documentation serves as a safety net during refactoring by exposing all usage patterns.

**Benefits**:

- **Visibility**: See all places where code is referenced
- **Impact assessment**: Understand documentation implications of changes
- **Architectural coherence**: Ensure refactoring preserves documented patterns
- **Regression prevention**: Documentation build fails if refactoring breaks examples

Continuous Integration Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Requirements**:

.. code-block:: yaml

   # Documentation verification in CI
   documentation_check:
     - verify_literal_includes_exist
     - verify_line_ranges_contain_expected_content
     - build_documentation_strict_mode
     - check_cross_reference_accuracy

**Failure Modes**:

- **Missing files**: Literal include references non-existent file
- **Invalid line ranges**: Line numbers exceed file length
- **Empty ranges**: Line range contains no code
- **Functionality mismatch**: Code at line range doesn't match description

Quality Metrics
---------------

Quantitative Metrics
~~~~~~~~~~~~~~~~~~~~~

**Documentation Accuracy Rate**:

.. code-block:: text

   Accuracy = (Valid Literal Includes / Total Code Examples) × 100
   Target: >95% for core architecture docs

**Implementation Coverage**:

.. code-block:: text

   Coverage = (Documented Public Methods / Total Public Methods) × 100
   Target: >80% for core modules

**Maintenance Efficiency**:

.. code-block:: text

   Efficiency = Development Time / Documentation Update Time
   Target: <5% overhead

Qualitative Indicators
~~~~~~~~~~~~~~~~~~~~~~~

**Developer Experience**:

- Developers trust documentation examples to work
- Code reviews catch documentation inconsistencies
- Refactoring confidence increases due to documentation safety net

**Architectural Clarity**:

- Design decisions are clearly explained and justified
- Implementation patterns are consistently documented
- Complex logic is broken down into understandable sections

**Codebase Health**:

- Documentation pressure improves code quality
- Architectural patterns become more consistent
- Complex methods are naturally decomposed for documentability

Success Metrics
---------------

Quantitative Measures
~~~~~~~~~~~~~~~~~~~~~

- **Python object reference coverage**: Percentage of code references using `:py:` directives vs manual examples
- **Build failure rate**: Frequency of documentation builds failing due to invalid object references
- **Implementation context coverage**: Percentage of Python object references with explanatory prose

Qualitative Measures
~~~~~~~~~~~~~~~~~~~~

- **Documentation accuracy**: Alignment between documented references and actual implementation
- **Developer comprehension**: Ease of understanding code after reading implementation context
- **Onboarding effectiveness**: New developer ability to navigate codebase using documentation

Target State
~~~~~~~~~~~~

- **100% Python object reference coverage** for all implementation references
- **Zero tolerance** for manual code examples that duplicate implementation
- **Automatic validation** integrated into CI/CD pipeline with Sphinx strict mode
- **Implementation-priming prose** for all Python object references
- **Fail-fast feedback** when implementation changes break documentation

Example Quality Standard
~~~~~~~~~~~~~~~~~~~~~~~~

**Excellent Documentation Pattern**:

.. code-block:: rst

   The simplified placeholder service uses the new contextvars-based resolution system
   to find field values. The system uses explicit context management and cross-dataclass inheritance
   through the dual-axis resolver, eliminating the need for complex composition awareness mechanisms.

**Benefits**: Validated reference + clear implementation context + architectural rationale + concrete examples.

Benefits
--------

For Developers
~~~~~~~~~~~~~~~

1. **Guaranteed accuracy**: Code examples always reflect current implementation
2. **Reduced maintenance**: No manual synchronization of code and documentation
3. **Architectural insight**: Documentation explains design decisions, not just syntax
4. **Refactoring safety**: Documentation automatically updates with code changes

For Users
~~~~~~~~~

1. **Reliable examples**: All code examples are guaranteed to work
2. **Current information**: Documentation never lags behind implementation
3. **Architectural understanding**: Learn not just how, but why
4. **Consistent patterns**: Same implementation referenced across multiple contexts

For Codebase Health
~~~~~~~~~~~~~~~~~~~

1. **Perpetual audit**: Documentation serves as continuous code review
2. **Architectural documentation**: Forces clear explanation of design decisions
3. **Implementation visibility**: Complex logic must be documentable to be maintainable
4. **Quality pressure**: Poor code becomes obvious when documented

Implementation Checklist
-------------------------

- [ ] Catalog all code examples in documentation
- [ ] Verify each example against current implementation
- [ ] Create systematic mapping of examples to source code
- [ ] Replace examples with literal includes (high priority first)
- [ ] Integrate literal include verification into CI pipeline
- [ ] Establish refactoring protocol for updating documentation
- [ ] Document architectural patterns and design rationale
- [ ] Set up automated accuracy metrics and monitoring

This methodology transforms documentation from a maintenance burden into a perpetual code audit system that ensures architectural coherence and implementation accuracy.
