Agent Refactoring Crash Course
==============================

Canonical agent-facing procedure for eliminating architectural rot in an unfamiliar codebase.

This document fuses the operational guidance from:

- ``respecting_codebase_architecture.rst``
- ``refactoring_principles.rst``
- ``systematic_refactoring_framework.rst``
- ``architectural_refactoring_patterns.rst``
- Paper 1 (axis derivation, nominal contracts, error-localization theory)
- Paper 2 (single-source structure, derivation, verifiable integrity)

Read this before making large architectural changes.

Purpose
-------

An agent should not treat refactoring as "clean up a few lines". It should treat refactoring as:

1. deriving the correct architecture from the domain requirements,
2. identifying where the current code violates that architecture, and
3. collapsing duplicated, scattered, or manually synchronized structure into one authoritative source plus derived views.

The slogan is:

**declare, introspect, derive**

- **Declare** the authoritative concepts explicitly with dataclasses, ABCs, enums, and nominal contracts.
- **Introspect** the code mechanically to discover duplication, scattered invariants, and fake-generic structure.
- **Derive** concrete implementations, views, wrappers, and export formats from the authoritative declarations.


The Six Laws
------------

Law 1: Requirements Derive Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Paper 1's core claim is that axis choices are not preferences. They are derived from the query domain.

Practical consequence:

- If the system needs provenance, identity, enumeration, or conflict-resolution, the code must use nominal contracts.
- In Python this means explicit classes, ABCs, enums, registries keyed by type identity, and direct method calls.
- Replacing those with structural probing (``hasattr``, ``getattr(..., default)``, method-existence checks, or ad-hoc dict dispatch) is architectural regression.


Law 2: One Authoritative Source, Derived Views Everywhere Else
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Paper 2's unit-rate / SSOT result says coherence happens only when there is one authoritative source and everything else is derived.

Practical consequence:

- There should be exactly one authoritative representation for each fact family.
- JSON, CSV, reports, caches, secondary dataclasses, adapters, command builders, and UI displays should be derived from that authority.
- If two locations can be edited independently to represent the same fact, rot is already present.


Law 3: Provenance Must Be Observable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Paper 2 also says verifiable integrity requires provenance observability.

Practical consequence:

- Refactors should make it obvious which object owns a fact and which objects are derived.
- A reader should be able to answer: "where does this value come from?" in O(1) conceptual effort.
- Hidden fallback chains, silent defaults, and repeated ad-hoc transformations destroy provenance.


Law 4: Respect Architectural Guarantees
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the architecture guarantees an attribute or method, access it directly.

Forbidden patterns:

- ``getattr(obj, 'field', default)`` for guaranteed fields
- ``hasattr(obj, 'method')`` for ABC-required methods
- ``try/except AttributeError`` to fabricate fallback behavior
- re-querying information that is already available in the current scope

Required pattern:

- direct access
- fail loud on architectural violations
- type-based dispatch when behavior genuinely varies by nominal type


Law 5: Factor Common Structure Algebraically
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The refactoring documents treat duplication like algebra.

Practical consequence:

- If two branches have the same shape but different parameters, factor the common shape.
- If two dataclasses differ only by a few fields, make the shared structure authoritative.
- If many methods have the same signature shape, create a base request / plan / result / strategy layer.


Law 6: Use Python's Type System and Metaprogramming Aggressively
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Do not hand-maintain repeated wrappers and repeated schemas.

Use:

- ``ABC`` for behavioral contracts
- dataclass inheritance for overlapping concerns
- enums for discrete behavior families
- ``Generic[T]`` when a pattern is real and repeated
- dataclass introspection (`fields(...)`) for export / validation / audit derivation
- generated or derived methods when the pattern is structural rather than incidental


Red Flags and Their Required Meanings
-------------------------------------

These are not style nits. They are symptoms of mathematical structural failure.

Repeated ``if/elif/else`` on the same variant axis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Symptom:

- the same mode test appears in multiple methods
- examples: ``if mode == ...``, ``if config.mode == ...``, ``if use_pocket_guided``

Meaning:

- the variant axis is real but undeclared

Required refactor:

- introduce an enum, ABC hierarchy, or strategy family
- move the varying behavior behind nominal dispatch


Highly similar method signatures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Symptom:

- two or more methods take nearly the same parameter bag
- the call sites forward the same fields repeatedly

Meaning:

- the parameter bundle is a latent domain object

Required refactor:

- create an authoritative request / context / plan dataclass
- use inheritance when the methods represent refinements of the same concept
- use composition only when the concerns are genuinely orthogonal


Explicit repeated ``None`` handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Symptom:

- many ``if x is None`` branches for the same field family
- repeated ``None if ... else ...`` conversions
- repeated fallback values for the same concept

Meaning:

- the code has not declared the correct states explicitly

Required refactor:

- replace optionality with nominal variants where possible
- push defaults to construction time so downstream code sees a valid object
- derive specialized subclasses instead of re-testing optionals at each use site


Manual synchronization between representations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Symptom:

- the same data is copied into multiple dicts / files / row objects / caches
- updates must be kept in sync manually

Meaning:

- the code violates SSOT

Required refactor:

- identify the authoritative representation
- derive all exports and views from it
- remove second writable sources


Single-use wrappers and pass-through helpers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Symptom:

- a helper is called only once
- it just forwards data or renames parameters

Meaning:

- unnecessary indirection

Required refactor:

- inline it if it does not represent a reusable contract
- keep it only if it defines a stable domain-level abstraction


Magic strings for behavior
~~~~~~~~~~~~~~~~~~~~~~~~~~

Symptom:

- behavior depends on string values rather than enum members or types

Meaning:

- the domain axis is encoded informally

Required refactor:

- use enums or nominal strategy classes


Agent Procedure
---------------

Step 1: Establish the Domain Axes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before editing code, answer these questions:

1. What are the real behavior families?
2. Which variations are domain-level and stable?
3. Which values are authoritative facts, and which are derived views?
4. Where does provenance matter?
5. Which axes are already present but scattered behind repeated conditionals?

If provenance or identity matters, prefer nominal types immediately.


Step 2: Locate the Authoritative Structures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Look for the latent SSOT candidates:

- repeated parameter bags
- duplicated dataclass fields
- repeated export dict builders
- repeated request/response/result wrappers
- repeated mode-selection branches

The correct question is not "how do I reduce lines?".
The correct question is: **what single declaration should own this fact?**


Step 3: Audit Mechanically First
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Do not rely on eyeballing alone. Run mechanical audits.

Minimum audit categories:

- duplicated method signatures
- duplicated dataclass field sets
- repeated if/else chains over the same axis
- ``hasattr`` / ``getattr`` / ``AttributeError`` fallback patterns
- repeated ``None`` handling
- single-use private helper methods

An AST audit is preferred because it catches structural similarity that grep misses.

Example audit script:

.. code-block:: python

   import ast
   from pathlib import Path
   from itertools import combinations

   root = Path("dq_dock_engine")
   files = list(root.rglob("*.py"))

   def fn_params(fn: ast.FunctionDef) -> list[str]:
       out = []
       for arg in fn.args.posonlyargs + fn.args.args:
           if arg.arg != "self":
               out.append(arg.arg)
       for arg in fn.args.kwonlyargs:
           out.append(arg.arg)
       return out

   def dataclass_fields(cls: ast.ClassDef) -> list[str]:
       result = []
       for stmt in cls.body:
           if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
               result.append(stmt.target.id)
       return result

   funcs = []
   classes = []

   for path in files:
       tree = ast.parse(path.read_text())
       for node in ast.walk(tree):
           if isinstance(node, ast.FunctionDef):
               funcs.append((path, node.name, fn_params(node)))
           if isinstance(node, ast.ClassDef):
               classes.append((path, node.name, dataclass_fields(node)))

   for a, b in combinations(funcs, 2):
       sa, sb = set(a[2]), set(b[2])
       overlap = sa & sb
       if len(overlap) >= 5:
           print("FUNCTION", a[0], a[1], b[0], b[1], sorted(overlap))

   for a, b in combinations(classes, 2):
       sa, sb = set(a[2]), set(b[2])
       overlap = sa & sb
       if len(overlap) >= 3:
           print("CLASS", a[0], a[1], b[0], b[1], sorted(overlap))

Also run targeted searches for architectural disrespect:

.. code-block:: bash

   rg "hasattr\(|getattr\(|except AttributeError|if .* is None|if .* is not None" dq_dock_engine
   rg "if .*mode|if .*strategy|if .*backend|if .*type|elif .*" dq_dock_engine


Step 4: Classify Each Smell Before Refactoring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For every hotspot, classify it into one of these buckets.

Bucket A: Real Variant Axis
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use when behavior changes by a real domain mode.

Refactor into:

- enum with behavior methods, or
- ABC + subclasses, or
- strategy objects


Bucket B: Shared Structural Core with Refinements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use when multiple dataclasses or methods represent the same concept with extra fields or stronger guarantees.

Refactor into:

- base dataclass + derived dataclasses
- base request / result / plan / preparation object


Bucket C: Orthogonal Concern
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use only when a concern truly crosses the domain concepts without being a subtype relation.

Refactor into:

- composition

Examples:

- logging
- file output destinations
- telemetry
- caching policies


Bucket D: Single-Use Noise
^^^^^^^^^^^^^^^^^^^^^^^^^^

Use when a helper does not define a reusable contract.

Refactor into:

- inlining


Step 5: Apply the Correct Refactoring Pattern
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pattern A: Repeated Variant Branches -> Polymorphism
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before:

.. code-block:: python

   if mode == CERTIFIED:
       return run_certified(...)
   if use_pocket_guided:
       return run_geometric(...)
   return run_generic(...)

After:

.. code-block:: python

   class Executor(ABC):
       @abstractmethod
       def execute(self, context): ...

   class CertifiedExecutor(Executor): ...
   class GeometricExecutor(Executor): ...
   class GenericExecutor(Executor): ...

   executor = derive_executor(requirements)
   return executor.execute(context)


Pattern B: Repeated Parameter Bags -> Request Hierarchy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before:

- multiple methods with 12-20 overlapping parameters
- repeated forwarding wrappers

After:

.. code-block:: python

   @dataclass(frozen=True, kw_only=True)
   class RequestBase:
       ...

   @dataclass(frozen=True, kw_only=True)
   class CertifiedRequest(RequestBase):
       ...

   @dataclass(frozen=True, kw_only=True)
   class GeometricRequest(RequestBase):
       ...


Pattern C: Repeated Similar Dataclasses -> Inheritance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before:

- two dataclasses with identical geometric core and one proof-bearing extension

After:

.. code-block:: python

   @dataclass(frozen=True)
   class BindingSite:
       center: Array
       radius: float

   @dataclass(frozen=True)
   class CertifiedBindingSite(BindingSite):
       theorem_handles: tuple[str, ...]


Pattern D: Repeated Export Builders -> Declarative Export Schema
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before:

- separate JSON and CSV dict literals with mostly the same keys

After:

- one authoritative row dataclass
- one derived export schema
- format-specific rendering derived from the dataclass and schema

Use dataclass introspection when the field mapping is systematic.


Pattern E: Repeated Manual Synchronization -> Derived Views
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before:

- a benchmark result, a summary row, a JSON record, and a CSV record all store overlapping writable facts

After:

- one authoritative result object
- summary rows and exports derived from it


Pattern F: Repeated ``None`` Checks -> State Refinement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before:

.. code-block:: python

   if binding_site is None:
       ...
   if binding_site is not None:
       ...

After:

- represent "no binding site", "geometric binding site", and "certified binding site" as explicit variants when they carry distinct semantics
- construct the correct variant once, then dispatch on the variant rather than re-checking optional fields everywhere


Inheritance vs Composition vs Enum vs Pure Function
---------------------------------------------------

Use inheritance when:

- two structures share most fields and one is a refinement of the other
- two methods share a contract and differ by variant-specific behavior
- provenance / identity of the variant matters

Use composition when:

- the concern is orthogonal and cross-cutting
- removing it does not change the domain identity of the host object

Use enums when:

- the variant family is closed and behavior is lightweight
- the main problem is magic strings or discrete mode dispatch

Use pure functions when:

- the logic is stateless and transformation-oriented
- the operation is shared independent of variant state

Use metaprogramming when:

- wrappers, exports, validation, or registration are structurally derivable
- the code is repeating the same pattern over declared fields or enum members


What the Agent Must Remove Aggressively
---------------------------------------

- ``hasattr`` checks for architecturally guaranteed members
- ``getattr`` fallbacks for required fields
- repeated mode branches spread across many functions
- repeated parameter forwarding methods
- duplicated dataclass cores
- duplicate JSON / CSV / report dictionaries carrying the same facts
- repeated defaulting of the same optional values
- helper methods with only one real call site and no contract value


What the Agent Must Preserve Aggressively
-----------------------------------------

- explicit nominal contracts
- authoritative dataclasses
- proof / provenance metadata
- fail-loud behavior
- direct access to guaranteed attributes
- single-source / derived-view architecture


Validation Checklist
--------------------

After each major refactor, verify all of these:

- the authoritative source for each fact family is obvious
- derived views are truly derived, not manually synchronized duplicates
- no new ``hasattr`` / ``getattr(..., default)`` fallback patterns were introduced
- the number of repeated mode checks decreased
- duplicated method signatures decreased
- duplicated dataclass cores decreased
- compile / type-check / test / benchmark still pass
- provenance remains visible: a reader can still trace where each fact comes from


Agent Stop Conditions
---------------------

An agent should stop refactoring a region only when:

1. the domain axes are explicit,
2. the authoritative sources are explicit,
3. the main behavior families are dispatched nominally,
4. repeated fallback logic is gone, and
5. remaining duplication is either trivial or genuinely orthogonal.


Final Commandment
-----------------

Do not ask, "How do I make this code more defensive?"

Ask:

- What is the authoritative declaration?
- What should be derived from it?
- What nominal type should own this behavior?
- Which repeated branches reveal an undeclared axis?
- Which repeated wrappers reveal a latent domain object?

If you answer those correctly, the rot becomes mechanically removable.
