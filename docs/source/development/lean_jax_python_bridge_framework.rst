Verified Lean-JAX-Python Bridge Framework
=========================================

**Status**: PROPOSED CANONICAL
**Applies to**: Lean-backed JAX integration, generated Python interfaces, proof-aware runtime wrappers

Overview
--------

OpenHCS already contains the two ingredients needed for a verified computational bridge:

- A Lean-side semantic model of JAX-like primitives in ``docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Computation/ArrayDSL.lean``
- A Python-side proof metadata pattern in ``dq_dock_engine/proof_status.py``

What is missing is the architectural layer between them. The repository does not currently have a real Lean-to-Python code generation pipeline. It has a specification, a Python runtime, and some manually maintained correspondence. That gap is exactly where drift, wrapper duplication, and unverifiable hand wiring appear.

This document defines the correct architecture for turning ``ArrayDSL.lean`` into a small Python/JAX code generation pipeline without violating OpenHCS refactoring principles.

Core Architectural Decision
---------------------------

Use a hybrid architecture:

1. **Lean defines semantics**
2. **Lean exports a machine-readable IR**
3. **Python generates thin JAX wrappers from that IR**
4. **A Python metaclass/decorator layer registers and annotates the generated wrappers**

This combines the best parts of all three candidate approaches:

- **Approach 1**: Turn ``ArrayDSL.lean`` into a small Python/JAX codegen pipeline
- **Approach 2**: Design a Lean IR that emits Python wrappers
- **Approach 3**: Use a Python metaclass/decorator interface aligned with the proven Python model

The correct answer is not to choose one of these in isolation. The correct answer is to stack them.

Why A Hybrid Architecture Is Required
-------------------------------------

Codegen-Only Is Insufficient
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If Python directly parses Lean source text and generates wrappers, the parser becomes the real source of truth. That is architecturally wrong.

- Lean comments and formatting become API surface
- Small syntax changes break Python generation
- Python accidentally owns Lean semantics

**Rule**: Python must never scrape ``.lean`` source files as its primary compilation mechanism.

IR-Only Is Insufficient
~~~~~~~~~~~~~~~~~~~~~~~

A Lean IR with no generated Python layer still leaves engineers hand-writing wrappers, registrations, imports, and proof annotations.

- Manual wrappers drift from the Lean model
- Runtime signatures diverge from exported semantics
- The proof/runtime bridge becomes documentation instead of infrastructure

Metaclass-Only Is Insufficient
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Python metaclasses and decorators are excellent for registration and runtime structure, but they do not create verified correspondence by themselves.

- They can enforce contracts
- They cannot discover verified primitives from Lean without an exported artifact
- They help runtime organization, not semantic extraction

Architectural Principles Applied
--------------------------------

Explicit Dependency Injection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The generator should consume an explicit exported artifact:

- Good: ``generate_wrappers(ir_path, output_path, template_env)``
- Bad: generator imports Lean internals implicitly or shells out to ad hoc grep pipelines

Fail-Loud Error Handling
~~~~~~~~~~~~~~~~~~~~~~~~

If the Lean export is malformed, generation must fail immediately.

- No fallback wrapper generation
- No best-effort signature guessing
- No silent omission of primitives

Indirection Minimization
~~~~~~~~~~~~~~~~~~~~~~~~

Generated Python wrappers should be thin and direct.

- No unnecessary adapter layers around ``jax.numpy``
- No string-dispatch at runtime if dispatch can be generated once
- No duplicate registry and lookup systems for the same primitive set

Consistent Interface Design
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every exported primitive should produce the same family of Python artifacts:

- one generated wrapper function
- one generated metadata object
- one registration entry
- one proof linkage entry

Recommended Architecture
------------------------

Layer 1: Lean Semantic Source Of Truth
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ArrayDSL.lean`` remains the authoritative semantic layer.

It should own:

- primitive names
- argument lists
- result shapes and kinds
- differentiability metadata
- lowering target identifiers
- proof linkage names where available

The existing ``primitiveJAXMapping`` list is a useful start, but it is not yet a sufficient IR because it stores free-form strings rather than structured lowering data.

Layer 2: Lean Export IR
~~~~~~~~~~~~~~~~~~~~~~~

Introduce a structured Lean export layer that sits next to ``ArrayDSL.lean``.

Preferred shape:

.. code-block:: lean

   namespace DecisionQuotient
   namespace Computation
   namespace ArrayDSL

   inductive ScalarType where
     | real
     | boolean

   inductive ExprKind where
     | tensor
     | scalar

   structure ArgSpec where
     name : String
     kind : ExprKind
     dtype : ScalarType

   structure LoweringSpec where
     target : String
     symbol : String
     supportsGrad : Bool

   structure PrimitiveIR where
     name : String
     args : List ArgSpec
     resultKind : ExprKind
     lowering : LoweringSpec
     leanSymbol : String
     theorem : Option String

   def exportPrimitives : List PrimitiveIR := ...

   end ArrayDSL
   end Computation
   end DecisionQuotient

This is the key refactoring. Replace stringly typed correspondence with structured correspondence.

Layer 3: Exporter Executable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add a tiny Lean exporter executable that writes JSON.

.. code-block:: text

   lake exe arraydsl-export --output build/arraydsl_primitives.json

The exporter is responsible for:

- serializing ``exportPrimitives``
- preserving stable field names
- failing if a primitive is incomplete

It is not responsible for:

- generating Python source code
- importing JAX
- making runtime registration decisions

Layer 4: Python Code Generator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add a Python generator that consumes only the exported IR.

Suggested layout:

- ``dq_dock_engine/codegen/arraydsl_codegen.py``
- ``dq_dock_engine/generated/arraydsl_primitives.py``
- ``dq_dock_engine/generated/arraydsl_registry.py``

The generator should create:

- thin JAX wrappers
- a registry of exported primitive metadata
- proof linkage fields compatible with ``dq_dock_engine/proof_status.py``

Example generated wrapper shape:

.. code-block:: python

   import jax.numpy as jnp

   def reduce_sum(arr):
       return jnp.sum(arr)

   reduce_sum._lean_symbol = "DecisionQuotient.Computation.ArrayDSL.reduce_sum"
   reduce_sum._jax_symbol = "jnp.sum"
   reduce_sum._supports_grad = True
   reduce_sum._proof_status = "CERTIFIED"

Layer 5: Python Runtime Registration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generated wrappers should be registered through a small, explicit runtime layer.

Use a metaclass or decorator for registration, but do not make the metaclass perform code generation.

Good division of responsibilities:

- Lean exporter: semantic export
- Python generator: source generation
- Python metaclass/decorator: runtime registration and validation

Lean IR Design Rules
--------------------

Rule 1: Structured Lowering, Never Free-Form Expressions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is too weak:

.. code-block:: lean

   ⟨"reduce_sum", "jnp.sum(arr)", true⟩

This is strong enough:

.. code-block:: lean

   {
     name := "reduce_sum"
     args := [⟨"arr", .tensor, .real⟩]
     resultKind := .scalar
     lowering := ⟨"jax.numpy", "sum", true⟩
     leanSymbol := "DecisionQuotient.Computation.ArrayDSL.reduce_sum"
     theorem := none
   }

Rule 2: Separate Semantic Name From Backend Symbol
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Lean name is the semantic contract. The JAX symbol is a lowering detail.

- ``reduce_sum`` is stable API meaning
- ``jax.numpy.sum`` is an execution target

This separation makes alternate backends possible later without changing the semantic layer.

Rule 3: Export Proof Linkage Explicitly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If a Python wrapper is meant to carry proof metadata, the export must say so directly.

- Good: ``theorem := some "ArrayDSL.lean::norm_nonneg_bound"``
- Bad: infer proof linkage from Python naming conventions

Rule 4: Encode Wrapper Shape In The IR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The generator should not guess whether an operation returns a scalar, tensor, mask, or callable.

If wrapper shape matters at runtime, encode it in Lean.

Python Runtime Interface
------------------------

Use dataclasses and explicit contracts instead of ad hoc dictionaries.

.. code-block:: python

   from dataclasses import dataclass
   from typing import Callable

   @dataclass(frozen=True)
   class PrimitiveMetadata:
       name: str
       lean_symbol: str
       jax_symbol: str
       supports_grad: bool
       theorem: str | None
       callable: Callable

This metadata object should be the registry value. Avoid plain nested dicts when the schema is fixed.

Metaclass And Decorator Pattern
-------------------------------

The proven Python model in ``docs/papers/proofs/paper2_LangPython.lean`` justifies using definition-time hooks for wrapper registration.

That means a runtime interface like this is appropriate:

.. code-block:: python

   class VerifiedPrimitiveMeta(type):
       REGISTRY: dict[str, type] = {}

       def __new__(mcls, name, bases, namespace, **kwargs):
           cls = super().__new__(mcls, name, bases, namespace)
           primitive_name = namespace["PRIMITIVE_NAME"]
           mcls.REGISTRY[primitive_name] = cls
           return cls

   class ReduceSumPrimitive(metaclass=VerifiedPrimitiveMeta):
       PRIMITIVE_NAME = "reduce_sum"
       LEAN_SYMBOL = "DecisionQuotient.Computation.ArrayDSL.reduce_sum"

       @staticmethod
       def lower(arr):
           import jax.numpy as jnp
           return jnp.sum(arr)

This pattern is good for registration. It is not good as the primary semantic source.

Preferred Runtime Pattern
~~~~~~~~~~~~~~~~~~~~~~~~~

Use generated functions for execution and a light registration layer for discovery:

.. code-block:: python

   def register_verified_primitive(metadata: PrimitiveMetadata) -> PrimitiveMetadata:
       PRIMITIVE_REGISTRY[metadata.name] = metadata
       return metadata

This is simpler than generating one class per primitive unless subclass behavior is actually needed.

Small Pipeline Scope
--------------------

Do not start with the entire DSL. Start with five primitives:

1. ``elemBinaryAdd``
2. ``elemBinarySub``
3. ``reduce_sum``
4. ``norm``
5. ``distance``

These primitives are enough to validate:

- scalar vs tensor returns
- binary vs unary operators
- lowering to ``jax.numpy``
- proof metadata propagation
- generated import stability

Phased Implementation Plan
--------------------------

Phase 1: Refactor ``ArrayDSL.lean`` Into Exportable IR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- keep existing semantic definitions
- replace or supplement ``primitiveJAXMapping`` with structured ``PrimitiveIR`` values
- add theorem linkage where it already exists

Phase 2: Add Lean JSON Export
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- add a small ``lake exe`` target
- emit deterministic JSON
- check the exported artifact into a temporary build location, not a hand-edited source location

Phase 3: Generate Python Wrappers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- generate one module of thin wrappers
- generate one registry module
- generate proof linkage metadata compatible with ``proof_status.py``

Phase 4: Add Runtime Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- verify every exported primitive appears in the registry
- verify every generated wrapper has Lean and JAX symbol metadata
- fail generation if any primitive is missing required fields

Phase 5: Integrate With Existing Proof Metadata
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- connect exported theorem names to the decorator/metadata pattern in ``dq_dock_engine/proof_status.py``
- allow generated wrappers to expose the same proof status interface as handwritten certified functions

Forbidden Patterns
------------------

Do not introduce these anti-patterns:

- Python regex parsing of ``ArrayDSL.lean`` as the long-term compiler
- handwritten wrappers that duplicate exported primitives one by one
- runtime ``getattr(..., default)`` fallbacks for generated metadata
- best-effort generation that silently skips unknown primitives
- mixing semantic extraction, source generation, and runtime registration into one giant script

Success Criteria
----------------

The bridge is architecturally correct when all of the following are true:

- Lean owns the primitive schema
- Python generation consumes only exported IR
- Generated wrappers are thin and deterministic
- Proof linkage is explicit, not inferred
- Registration happens at definition time with no manual registry editing
- Missing or malformed primitive metadata fails loudly

Recommended Initial File Layout
-------------------------------

.. code-block:: text

   docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Computation/ArrayDSL.lean
   docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Computation/ArrayDSLExport.lean
   docs/papers/paper4_decision_quotient/proofs/Main.lean
   dq_dock_engine/codegen/arraydsl_codegen.py
   dq_dock_engine/generated/arraydsl_primitives.py
   dq_dock_engine/generated/arraydsl_registry.py
   tests/dq_dock_engine/test_arraydsl_codegen.py

Final Recommendation
--------------------

Build the bridge in this order:

1. **Design the Lean IR first**
2. **Export that IR from Lean second**
3. **Generate Python/JAX wrappers third**
4. **Add Python metaclass/decorator registration fourth**

That ordering respects the codebase architecture.

It keeps Lean as the semantic authority, Python as the execution surface, and metaprogramming as a structural tool rather than a source of hidden behavior.
