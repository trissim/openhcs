Glossary
========

.. glossary::
   :sorted:

   Artifact
      A typed non-main-flow result or dependency, such as measurements, object
      labels, relationships, tables, positions, or files.

   Artifact contract
      A callable or module declaration describing named artifact inputs and
      outputs. The compiler uses it to build producer/consumer dependencies.

   Callable contract
      The compiler-visible metadata for a processing callable: memory types,
      locality, required components, runtime-bound parameters, artifacts,
      adapters, and execution scope.

   Channel
      A wavelength or contrast channel, often associated with a fluorophore,
      stain, or transmitted-light acquisition.

   Compiled execution bundle
      The validated worker-safe steps, frozen processing contexts, and
      environment decisions consumed by plate execution.

   Compiled step plan
      The typed compiler-owned plan for one step on one execution axis. Runtime
      code reads it but does not repeat its inference.

   Compilation session
      The cross-step authority for source, artifact, and plan state while a
      pipeline is compiled.

   Execution axis
      The component that partitions independent processing contexts, normally
      well.

   Function pattern
      A callable, a callable plus keyword arguments, a chain, or a dictionary
      that routes compiled group keys to callables.

   Group by
      A processing declaration that partitions already assembled arrays. It
      controls fan-out only when a dictionary function pattern is used.

   Main flow
      The primary image/result stream passed from one pipeline step to the next.
      Named artifacts are a separate dependency mechanism.

   Materialization
      Persisting a declared result to a storage backend or streaming backend.
      Intermediate runtime flow does not imply persistent materialization.

   Measurement
      A typed quantitative result associated with images, objects, relationships,
      wells, or a plate.

   Microscope handler
      A registered source-format declaration that owns detection, parsing,
      metadata projection, virtual-workspace behavior, and compatible backends.

   Module artifact contract
      The artifact semantics declared by a module family, including a lowered
      CellProfiler module.

   Nominal registry
      A registry rooted at an owning base declaration. Generic consumers query
      the root registry instead of maintaining copied name tables.

   ObjectState
      The first-party system that owns editable object state, scope-aware lazy
      configuration resolution, snapshots, provenance, and time travel.

   PipelineConfig
      The inheritable configuration declaration paired with an ordered list of
      ``FunctionStep`` objects.

   Processing contract
      The semantic locality of a callable, such as plane-local ``PURE_2D`` or
      whole-stack ``PURE_3D``. It is independent of array memory type.

   Runtime adapter
      A declared boundary that converts typed compiler/runtime values into the
      invocation shape expected by a callable.

   Runtime value
      A typed value used during execution, including images, labels,
      measurements, relationships, and other artifact payloads.

   Site
      A field of view or tile acquired within a well.

   Source binding
      A declaration that connects a callable's semantic input to a source or
      artifact plan.

   Source universe
      The compiler's typed set of source identities and coordinates available
      to satisfy a step.

   Source workspace projection
      The virtual source view exposed by a microscope handler and PolyStore to
      initialization and compilation.

   Variable components
      The declaration that says what each position on the transported third
      array axis means, such as site, channel, or Z.

   Virtual workspace
      A source/reference projection that gives disparate storage backends a
      common path-like organization without requiring local copies.

   Well
      A sample location on a plate, normally identified by a row and column such
      as ``A01``.

   Z-stack
      Images acquired at multiple focal depths for the same otherwise fixed
      coordinate.
