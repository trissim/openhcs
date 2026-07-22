Source model
============

The source system converts plate files, explicit image-plane declarations, and
metadata into typed, named inputs for pipeline steps. ``SourceBindingsConfig``
is the public pipeline authority; the deleted source-schema layer is not part of
the current model.

Generic folders and unsupported microscopes
-------------------------------------------

Source bindings are the nominal microscope-independent ingestion path for
arbitrary folders. When microscope auto-detection fails and a non-empty
``SourceBindingsConfig`` is present, the handler factory selects
``SourceBindingsHandler``. That handler recursively inventories the folder and
projects only the ordinary files selected by the declarations into a virtual
workspace.

This fallback is distinct from semantic source selection on a recognized store.
Native HCS handlers retain ownership of recognized layouts; CZI, OME-TIFF, and
other rich containers retain Bio-Formats/store ownership. A handler that declares
``projects_declared_source_bindings()`` may apply ``SourceBindingsConfig`` to the
planes it emits, but the config does not become the container decoder or replace
the detected handler.

The correct workflow is:

1. Inventory and sample the actual files.
2. Bound the source universe with typed ``SourceFilterClause`` values.
3. Extract well, site, channel, Z, time, or experiment metadata with named
   captures in ``MetadataExtractionRule``.
4. Name semantic inputs with ``NamedSourceBinding`` and declare any component
   identity or cross-alias matching rules.
5. Put discovery defaults on ``PipelineConfig.source_bindings_config`` and use
   ``FunctionStep.source_bindings`` for step-local source selection.
6. Compile and inspect the source workspace and artifact plan before execution.

Filename knowledge therefore ends at the source-binding authority. Do not add a
one-off microscope enum, parse filenames in a processing callable, or recreate
source matching with dictionaries and fallback strings.

Declaration vocabulary
----------------------

``ImagePlaneSource``
  An explicit source URI with optional series, index, and channel identity.

``SourceSelector`` and ``SourceFilterClause``
  Typed matching rules over file, directory, and extension text. Metadata rules
  extract semantic fields after filtering.

``NamedSourceBinding``
  Assigns an alias and component identity to a selected source. Bindings can
  represent matched sources or values broadcast across an image set.

``SourceBindingsConfig``
  Pipeline-level source declarations, imported metadata, grouping metadata, and
  explicit image-plane sources.

``StepSourceBindingsConfig``
  Step-level choice of which named source views participate in that step.

``MetadataExtractionRule``
  A regex over file or folder names with named capture groups. The captured
  values become typed source metadata used for matching and grouping.

``SourceBindingMatchPlan``
  The explicit order- or metadata-based rule for pairing multiple aliases into
  one logical source set.

Compilation
-----------

The compiler resolves declarations into ``CompiledSourceBindingPlan`` and
``CompiledSourceUniversePlan`` fields on each ``CompiledStepPlan``. The source
universe establishes which files and semantic source identities are available;
the binding plan selects the views required by the step.

Source artifact inputs can be satisfied by bindings or metadata and therefore
need not have a runtime artifact plan. The path planner owns that satisfaction
decision. Artifact selectors validate exact plans that are present without
inventing plans for source-satisfied inputs.

Virtual workspace projection
----------------------------

Microscope handlers and PolyStore can expose a virtual workspace whose paths
refer to source pixels held by another backend. ``SourcePixelRef`` is the
PolyStore-owned backend reference. OpenHCS projects workspace metadata into
``VirtualWorkspaceSourceProjection``, which provides:

- virtual path to backend source reference mapping
- source metadata and nominal source projection per path
- pipeline-start files for each execution axis
- payload provenance and leading-axis composition mode
- exact address resolution through the configured FileManager

The projection is part of ``CompilationSession`` so every axis compiles against
the same explicit view it will use at runtime.

A backend address and a physical source path are different projections.
PolyStore's ``BackendBase.resolve_listed_address()`` owns normalization of
addresses returned by backend listings, while
``BackendBase.physical_source_path()`` optionally projects an address onto a
host file. The latter is absent by default and composes the strict
``DataSource.source_path()`` contract only for physical data sources. Virtual
backends such as OMERO therefore retain their backend address as source and
provenance identity without pretending that it is a local file. Runtime image
metadata asks the backend owner for the optional physical path and, when none
exists, derives available dtype and intensity facts from the loaded pixels.
Generic OpenHCS code must not recover this distinction from backend names or
path-string syntax.

Agents should validate the projection rather than merely validate Python
syntax: inspect required and optional alias counts, unmatched sources, metadata
capture values, source-set keys, component identities, virtual paths, backing
source paths, and stack composition on a representative subset.

Diagnostics and bounded inspection
----------------------------------

``SourceProjectionSet`` retains both validated projections and typed
``SourceDatasetDiagnostic`` values. A store-specific diagnostic leaf owns its
canonical payload; generic workspace, microscope, metadata, and MCP projections
carry that payload without matching concrete class names or reconstructing its
fields. Diagnostics are evidence about the exact selected dataset and remain
attached when the source projection crosses the virtual-workspace boundary.

For example, Bio-Formats represents an excluded packed-RGB reader series as
``BioFormatsPackedRgbSeriesExclusion``. It records the source files, OME image
identity, series index, and reader-declared RGB channel count. That exclusion is
not silently converted into scalar microscopy channels. A source containing no
valid scalar planes, or multiple unresolved dataset/sample identities, fails
with its typed source error instead of selecting a guessed series.

OpenHCS source inspection delegates bounded pixel reads and resolution
provenance to PolyStore's sampling contract. The source model owns the
microscopy identity and diagnostic interpretation; PolyStore owns the sampling
request/result, backend address, selected resolution, and statistics scope.
Inspection code should sample a representative region rather than materialize a
rich container merely to discover its layout.

Provenance and axes
-------------------

Runtime image payloads carry source metadata and source-plane provenance.
Projection code uses declared plane axes and component metadata; raw ndarray
shape is not enough to recover semantic source identity. Stack and bundle
composition are distinguished explicitly when multiple named source aliases are
combined.

Image-file semantics
--------------------

``ImageFileFormat`` is the OpenHCS nominal root for domain image-file
semantics. Each registered format owns its suffixes, pixel preparation,
read/write behavior, source dtype and intensity scale, and any explicit
pixel-band channel axis/count. ``ImageFileSourceMetadata`` carries those facts
into source projection without inferring a semantic channel from array rank.

This authority is distinct from PolyStore. PolyStore owns generic backend
addressing, persistence, and storage-format mechanics; OpenHCS owns how an image
container's pixels and intensity metadata participate in microscopy source and
runtime semantics. New image formats extend ``ImageFileFormat`` and its nominal
registry rather than adding suffix tables to source consumers.

CellProfiler import
-------------------

CellProfiler setup modules contribute source declarations while lowering a
``.cppipe``. The importer constructs ``SourceBindingsConfig`` directly and folds
it into the returned ``PipelineConfig``. Executable modules then refer to those
ordinary source-bound artifacts.

Package boundary
----------------

PolyStore owns FileManager, backend address resolution, ``SourcePixelRef``, and
generic virtual-workspace/storage and bounded-sampling mechanics. OpenHCS owns
``ImageFileFormat``, microscope metadata, typed source diagnostics, semantic
source bindings, source universes, component identity, and how these facts
become compiler and runtime plans.
