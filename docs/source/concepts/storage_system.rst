Storage and materialization
===========================

OpenHCS separates semantic artifact identity from storage. Pipeline functions
do not choose filenames or storage backends directly; declarations and compiler
plans determine where values live and which outputs persist.

Three storage intents
---------------------

``read_backend``
  How plate/source inputs are read. ``AUTO`` lets the microscope and metadata
  integration select the appropriate backend.

``intermediate_backend``
  Where ordinary step results are stored between steps. Memory is the normal
  default, but a compiled plan may select another backend when required.

``materialization_backend``
  Where explicitly materialized outputs persist, such as disk, Zarr, or an
  application-specific backend.

These settings live in ``VFSConfig`` and its lazy projections. PolyStore owns
the generic ``ZarrConfig``, compressor factory registry, and chunk-strategy
enum. OpenHCS contributes a fieldless registered subtype so those exact storage
identities participate in global, pipeline, and step configuration without a
second enum or field declaration.

Artifacts first, paths second
-----------------------------

Callable and module contracts declare typed artifact inputs and outputs. The
artifact graph resolves producers and dependencies. Path planning then adds
backend addresses and materialization targets to typed plans.

A Python output-slot name or ``special_outputs`` compatibility annotation does
not make a value persistent. Materialization is an explicit configuration and
artifact-plan decision.

Runtime storage
---------------

``RuntimeValueStore`` records typed runtime values under semantic artifact keys
and exact backend locations. A later producer may explicitly replace the current
binding while observation history is retained. Consumers use compiled typed
queries rather than searching filenames.

PolyStore boundary
------------------

PolyStore owns generic ``FileManager`` behavior, backend registration and
lifecycle, formats, ROI, virtual workspaces, ``SourcePixelRef``, address
resolution, and Zarr backend configuration mechanics. OpenHCS owns:

- backend intent in pipeline configuration;
- source-binding and virtual-workspace semantic projection;
- artifact and materialization plans;
- application-specific OMERO and viewer adapters;
- runtime-store consistency with compiled paths.

Output discovery
----------------

Use the Plate Manager, artifact-plan inspection, runtime observations, and the
orchestrator's result-path surface to find outputs. Do not assume every step
creates a disk directory or that the physical layout itself expresses artifact
semantics.

See :doc:`../architecture/artifact_contract_system`,
:doc:`../architecture/runtime_value_system`, and
:doc:`../architecture/external_foundations`.
