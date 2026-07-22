Loading and naming image sources
================================

An OpenHCS plate can contain one image store or several stores together. Two
separate decisions are involved:

1. a microscope or store handler owns **ingestion** and publishes addressable
   planes;
2. source bindings optionally **select and name** those planes for pipeline
   inputs.

For a recognized HCS layout, OpenHCS keeps the native microscope handler. CZI,
OME-TIFF, and other supported rich containers are decoded by the Bio-Formats
store handler using their embedded metadata. Add the containing directory to
the Plate Manager and initialize with auto-detection; a source binding is not
required merely to open or browse the store.

``SourceBindingsHandler`` has a different role: it is the ingestion fallback for
an arbitrary image folder whose layout is not recognized. A non-empty
``PipelineConfig.source_bindings_config`` tells that handler which ordinary
TIFF, PNG, JPEG, or other registered files form the source universe and how
filename metadata should be interpreted.

OpenHCS asks each image store to publish addressable planes before applying the
bindings. A binding selects those planes by declared component metadata or by
their original file provenance and gives them a stable name such as ``DNA`` or
``Mask``. Pipeline declarations do not traverse CZI, OME-TIFF, or OME-Zarr
internals and do not select a storage backend.

This means ``SourceBindingsConfig`` can still name channels or samples from a
CZI after Bio-Formats has decoded it. In that case the Bio-Formats handler remains
the ingestion owner; the binding is semantic selection, not a replacement CZI
loader.

Native layout versus complete metadata
----------------------------------------

Filename/layout recognition and automatic detection are related but distinct.
Run ``openhcs_inspect_plate_path`` with ``microscope_type="auto"`` first. Its
``format_specific_handler_candidates`` are derived live from the registered
``MicroscopeHandler`` parsers and detection contracts. If a broad store handler
was selected but a format-specific candidate recognizes every tested physical
filename under its declared ``root_dir``, do not assume the broad handler is the
better semantic owner merely because native metadata is missing.

Opera Phenix is a representative case. A complete plate export contains the
``Images/`` layout plus ``Index.xml``; that metadata enables native automatic
detection, grid geometry, pixel size, and other plate facts. A deliberately
small subset may contain correctly named files such as
``r04c09f11p01-ch1sk1fk1fl1.tiff`` under ``Images/`` without ``Index.xml``. For
native Opera Phenix semantics, obtain the complete export. The Opera Phenix
owner requires its ``Index.xml`` detection contract; matching filenames alone do
not make the subset a valid native plate. If those TIFFs are intentionally being
treated as loose ordinary images, select them with ``SourceBindingsConfig`` and
declare their Well/Site/Channel/Z/Time identities instead of explicitly selecting
``microscope_type="opera_phenix"`` or accepting a broad decoder's inferred sample
layout.

The folder-onboarding authoring context lists every currently registered handler
and its handler-owned selection role. Use the format-specific owner for recognized
vendor layouts, the broad structured-store owner for supported rich containers
without a stronger native match, and the declared-file fallback for arbitrary
ordinary files whose schema you provide.

``source_stack_components`` describes axes physically present inside one selected
file or store payload. Do not declare ``SITE`` or another source-stack axis when
each site is a separate ordinary 2-D file. Give each file its component identity,
then use ``processing_config.variable_components`` to assemble those selected files
along the callable's stack axis.

Coordinates and source identity
-------------------------------

Every resolved plane has one sample identity represented by ``WELL``. This is the
embedded plate well when the image metadata declares one; otherwise it is the
exact source-container identity. ``SITE``, ``CHANNEL``, ``Z_INDEX``, and
``TIMEPOINT`` identify the plane within that sample. If a store declares an axis
absent or singleton, its coordinate is ``"1"``. Embedded coordinates are never
replaced by filename guesses, and conflicting plane identities stop
initialization with an explicit error.

Each plane also retains a typed reference to its owning store. The image browser,
metadata browser, compiler, and runtime all consume the same resolved plate
projection. Saving a changed source-binding config invalidates that projection;
the next normal initialization rebuilds it and updates the available aliases and
coordinates without a separate UI metadata copy.

Binding positions are not microscope channel values
---------------------------------------------------

Keep three layers separate:

pipeline source universe
  ``PipelineConfig.source_bindings_config`` declares the full named physical
  source universe and the inputs consumed by the nominal handler projection. It
  may correctly contain Hoechst channel 1, MAP2 channel 2, and SMI312 channel 4
  even when one step needs only two of them.
  Seeing all three in the source workspace inventory is valid.

per-step binding plan
  Resolved ``FunctionStep.source_bindings`` selects and orders the subset for one
  invocation after config inheritance. Compilation must preserve that subset and
  order in ``CompiledSourceBindingPlan.bindings``.

callable stack positions
  ``processing_config.variable_components`` owns the meaning of the assembled
  array axis. A callable ``channel_index`` is a zero-based position on that
  invocation's assembled axis, not a physical microscope ``CHANNEL`` value.

When viewer routes carry physical component identities from the first layer,
they retain the declared values; they do not renumber them to the third layer's
zero-based positions.

For example, the pipeline universe may be ``(Hoechst[channel=1],
MAP2[channel=2], SMI312[channel=4])``. A MetaXpress neurite step has two useful
declaration choices:

simplest inherited full stack
  Set ``input_source=PIPELINE_START`` and
  ``variable_components=[CHANNEL]``, but omit ``FunctionStep.source_bindings``.
  The step inherits the complete pipeline order, so use
  ``nuclear_stain.channel_index=0``, ``cell_body.channel_index=1``, and
  ``neurite_channel_index=2``. Source provenance retains physical channels 1,
  2, and 4; viewer routes expose those values when the output carries them.

legacy shared-signal subset
  Explicitly enable step source bindings and order them as ``(SMI312,
  Hoechst)``. The callable then uses ``neurite_channel_index=0`` and
  ``nuclear_stain.channel_index=1`` while leaving
  ``cell_body.channel_index`` omitted, so SMI312 supplies both the body and
  neurite signal. Source provenance remains physical channels 4 and 1. This is
  not equivalent to MAP2-seeded analysis: retain MAP2 and use the inherited
  three-channel form whenever MAP2 owns the neuronal bodies.

In either option, ``variable_components=[CHANNEL]`` assembles the channel stack.
Do not use ``group_by=CHANNEL`` for assembly. ``group_by`` partitions an already
assembled value and selects branches only for a dictionary function pattern.
For this non-dictionary MetaXpress callable, an overlapping
``group_by=CHANNEL`` is redundant and the compiler normalizes it to
``GroupBy.NONE``.

The explicit-subset regression is intentionally generic rather than a
MetaXpress channel map: an implicit-main-flow callable must retain its ordered
primary-plane bindings even when it also declares special artifact inputs. That
same rule protects any explicit source subset or reorder.

If a step that selects two bindings appears as three physical channels in the
viewer, do not compensate by guessing a third callable index. Diagnose the three
layers in order:

1. Inspect the pipeline source universe and its workspace inventory. The MAP2
   channel 2 entry is valid here.
2. Read the resolved ``FunctionStep.source_bindings`` after inheritance and
   confirm the intended subset and exact order.
3. Inspect ``CompiledSourceBindingPlan.bindings`` for that step. An unexpectedly
   empty or broader plan means the per-step selection was not preserved; the
   callable may then receive all three universe planes.
4. Check the runtime-matched files for that step and their exact component
   metadata.
5. Query the current execution's raw viewer payloads and compare each
   ``layer_route_key`` and ``payload_route_key`` with its physical component
   values. A persistent viewer can also contain routes from an earlier
   submission.

MAP2 channel 2 in the full source workspace is therefore not itself a leak. It
is a leak on a current route for the explicit selected-stack option because that
step's resolved and compiled plan should select only SMI312 and Hoechst. MAP2 on
a route from the inherited-full-stack option, an older submission, or another
step that selects MAP2 is valid evidence for that route. None of these cases
changes the zero-based positions of the selected callable stack.

Step source subsets and reordering select or re-enter original declared sources,
especially at ``PIPELINE_START`` or on a branch that deliberately returns to the
source universe. They do not reinterpret a previous step's output. Stitching,
Z projection, channel projection, or filtering may change that output's slice
count and axes; downstream code must use the current artifact provenance plus
its own ``variable_components`` declaration. ``group_by`` then partitions that
already assembled downstream value.

Executable code-mode declarations
---------------------------------

The following block constructs a complete Pipeline Editor code document for a
directory that mixes stores. The intermediate config values deliberately show
both source-binding roles:

- ``tiff_png_config`` can drive ``SourceBindingsHandler`` when the directory is
  an otherwise unrecognized collection of ordinary files;
- the CZI, OME-TIFF, and OME-Zarr configs name/select planes after their store
  handlers decode them.

Choose one as ``pipeline_config`` and keep the ``pipeline_steps`` assignment in
the same document. Filenames are exact examples; replace them with names present
directly under the selected plate directory.

.. code-block:: python

   from openhcs.constants.input_source import InputSource
   from openhcs.core.config import (
       LazyProcessingConfig,
       LazyStepSourceBindingsConfig,
       PipelineConfig,
   )
   from openhcs.core.source_bindings import (
       NamedSourceBinding,
       SourceBindingsConfig,
       SourceFilterClause,
       SourceFilterMatchType,
       SourceFilterSubject,
       SourceSelector,
   )
   from openhcs.core.steps.function_step import FunctionStep
   from openhcs.processing.backends.processors.numpy_processor import (
       stack_percentile_normalize,
   )


   def bind_file(alias: str, filename: str) -> NamedSourceBinding:
       return NamedSourceBinding(
           alias=alias,
           selector=SourceSelector(
               filters=(
                   SourceFilterClause(
                       subject=SourceFilterSubject.FILE,
                       match_type=SourceFilterMatchType.EQUALS,
                       value=filename,
                   ),
               ),
           ),
       )


   tiff_png_config = PipelineConfig(
       source_bindings_config=SourceBindingsConfig(
           bindings=(
               bind_file("DNA", "nuclei.tif"),
               bind_file("Mask", "segmentation.png"),
           ),
       ),
   )

   czi_config = PipelineConfig(
       source_bindings_config=SourceBindingsConfig(
           bindings=(bind_file("DNA", "experiment.czi"),),
       ),
   )

   ome_tiff_config = PipelineConfig(
       source_bindings_config=SourceBindingsConfig(
           bindings=(bind_file("DNA", "plate.ome.tif"),),
       ),
   )

   ome_zarr_config = PipelineConfig(
       source_bindings_config=SourceBindingsConfig(
           bindings=(bind_file("DNA", "plate.zarr"),),
       ),
   )

   mixed_store_config = PipelineConfig(
       source_bindings_config=SourceBindingsConfig(
           bindings=(
               bind_file("DNA", "plate.zarr"),
               bind_file("Brightfield", "brightfield.tif"),
               bind_file("Mask", "labels.png"),
           ),
       ),
   )

   pipeline_config = mixed_store_config

   pipeline_steps = [
       FunctionStep(
           name="Normalize DNA",
           func=stack_percentile_normalize,
           processing_config=LazyProcessingConfig(
               input_source=InputSource.PIPELINE_START,
           ),
           source_bindings=LazyStepSourceBindingsConfig(
               enabled=True,
               bindings=(NamedSourceBinding(alias="DNA"),),
           ),
       ),
   ]

After applying the code, initialize the selected plate normally with auto-
detection. OpenHCS first chooses the native, Bio-Formats/store, or arbitrary-
folder ingestion owner, then resolves every binding over the planes that owner
published. CZI and embedded OME metadata require the Bio-Formats runtime included
by an OpenHCS installation that enables those formats. Do not force a CZI or OME
container through ``SourceBindingsHandler`` when its structured decoder is
missing or unhealthy; repair that decoder instead.

Selecting channels and samples
------------------------------

Use component selectors when the store already declares the coordinate you need.
This config fragment names channel ``2`` from well ``A01`` without encoding
either value in a format-specific path rule. Use it as ``pipeline_config`` in
the complete document above; source-backed MCP execution and Pipeline Editor
code mode must also receive ``pipeline_steps``.

.. code-block:: python

   from openhcs.constants.constants import AllComponents
   from openhcs.core.config import PipelineConfig
   from openhcs.core.source_bindings import (
       ComponentSelector,
       NamedSourceBinding,
       SourceBindingsConfig,
       SourceSelector,
   )

   pipeline_config = PipelineConfig(
       source_bindings_config=SourceBindingsConfig(
           bindings=(
               NamedSourceBinding(
                   alias="DNA",
                   selector=SourceSelector(
                       components=(
                           ComponentSelector(AllComponents.WELL, "A01"),
                           ComponentSelector(AllComponents.CHANNEL, "2"),
                       ),
                   ),
               ),
           ),
       ),
   )

Keep ``required=True`` (the default) when a missing source should stop
initialization. This makes misspelled filenames, absent channels, incompatible
datasets, and coordinate collisions visible before a pipeline runs.

Diagnostics and rich containers
-------------------------------

Plate inspection reports structured source diagnostics alongside the resolved
coordinates. Treat them as part of the dataset result, not as log decoration.
For Bio-Formats inputs, a packed-RGB series is reported with its exact OME image
and series identity and excluded from scalar microscopy planes. View or extract
that series with an RGB-capable tool; do not reinterpret its packed color bands
as independent OpenHCS channels.

If a container exposes only excluded/non-scalar series, or if its embedded
dataset or sample identity is ambiguous, initialization stops with a typed
error. Resolve the decoder or choose the exact dataset from authoritative
metadata instead of guessing a series number from filenames. Use bounded image
sampling to confirm a representative plane and its selected-resolution
provenance before authoring a full pipeline.
