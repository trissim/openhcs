Data dimensions and execution axes
==================================

High-content microscopy data varies over semantic components such as well,
site, channel, Z plane, and timepoint. OpenHCS keeps those identities separate
from raw ndarray dimensions.

Plate and execution axis
------------------------

The orchestrator discovers component keys through the microscope/source model.
One configured component—normally well—is the multiprocessing execution axis.
Compilation creates one context and typed plan set for each selected axis value.

Image stack axis
----------------

Image callables receive 3D image data, including logical single-plane inputs.
``variable_components`` declares which semantic components vary along the stack
axis. The axis might represent sites, channels, Z planes, or another declared
component; it must not be called Z merely because it occupies array axis 0.

Grouping
--------

``group_by`` partitions assembled arrays after stack construction. Dictionary
function patterns route groups to different callable patterns. A non-dictionary
pattern does not gain different callable semantics from grouping.

Processing locality
-------------------

An array that contains several Z planes is not automatically treated as one
volumetric object domain. Component axes describe what the data contains;
callable declarations describe how a function consumes it.

``ProcessingContract`` controls how a callable handles the assembled runtime
stack independently of the component identities:

- ``PURE_2D``: each plane is semantically independent;
- ``PURE_3D``: the callable depends on the full stack;
- ``FLEXIBLE``: an explicit control selects either semantic mode;
- ``VOLUMETRIC_TO_SLICE``: a stack is consumed into a collapsed plane domain.

Other callable declarations can preserve a complete image payload or a
full-stack object-label input. The effective behaviour therefore comes from
the callable's complete contract, not from array shape or one enum in
isolation. For example, the CellProfiler-compatible Watershed route consumes a
complete 3D payload, whereas ``IdentifyPrimaryObjects``,
``IdentifySecondaryObjects``, and ``IdentifyTertiaryObjects`` currently operate
plane by plane.

Object identity across Z
------------------------

Object labels make the distinction explicit:

``PAYLOAD`` domain
  One object-ID domain applies across the complete label payload. A label may
  span several Z planes and volumetric measurements produce one row for that
  object.

``PLANE`` domain
  Each outer plane has an independent object-ID domain. The same integer label
  on two planes does not imply the same biological object, and OpenHCS does not
  silently stitch those labels into a volume.

This is why a pipeline can contain true volumetric segmentation and
measurement steps alongside plane-local steps without a global 2D/3D mode.
Choose functions whose declared dimensional behaviour matches the biological
question.

Runtime identity
----------------

Runtime payloads carry a declared plane axis, component/source metadata, and
source-plane provenance. Artifact keys additionally carry execution-axis and
optional component-group scope. Projection validates those declarations; it
does not infer identity from shape or a filename suffix.

Example
-------

.. code-block:: python

   from openhcs.constants import GroupBy, VariableComponents
   from openhcs.core.config import LazyProcessingConfig, ProcessingConfig

   processing = ProcessingConfig(
       variable_components=(VariableComponents.SITE,),
       group_by=GroupBy.CHANNEL,
   )
   step = FunctionStep(
       func={"1": nuclei, "2": neurites},
       processing_config=LazyProcessingConfig.from_config(processing),
   )

Here, site defines the stack axis and channel partitions those site stacks for
dictionary routing. Neither setting alone declares whether ``nuclei`` or
``neurites`` has per-plane or whole-stack semantics; their callable contracts
do that.

See :doc:`../reference/dimensionality_and_measurements` for the current
capability boundary and :doc:`../architecture/processing_semantics` for the
compiler/runtime model.
