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

``ProcessingContract`` is independent of the component axes:

- ``PURE_2D``: each plane is semantically independent;
- ``PURE_3D``: the callable depends on the full stack;
- ``FLEXIBLE``: an explicit control selects either semantic mode;
- ``VOLUMETRIC_TO_SLICE``: a stack is consumed into a collapsed plane domain.

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

See :doc:`../architecture/processing_semantics` for the compiler/runtime model.
