Creating custom functions
=========================

Custom functions are normal Python callables with an OpenHCS memory decorator.
The decorator declares their input, output, and execution framework roles for
the callable contract used by discovery and compilation.

From the desktop application, choose **Tools > Custom Functions > Create New
Function**. The editor starts from a validated template and registers the saved
function in the OpenHCS function catalog.

.. code-block:: python

   from openhcs.core.memory.decorators import numpy
   from openhcs.processing.backends.lib_registry.unified_registry import (
       ProcessingContract,
   )

   @numpy(contract=ProcessingContract.PURE_2D)
   def subtract_background(image, *, offset=100):
       return image - offset

Choose the processing contract deliberately:

- ``PURE_2D`` means each plane is semantically independent;
- ``PURE_3D`` means the result depends on the whole stack;
- ``VOLUMETRIC_TO_SLICE`` consumes a volume and returns plane-local output;
- ``FLEXIBLE`` is for callables that explicitly support either form.

Do not attach compiler fields manually or inspect array framework names inside
the function. Declare memory, locality, required axes, runtime parameters, and
artifacts through their owning decorators/contracts. For advanced authoring,
see :doc:`../development/callable_artifact_authoring`.
