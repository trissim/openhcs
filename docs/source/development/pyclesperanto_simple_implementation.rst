pyclesperanto integration
=========================

ArrayBridge owns pyclesperanto array detection, conversion, stacking, device
selection, dtype handling, and cleanup policy. OpenHCS owns callable discovery,
processing contracts, compiler integration, and its processing-function
library.

To add an OpenHCS pyclesperanto callable:

1. place it in the owning processing module;
2. import and apply the OpenHCS re-exported ``pyclesperanto`` decorator;
3. declare the correct ``ProcessingContract`` and any required axes, artifacts,
   or runtime-bound parameters;
4. let the registered library strategy expose it to discovery;
5. test its callable contract, dimensional semantics, and runtime behavior.

For example:

.. code-block:: python

   from openhcs.core.memory import pyclesperanto
   from openhcs.processing.backends.lib_registry.unified_registry import (
       ProcessingContract,
   )

   @pyclesperanto(contract=ProcessingContract.PURE_3D)
   def gaussian_volume(image, sigma: float = 1.0):
       import pyclesperanto as cle

       return cle.gaussian_blur(image, sigma_x=sigma, sigma_y=sigma, sigma_z=sigma)

The OpenHCS wrapper preserves callable metadata while ArrayBridge declares the
input, output, and execution framework roles. Compilation collects those roles,
resolves the available pyclesperanto device, and stores the framework-local
binding on the compiled step. Runtime enters that exact device scope; the
callable must not select a process-global device itself.

Do not duplicate ArrayBridge framework configuration, append the function to a
manual catalog, or make generic compiler code import the concrete
pyclesperanto module. If a framework behavior is missing, extend the
ArrayBridge authority; if an OpenHCS semantic is missing, extend the callable or
registry declaration that owns it.

See :doc:`callable_artifact_authoring`,
:doc:`../guides/memory_type_integration`, and
:external+arraybridge:doc:`ArrayBridge documentation <index>`.
