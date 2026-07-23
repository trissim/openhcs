pyclesperanto integration
=========================

ArrayBridge owns pyclesperanto array detection, conversion, stacking, device
selection, dtype handling, and cleanup policy. OpenHCS owns callable discovery,
processing contracts, compiler integration, and its processing-function
library.

To add an OpenHCS pyclesperanto callable:

1. place it in the owning processing module;
2. decorate it with the OpenHCS re-exported ``pyclesperanto`` decorator;
3. declare the correct ``ProcessingContract`` and any required axes, artifacts,
   or runtime-bound parameters;
4. let the registered library strategy expose it to discovery;
5. test its callable contract, dimensional semantics, and runtime behavior.

Do not duplicate ArrayBridge framework configuration, append the function to a
manual catalog, or make generic compiler code import the concrete
pyclesperanto module. If a framework behavior is missing, extend the
ArrayBridge authority; if an OpenHCS semantic is missing, extend the callable or
registry declaration that owns it.

See :doc:`callable_artifact_authoring`,
:doc:`../guides/memory_type_integration`, and the ArrayBridge documentation.

