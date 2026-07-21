Automatic dtype conversion
==========================

``DtypeConfig`` controls how decorated callables reconcile input and output
dtypes. The setting is inherited through pipeline and step ObjectState scopes
and is resolved during compilation.

The two primary policies are:

``NATIVE_OUTPUT``
  Keep the callable/framework's native output dtype without intensity scaling.

``PRESERVE_INPUT``
  Convert the result back toward the input dtype using ArrayBridge's declared
  conversion policy.

Set the global or pipeline default in the desktop configuration form, then
override the dtype configuration only on steps that need a different policy.
Do not cast merely to cross NumPy/CuPy/Torch boundaries; memory conversion and
dtype policy are separate declarations.

ArrayBridge owns generic dtype conversion behavior. OpenHCS records the resolved
configuration in the callable/runtime plan. See
:doc:`../guides/memory_type_integration`.
