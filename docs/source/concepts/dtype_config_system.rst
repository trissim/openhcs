Data-type conversion configuration
==================================

``DtypeConfig`` is the OpenHCS configuration projection of ArrayBridge's nominal
``DtypeConversionConfig``. It controls how decorated callables convert their
output data type.

Modes
-----

``NATIVE_OUTPUT``
  Preserve the callable/framework's natural output dtype. This is the OpenHCS
  pipeline default.

``PRESERVE_INPUT``
  Convert the result back to the input dtype using ArrayBridge's declared
  scaling behavior.

Explicit dtype members
  Force a supported target such as ``UINT8``, ``UINT16``, ``INT16``, ``INT32``,
  ``FLOAT32``, or ``FLOAT64`` when the installed ArrayBridge version exposes
  that member.

Configuration
-------------

.. code-block:: python

   from arraybridge.decorators import DtypeConversion
   from openhcs.core.config import DtypeConfig

   preserve = DtypeConfig(
       default_dtype_conversion=DtypeConversion.PRESERVE_INPUT,
   )

Global, pipeline, and step ObjectState scopes can provide inherited dtype
configuration. A step-level value is passed through the ``dtype_config`` field
of ``FunctionStep``; it is independent of ``processing_config``.

Callable boundary
-----------------

ArrayBridge memory decorators expose a runtime-bound ``dtype_config`` parameter
through the callable contract. OpenHCS resolves the inherited configuration and
the compiler plans any memory conversion. User function kwargs should not supply
or shadow runtime-bound parameters.

Ownership
---------

ArrayBridge owns conversion runners, dtype scaling, framework-specific transfer,
and direct-call defaults. OpenHCS owns configuration inheritance and how the
resolved value enters callable contracts and compiled execution.

See :doc:`function_library` and
:doc:`../architecture/external_foundations`.
