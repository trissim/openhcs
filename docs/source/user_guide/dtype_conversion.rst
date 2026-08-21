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

In **Global Configuration**, open **Dtype Config** and set **Default Dtype
Conversion** to choose the application default. To override one pipeline or
step, open its configuration, expand **Dtype Config**, and set the same field at
that narrower scope. Leave the lazy field unset to inherit the broader value.

The equivalent step declaration is:

.. code-block:: python

   from arraybridge.decorators import DtypeConversion
   from openhcs.core.config import LazyDtypeConfig

   step.dtype_config = LazyDtypeConfig(
       default_dtype_conversion=DtypeConversion.PRESERVE_INPUT,
   )

Compile again after changing the policy. Do not cast merely to cross
NumPy/CuPy/Torch boundaries; memory conversion and dtype policy are separate
declarations.

ArrayBridge owns generic dtype conversion behavior. OpenHCS records the resolved
configuration in the callable/runtime plan. See
:doc:`../concepts/dtype_config_system` for the policy boundary and
:doc:`../guides/memory_type_integration` for framework conversion.
