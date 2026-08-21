CPU-only mode
=============

Install OpenHCS without the ``gpu`` extra. For a desktop session, pass
``--no-gpu``:

.. code-block:: bash

   openhcs --no-gpu

For any OpenHCS entry point, set the process-wide authority before startup:

.. code-block:: bash

   OPENHCS_CPU_ONLY=true openhcs

Both forms select the same CPU-only mode before optional runtimes are imported.
The mode bypasses GPU inventory, restricts function discovery to CPU-compatible
registry declarations, and prevents the application, execution server, and
workers from importing GPU backends. A pipeline containing a GPU-only callable
should fail during selection or compilation rather than fall back silently to
an unrelated implementation.

For reproducible CPU environments, also avoid installing optional CUDA
libraries. ``CUDA_VISIBLE_DEVICES`` controls device visibility in the usual way,
but it is not a replacement for the OpenHCS discovery policy.

Use NumPy-backed callables or an explicitly CPU-capable implementation. Memory
framework choice is owned by callable contracts; do not patch the function
registry or compiled plans to force a backend.
