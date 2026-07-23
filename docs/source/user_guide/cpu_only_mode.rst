CPU-only mode
=============

Install OpenHCS without the ``gpu`` extra and set ``OPENHCS_CPU_ONLY=true``
before starting the application or test process:

.. code-block:: bash

   OPENHCS_CPU_ONLY=true openhcs

CPU-only mode restricts function discovery to CPU-compatible declarations and
prevents worker startup from importing GPU backends. A pipeline containing a
GPU-only callable should fail during selection or compilation rather than fall
back silently to an unrelated implementation.

For reproducible CPU environments, also avoid installing optional CUDA
libraries. ``CUDA_VISIBLE_DEVICES`` controls device visibility in the usual way,
but it is not a replacement for the OpenHCS discovery policy.

Use NumPy-backed callables or an explicitly CPU-capable implementation. Memory
framework choice is owned by callable contracts; do not patch the function
registry or compiled plans to force a backend.
