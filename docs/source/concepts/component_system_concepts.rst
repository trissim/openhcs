Component identities
====================

Components name semantic microscopy dimensions such as well, site, channel,
Z index, and timepoint. OpenHCS creates process-stable enum families from the
authoritative component configuration.

Enum families
-------------

``AllComponents``
  The complete component set, including the multiprocessing execution axis.

``VariableComponents``
  Components that may vary along a step's assembled image stack. The
  multiprocessing axis is excluded.

``GroupBy``
  Components available for grouping plus the explicit ``NONE`` member.

``SequentialComponents``
  Components available for sequential-processing policy.

Multiprocessing axis
--------------------

The configured multiprocessing axis—normally well—partitions orchestrator work.
The compiler creates a context for every selected value on this axis. It is an
``AllComponents`` member but not a ``VariableComponents`` member.

Stack and grouping use
----------------------

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

Site declares the stack axis. Channel partitions those site stacks for
dictionary routing. ``ProcessingContract`` separately declares whether each
callable has per-plane or whole-stack semantics.

Runtime scope
-------------

Compiled group identity is represented by ``ComponentGroupScope`` and runtime
artifact keys use ``RuntimeExecutionAxisScope``. Component identity is not
recovered from path text at runtime.

Extension rule
--------------

Extend the authoritative component configuration and its generic projections.
Do not create copied component lists in compiler, UI, storage, or backend code.

See :doc:`data_dimensions` and
:doc:`../architecture/processing_semantics`.
