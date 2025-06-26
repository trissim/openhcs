Pipeline Factory
==============

.. module:: ezstitcher.core

This module contains the AutoPipelineFactory class that creates pre-configured pipelines
for all common workflows, leveraging specialized steps to reduce boilerplate code.

For conceptual explanation, see :doc:`../concepts/pipeline_factory`.
For information about pipeline configuration, see :doc:`../concepts/pipeline`.

AutoPipelineFactory
-----------------

.. autoclass:: AutoPipelineFactory
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: create_pipelines
