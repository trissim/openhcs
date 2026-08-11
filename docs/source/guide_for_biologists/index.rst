OpenHCS for imaging scientists
==============================

This section is for scientists who want to assess OpenHCS, complete a first
workflow, and then use it on microscopy data without starting from Python.

Start here
----------

1. Read :doc:`intro` for the purpose and fit of OpenHCS.
2. Use :doc:`../getting_started/getting_started` to install and launch it.
3. Complete :doc:`intro_stitching`, a bounded first-workflow tutorial with
   visible checkpoints.

Learn by doing
--------------

:doc:`intro_stitching`
  Generate a synthetic plate, compile the included eight-step pipeline, run it,
  and inspect the result. Follow this one path before adapting OpenHCS to your
  own data.

Complete a task
---------------

- :doc:`image_sources` — identify and name microscopy inputs.
- :doc:`../getting_started/getting_started` — install, launch, verify, or update
  OpenHCS.
- :doc:`troubleshooting_FAQ` — recover from common launch, source, compilation,
  and viewer problems.
- :doc:`../user_guide/index` — edit pipelines, functions, viewers, results, and
  local agent integrations.

Look up facts
-------------

- :doc:`basic_interface` — desktop windows and controls.
- :doc:`../reference/configuration` — exact configuration fields, defaults,
  and accepted values.
- :doc:`glossary` — microscopy and OpenHCS terminology.

Understand the workflow
-----------------------

- :doc:`domain_expert_onboarding` — decide whether OpenHCS fits your data and
  analysis goal.
- :doc:`configuration_reference` — understand configuration scopes,
  inheritance, and resolution.
- :doc:`../concepts/domain_fundamentals` — understand why source dimensions,
  compilation, and typed results matter.
- :doc:`../concepts/pipelines_and_steps` — understand the pipeline declaration
  model.


.. toctree::
   :hidden:
   :maxdepth: 20

   intro
   domain_expert_onboarding
   installation_and_setup
   basic_interface
   image_sources
   configuration_reference
   glossary
   troubleshooting_FAQ
