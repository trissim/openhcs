OpenHCS documentation
=====================

OpenHCS is designed for imaging scientists and research software teams that
need to turn plate-based microscopy data into measurements they can review and
rerun. It keeps source selection, processing steps, and result definitions in
one validated pipeline across the desktop GUI, Python, supported CellProfiler
imports, and local agents.

Is OpenHCS right for your work?
-------------------------------

Start with :doc:`guide_for_biologists/domain_expert_onboarding` if your data is
organised by plates, wells, sites, channels, Z planes, or time points and you
need a repeatable analysis rather than a one-off manual inspection. It explains
where OpenHCS fits and what information to collect before building a pipeline.

Choose by what you need now
---------------------------

**Learn by doing — tutorial**
  Follow :doc:`guide_for_biologists/intro_stitching` to generate a bounded
  synthetic plate, compile its included pipeline, run it, and inspect the
  result. This is the shortest path to a complete first workflow.

**Complete a task — how-to guides**
  Use :doc:`getting_started/getting_started` to install and launch OpenHCS,
  then choose a task from :doc:`user_guide/index` or :doc:`guides/index`.
  These pages assume you know the outcome you need.

**Look up exact facts — reference**
  Use :doc:`api/index` for the supported Python boundary,
  :doc:`guide_for_biologists/configuration_reference` for configuration fields,
  and :doc:`appendices/glossary` for terminology.

**Understand the model — explanation**
  Read :doc:`concepts/index` for the scientific and pipeline model. Maintainers
  and integrators should continue with :doc:`architecture/quick_start` and
  :doc:`architecture/index` for ownership and runtime boundaries.

Documentation by need
---------------------

.. toctree::
   :maxdepth: 2
   :caption: Start here

   guide_for_biologists/index

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   guide_for_biologists/intro_stitching

.. toctree::
   :maxdepth: 2
   :caption: How-to guides

   getting_started/getting_started
   user_guide/index
   guides/index

.. toctree::
   :maxdepth: 2
   :caption: Explanation

   concepts/index
   architecture/index

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/index
   reference/index
   appendices/index

.. toctree::
   :maxdepth: 2
   :caption: Contributing and extending

   development/index

.. toctree::
   :hidden:

   installation

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
