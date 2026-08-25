Component-system integration
============================

The microscope/source model discovers component values for a plate. The
compiler carries those values into source universes, execution-axis contexts,
stack construction, grouping, and filtering.

The same component declarations have distinct jobs:

``variable_components``
  Determines what each position on the transported third array axis means.

``group_by``
  Partitions already assembled inputs and controls dictionary-pattern routing.

``sequential_components``
  Chooses dimensions processed in bounded sequential slices.

``multiprocessing_axis``
  Partitions independent compiled contexts, normally by well.

Metadata caches and source projections are the authority for values available
on a particular plate. UI selectors, compiler stages, and runtime filters query
those projections; they do not invent component keys or depend on a concrete
microscope module.

Complete component values cross internal boundaries as
``OpenHCSComponentValues`` keyed by ``AllComponents`` members in declaration
order. Filename parsers carry that value set in ``FilenameParseResult`` and
metadata handlers project it through ``MetadataComponentValueSet``. Missing
values remain explicit entries rather than making each consumer maintain a
separate list of axes. String-keyed mappings are produced only at declared wire
or persistence boundaries.

See :doc:`component_validation_system`, :doc:`source_model`, and
:doc:`processing_semantics`.
