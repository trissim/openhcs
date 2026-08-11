Experimental-analysis workbook syntax
=====================================

The standalone experimental-analysis engine reads an Excel workbook that maps
well-level measurements to conditions, doses, biological replicates, technical
replicates, controls, exclusions, and physical plates. This page describes the
workbook accepted by the current parser. For the desktop task sequence, use
:doc:`../user_guide/experimental_layouts`.

Download :download:`experimental_config_example.xlsx
<../examples/experimental_config_example.xlsx>` as a complete workbook to edit.

Workbook structure
------------------

The current parser reads two sheets with exact names:

``drug_curve_map``
  Experimental conditions and their well assignments.

``plate_groups``
  Mapping from each biological replicate and plate-group identifier to the
  physical plate identifier present in the consolidated results.

Other sheets are ignored. In both parsed sheets, the first column contains row
labels and is loaded as the table index.

``drug_curve_map`` rows
-----------------------

Rows are stateful and must appear in this order: global parameters, optional
control or exclusion blocks, then one or more condition blocks. Blank rows may
separate blocks.

Global parameters
~~~~~~~~~~~~~~~~~

``N``
  Positive number of biological replicates. Replicates are named ``N1`` through
  ``N<N>``.

``Scope`` or ``Microscope``
  Result format. Accepted values are ``EDDU_CX5`` and
  ``EDDU_metaxpress``.

``Per Well Datapoints``
  Optional output policy. ``True``, ``1``, ``Yes``, ``On``, or ``Enabled``
  preserves individual wells. Matching is case-insensitive. Any other value,
  including an omitted row, uses averaged technical replicates.

.. code-block:: text

   N                    3
   Scope                EDDU_metaxpress
   Per Well Datapoints  False

Control block
~~~~~~~~~~~~~

Controls must appear before the first ``Condition`` row.

``Controls`` or ``Control Well``
  Control well identifiers.

``Plate Group``
  Plate-group identifier aligned with each control well.

``Group N``
  Optional biological-replicate number aligned with each control well. Without
  this row, the control well and plate-group pairs apply to every replicate.

.. code-block:: text

   Controls     A01  B01  E01  F01
   Plate Group  1    1    1    1
   Group N      1    1    2    2

Exclusion block
~~~~~~~~~~~~~~~

An exclusion block requires all three aligned rows. Excluded wells are removed
from their replicate and plate group, including from the corresponding control
set.

``Exclude Wells``
  Well identifiers to exclude.

``Plate Group``
  Plate-group identifier aligned with each excluded well.

``Group N``
  Biological-replicate number aligned with each excluded well.

.. code-block:: text

   Exclude Wells  A01  B03
   Plate Group    1    2
   Group N        1    2

Condition blocks
~~~~~~~~~~~~~~~~

``Condition``
  Condition name. A new row closes any preceding control or exclusion block and
  begins a condition.

``Dose``
  Ordered dose or treatment-level values for the condition.

``Wells``
  Ordered wells applied to every biological replicate.

``WellsN``
  Ordered wells applied only to replicate ``N``. ``WellN``, embedded spaces,
  underscores, and multi-digit identities such as ``Wells 12`` are accepted.
  The replicate number must be within ``1..N``.

``Plate Group``
  Ordered plate-group identifiers for the preceding ``Wells`` or ``WellsN``
  row.

The ``Dose``, well, and ``Plate Group`` rows in one assignment must have equal
column counts. Columns are mapped positionally: the first dose is paired with
the first well and first plate group, and so on. Repeating a ``WellsN`` row plus
its ``Plate Group`` row adds technical replicates to those same dose positions.

.. code-block:: text

   Condition    Drug_A
   Dose         0    10   50   100
   Wells1       A01  A02  A03  A04
   Plate Group  1    1    1    1
   Wells1       B01  B02  B03  B04
   Plate Group  1    1    1    1
   Wells2       A05  A06  A07  A08
   Plate Group  1    1    1    1

In this example, ``A01`` and ``B01`` are technical replicates for dose ``0``
in biological replicate ``N1``. ``A05`` is the dose-``0`` well for ``N2``.

``plate_groups`` sheet
----------------------

The first populated row is skipped as a header. Following data columns are
addressed by their one-based worksheet positions (group ``1``, group ``2``, and
so on). The first column of each later row contains a biological-replicate name;
each remaining cell is the physical plate identifier for that replicate and
group.

.. code-block:: text

             1          2
   N1  20220818   20220819
   N2  20220820   20220821
   N3  20220822   20220823

An assignment such as ``("A01", 2)`` under ``N1`` therefore reads well
``A01`` from physical plate ``20220819``.

Parser constraints
------------------

- ``N`` must be read before a condition or replicate-specific well row.
- Every condition assignment requires ``Dose``, a well row, and its following
  ``Plate Group`` row.
- Dose, well, and plate-group column counts must match exactly.
- A replicate-specific well row must select a replicate within ``1..N``.
- ``EDDU_CX5`` and ``EDDU_metaxpress`` are the current result scopes.
- The consolidated result and workbook must use the same well identifiers.

Programmatic boundary
---------------------

``ExperimentalAnalysisConfig`` configures the standalone engine. It is not a
pipeline or step configuration.

.. code-block:: python

   from openhcs.core.config import ExperimentalAnalysisConfig
   from openhcs.processing.backends.experimental_analysis import (
       ExperimentalAnalysisEngine,
   )

   config = ExperimentalAnalysisConfig()
   engine = ExperimentalAnalysisEngine(config)
   result = engine.run_directory("analysis_directory")

The returned dictionary contains the declared format, features, conditions,
feature tables, parsed experimental configuration, and processed source data.
The directory workflow projects all five input/output filenames and the
normalization method from ``config``.

The default ``results_file_name`` is the MetaXpress-style CSV used by the
desktop task. For a workbook whose scope is ``EDDU_CX5``, construct the config
with the CX5 workbook's filename before calling ``run_directory``; the workbook
scope, rather than a filename heuristic, still selects the result reader.
