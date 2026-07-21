Introductory plate workflow
===========================

This walkthrough introduces the current desktop workflow using a synthetic
plate. It focuses on the declaration model; labels can vary slightly by desktop
theme and release.

Create and initialize a plate
-----------------------------

1. Launch ``openhcs``.
2. Choose **View > Generate Synthetic Plate** and create the example dataset.
3. In Plate Manager, select the new plate and initialize it.
4. Open the Image Browser to confirm that wells, sites, channels, and other
   source components were discovered correctly.

Initialization selects a registered microscope handler and builds the source
workspace used by compilation. Resolve source/metadata errors here before
editing the pipeline.

Build the pipeline
------------------

Open the Pipeline Editor for the plate. Add or edit an ordered set of steps.
Each ``FunctionStep`` contains a callable pattern plus nested configuration.

For a stitching operation, ``variable_components`` usually includes ``SITE``:
the transported stack then contains the sites that belong to one otherwise
fixed coordinate. This tells the callable what positions on its third axis
mean.

``group_by`` is different. It partitions arrays after stack construction and
only routes to different callables when the function pattern is a dictionary.
Do not select a group merely to create the site stack.

Choose functions and parameters
-------------------------------

Open a step's function-pattern editor and select a registered callable. The
editor shows parameters derived from its signature and callable contract. A
chain runs several callables in order; a dictionary pattern selects a callable
for each compiled group key.

Materialize or stream results
-----------------------------

Enable step materialization when an image result must be saved. Enable Napari
or Fiji streaming when it should be displayed during execution. These outputs
are compiler-planned; the viewer is not called directly by the pipeline.

Compile, then run
-----------------

Compile the selected plate before execution. Compilation checks the source,
component semantics, callable and artifact contracts, memory/device needs, and
output plans. Fix the first reported declaration error, then compile again.
Run the compiled selection and inspect progress, logs, viewers, and materialized
outputs.

Next, read :doc:`configuration_reference`,
:doc:`../concepts/data_dimensions`, and
:doc:`../user_guide/real_time_visualization`.
