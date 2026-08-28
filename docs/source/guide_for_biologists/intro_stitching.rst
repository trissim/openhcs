Your first plate workflow
=========================

In this tutorial you will generate a bounded synthetic microscopy plate, use
the pipeline supplied with it, compile the analysis, and inspect the completed
result. You do not need your own data or Python code.

Before you begin
----------------

Install the desktop application with Napari support. The Windows and macOS
installers include it. For a manual Python installation, use
``python -m pip install "openhcs[gui,napari]"``.

Launch ``openhcs``. You should see Plate Manager on the left and Pipeline
Editor on the right.

Generate the example plate
--------------------------

1. Choose **View > Generate Synthetic Plate**.
2. Leave the generated parameters at their defaults and leave **Output
   Directory** set to ``<temp directory>``.
3. Select **Generate Plate**.

After the dialog closes, a new plate appears in Plate Manager and Pipeline
Editor shows the supplied eight-step workflow. The first step is **Image
Enhancement Processing** and the final step is **Cell Counting**. These visible
changes confirm that both the data and its pipeline were created.

Initialise the source data
--------------------------

1. Select the generated plate row.
2. Select **Init** in Plate Manager.
3. Wait for initialisation to finish.

When initialisation succeeds, **Compile** becomes available. Open the metadata
viewer if you want to confirm the discovered wells, sites, channels, and Z
planes before continuing.

Inspect the supplied pipeline
-----------------------------

Read the Pipeline Editor from top to bottom. The workflow normalises the source
images, combines channels, projects Z planes, computes tile positions,
assembles the sites, and counts cells. Select any step to see its declared
function and configuration, but do not change values during this tutorial.

Compile the workflow
--------------------

1. Keep the generated plate selected.
2. Select **Compile**.
3. Wait for compilation to finish.

Compilation checks the source dimensions and the requirements of all eight
steps before processing begins. A successful compile enables **Run**. If an
error is reported, fix the first error before continuing rather than running a
partially valid workflow.

.. openhcs-gallery:: first-plate-workflow

Run and inspect the result
--------------------------

1. Select **Run**.
2. Watch the status and progress surfaces while the plate is processed.
3. Wait for the run to complete and for the detached Napari window to settle.
4. Inspect the streamed cell-count result in Napari.
5. Back in OpenHCS, select **Results** to inspect the available measurement
   snapshot.

You have now completed the same lifecycle used for real data: add a source,
initialise it, inspect its pipeline, compile, run, and review the result.

Continue with your own data
---------------------------

- Use :doc:`domain_expert_onboarding` to collect the source and biological
  facts needed for a real workflow.
- Use :doc:`image_sources` to map your files and dimensions.
- Use :doc:`../user_guide/real_time_visualization` to choose which scientific
  results should be streamed for review.
- Read :doc:`../concepts/data_dimensions` when adapting the example's site,
  channel, or Z behaviour.
