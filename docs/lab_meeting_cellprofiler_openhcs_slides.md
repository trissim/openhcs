# OpenHCS CellProfiler Integration: Lab Meeting Draft

Source deck target: Wednesday lab meeting. Audience: wet-lab cell biology group.

Core claim for this version: OpenHCS preserves CellProfiler semantics with bitwise or tolerance-level parity and is at least 4x faster on single-core, single-sample execution across the 18 plotted official CellProfiler example pipelines. The expanded 25-pipeline set is green but figures are still being finalized; the next target set is 33.

## Slide 1: Title

**OpenHCS can run CellProfiler pipelines with identical results and much faster execution**

Subtitle: preserving trusted biology-facing image-analysis semantics while making high-content screening easier to scale.

Speaker note:
CellProfiler is already trusted in the field. The point is not to replace its scientific meaning. The point is to make the same pipeline semantics run inside OpenHCS, alongside the rest of the OpenHCS runtime, with better performance and composition.

## Slide 2: Why CellProfiler Matters

**CellProfiler is one of the canonical tools for quantitative biological image analysis**

- Free, open-source image-analysis software built for high-throughput microscopy.
- Designed for biologists to build modular pipelines without writing code.
- Used for cell counting, segmentation, morphology, intensity, colocalization, tracking, object relationships, worms, yeast colonies, illumination correction, and high-content screening.
- NIGMS describes CellProfiler as cited in more than 15,000 scientific papers.
- The original 2006 CellProfiler paper itself currently has about 4,968 citations by publisher/Dimensions metrics.

Speaker note:
The important setup is that CellProfiler is not a toy baseline. It is a widely cited, field-defining software system for image-based phenotyping.

Source notes:
- NIGMS Biomedical Beat states CellProfiler has been cited in more than 15,000 scientific papers.
- Genome Biology article metrics list 4,968 citations and 221k accesses for the 2006 paper.
- The original paper describes CellProfiler as free, open source, and designed for flexible high-throughput cell image analysis.

## Slide 2b: CellProfiler Has Been Funded As Infrastructure

**CellProfiler is not just a paper; it has been sustained as community infrastructure**

- Early CellProfiler work was supported by fellowships, academic grants, NIH, DoD, and foundation funding.
- Broad reports that a 2008 NIH grant enabled long-term software engineering support for CellProfiler.
- CellProfiler 3.0/3D work was supported by NIH and the Allen Institute for Cell Science.
- Current Carpenter/Broad imaging work is supported by NIGMS grants including R35GM122547 and P41GM135019.

Speaker note:
I would present funding qualitatively unless we pull exact NIH RePORTER totals. The defensible statement is that CellProfiler has had sustained NIH and institutional/foundation support as infrastructure, not that we know a precise lifetime dollar total.

## Slide 3: What CellProfiler Is Used For

**CellProfiler turns microscopy images into measurements**

- Identify cells, nuclei, organelles, worms, colonies, and other objects.
- Measure count, area, shape, intensity, texture, granularity, and spatial relationships.
- Build reproducible pipelines for high-content screens and routine image quantification.
- Export tables and images that downstream analysis can use.

Speaker note:
This is the wet-lab bridge: CellProfiler takes images and returns measurements that become figures, QC, and biological conclusions.

## Slide 4: Why This Is A Serious Benchmark

**Matching CellProfiler means matching a mature scientific pipeline language**

- A `.cppipe` file is not just a script; it is a stored scientific analysis protocol.
- Correctness requires preserving file semantics, image/object names, per-well grouping, object measurements, relationships, and numerical behavior.
- A useful conversion cannot just be “similar”; it must preserve the outputs scientists already trust.

Speaker note:
This frames parity as the hard part. Speed without semantic preservation is not useful for biology. The claim is that OpenHCS preserves the behavior and then accelerates it.

## Slide 5: OpenHCS Integration

**OpenHCS treats CellProfiler as a pipeline semantics source**

- Loads CellProfiler `.cppipe` pipelines.
- Converts CellProfiler modules into OpenHCS steps.
- Preserves CellProfiler image/object/measurement semantics through typed runtime adapters.
- Executes through OpenHCS runtime infrastructure instead of the native CellProfiler GUI/runtime path.
- Produces comparable measurement outputs for parity checking.

Speaker note:
The key phrase is “semantics source.” OpenHCS is not merely shelling out to CellProfiler. It is absorbing and executing the pipeline meaning inside the OpenHCS runtime.

## Slide 6: OpenHCS Is Broader Than CellProfiler

**CellProfiler joins the OpenHCS ecosystem**

- OpenHCS already integrates multiple scientific-imaging surfaces.
- Fiji/ImageJ-compatible ROI outputs and Fiji/ImageJ streaming are supported.
- napari streaming is supported for live visualization.
- PolyStore treats storage and streaming targets as backends, so viewers are part of the same I/O model.
- CellProfiler conversion therefore becomes one engine inside a broader canonical pipeline system.

Speaker note:
This is the architectural pitch. OpenHCS is not “a faster CellProfiler clone.” It is a general pipeline/runtime substrate that can host CellProfiler semantics alongside Fiji, napari, and other backends.

## Slide 7: Benchmark Design

**Benchmark question: can OpenHCS preserve CellProfiler outputs and run faster?**

- Native CellProfiler run: reference output.
- OpenHCS run: converted `.cppipe` executed through OpenHCS.
- Parity metric: compare outputs up to defined numerical tolerance.
- Runtime metric: single-core, single-sample execution time.
- Dataset for plotted results: 18 official CellProfiler example pipelines.

Speaker note:
Keep this simple. The audience should understand that each pair is the same pipeline, once in native CellProfiler and once in OpenHCS.

## Slide 8: Accuracy Result

**Result: semantic parity across tested pipelines**

![Accuracy](/tmp/openhcs_labmeeting_prelim/cppipe_figures_20260505_v7_compact/cppipe_accuracy.png)

Speaker note:
This is the most important slide scientifically. The point is not just that the chart is high. The point is that OpenHCS can reproduce the same pipeline outputs, not merely run something vaguely similar.

## Slide 9: Accuracy Zoom

**Zoomed parity view**

![Accuracy zoom](/tmp/openhcs_labmeeting_prelim/cppipe_figures_20260505_v7_compact/cppipe_accuracy_zoom.png)

Speaker note:
Use this if someone asks whether “100%” is hiding meaningful differences. This slide shows the zoomed view around perfect parity.

## Slide 10: Raw Runtime

**OpenHCS runs the same pipelines in fewer seconds**

![Raw seconds](/tmp/openhcs_labmeeting_prelim/cppipe_figures_20260505_v7_compact/cppipe_raw_seconds.png)

Speaker note:
This slide is absolute runtime. It answers “how many seconds do I wait?” rather than only showing relative speedup.

## Slide 11: Runtime On Log Scale

**Runtime differences remain visible across short and long pipelines**

![Raw seconds log](/tmp/openhcs_labmeeting_prelim/cppipe_figures_20260505_v7_compact/cppipe_raw_seconds_log.png)

Speaker note:
Use this for readability when some pipelines are much slower than others. It prevents the long-running pipelines from visually flattening everything else.

## Slide 12: Speedup

**Minimum 4x speedup across the plotted 18 pipelines**

![Speedup](/tmp/openhcs_labmeeting_prelim/cppipe_figures_20260505_v7_compact/cppipe_speedup.png)

Speaker note:
This is the headline performance result. It is single-core, single-sample, so it is not relying on parallelism to inflate the comparison.

## Slide 13: Speedup On Log Scale

**Some pipelines are dramatically faster**

![Speedup log](/tmp/openhcs_labmeeting_prelim/cppipe_figures_20260505_v7_compact/cppipe_speedup_log.png)

Speaker note:
This slide helps when speedups have a wide range. Keep the verbal message simple: every plotted pipeline clears the threshold, and some are much higher.

## Slide 14: Why Single-Core Matters

**This is the conservative benchmark**

- These results are single-core and single-sample.
- They do not depend on well-level multiprocessing.
- In real high-content screening, multiple wells/samples can be processed in parallel.
- The single-core result establishes that the execution engine itself is faster before scaling out.

Speaker note:
This prevents confusion: OpenHCS can also multiprocess, but the 4x claim here is not a parallelism artifact.

## Slide 15: Current Status

**Where the benchmark stands**

- 18 official CellProfiler example pipelines: plotted, parity green, minimum 4x speedup.
- 25 pipelines: parity green and speed target green, graph packaging still catching up.
- 33 pipelines: in progress, intended as broader evidence across more public datasets/pipelines.
- Current focus: keep expanding coverage without weakening semantics.

Speaker note:
Be transparent. The polished figure set is 18. The expanded set is promising but not the figure claim yet.

## Slide 16: Take-Home Message

**OpenHCS can preserve trusted image-analysis semantics while making them faster and more composable**

- CellProfiler is a major scientific imaging tool.
- OpenHCS can execute CellProfiler pipeline meaning with matching outputs.
- The plotted benchmark shows at least 4x single-core speedup on all 18 tested official examples.
- OpenHCS also connects this to Fiji/ImageJ, napari, and a broader pipeline/runtime architecture.

Speaker note:
End with the scientific value: same trusted measurements, faster turnaround, better integration, and a path toward larger high-content screens.

## Backup Slide: Exact Figure Files

- `/tmp/openhcs_labmeeting_prelim/cppipe_figures_20260505_v7_compact/cppipe_accuracy.png`
- `/tmp/openhcs_labmeeting_prelim/cppipe_figures_20260505_v7_compact/cppipe_accuracy_zoom.png`
- `/tmp/openhcs_labmeeting_prelim/cppipe_figures_20260505_v7_compact/cppipe_raw_seconds.png`
- `/tmp/openhcs_labmeeting_prelim/cppipe_figures_20260505_v7_compact/cppipe_raw_seconds_log.png`
- `/tmp/openhcs_labmeeting_prelim/cppipe_figures_20260505_v7_compact/cppipe_speedup.png`
- `/tmp/openhcs_labmeeting_prelim/cppipe_figures_20260505_v7_compact/cppipe_speedup_log.png`

## Backup Slide: Source Links

- CellProfiler 2006 paper and citation/access metrics: https://genomebiology.biomedcentral.com/articles/10.1186/gb-2006-7-10-r100/metrics
- CellProfiler 2006 paper full text: https://genomebiology.biomedcentral.com/articles/10.1186/gb-2006-7-10-r100
- CellProfiler official introduction: https://cellprofiler.org/introduction
- CellProfiler official about page: https://cellprofiler.org/about/
- Broad CellProfiler software page: https://www.broadinstitute.org/scientific-community/software/cellprofiler
- NIGMS Biomedical Beat CellProfiler profile: https://biobeat.nigms.nih.gov/2023/03/automating-cellular-image-analysis-to-find-potential-medicines/
- Broad CellProfiler 3D article with NIH/Allen Institute support context: https://broadinstitute.org/blog/cellprofiler-goes-3d
- Broad CellProfiler history/funding context: https://www.broadinstitute.org/node/5140
- OpenHCS README for Fiji/napari/PolyStore positioning: `README.md`
