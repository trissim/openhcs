=========================
Integrating EZStitcher with Other Tools
=========================

This page demonstrates **two realistic integration patterns** that crop up in fluorescence-microscopy workflows:

1. **Illumination correction with *BaSiCPy*** ➜ **self-supervised denoising with *N2V2* (Careamics)**
2. **Template-matching cropper** - automatically grab a region of interest from the stitched mosaic.

If you have not read :doc:`advanced_usage`, start there first - it explains custom functions and multithreading, which we reuse below.

All functions passed to :class:`~ezstitcher.core.steps.Step` **must** accept a *list of images* and return a *list of images* of equal length. Saving to disk should be handled by the pipeline or a separate post-hoc script.

--------------------------------------------------------------------
1. BaSiCPy + N2V2 denoising
--------------------------------------------------------------------

`BaSiCPy <https://github.com/peng-lab/BaSiCPy>`_ removes slowly varying illumination fields.
`Careamics-N2V2 <https://careamics.github.io>`_ performs noise-to-void denoising without clean targets.

.. code-block:: python

   from pathlib import Path
   import numpy as np

   import basicpy                         # pip install basicpy (Peng-Lab fork)
   from careamics.models import N2V2      # pip install careamics==0.1

   from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator
   from ezstitcher.factories import AutoPipelineFactory
   from ezstitcher.core.pipeline import Pipeline
   from ezstitcher.core.steps    import Step

   # -------------- orchestrator + stitching -----------------------
   plate_path   = Path("~/data/PlateA").expanduser()
   orchestrator = PipelineOrchestrator(plate_path)

   base_factory = AutoPipelineFactory(
       input_dir=orchestrator.workspace_path,
       normalize=True,
   )
   pipelines = base_factory.create_pipelines()  # [0] = position, [1] = assembly

   # -------------- helper functions -------------------------------
   def flatfield_basicpy(images):
       """Return BaSiCPy-corrected stack."""
       stack = np.dstack(images)  # z-axis last for BaSiCPy
       shading, background = basicpy.BaSiC(threshold=0.01).fit(stack)
       corrected = (stack - background) / shading
       return [corrected[..., i] for i in range(corrected.shape[-1])]

   n2v2 = N2V2.from_pretrained("n2v2_fluo")

   def denoise_n2v2(images):
       return [n2v2.predict(im, axes="YX") for im in images]

   post_pipe = Pipeline(
       input_dir=pipelines[1].output_dir,          # stitched TIFFs
       output_dir=Path("out/illcorr_n2v2"),
       steps=[
           Step(name="BaSiC flat-field", func=flatfield_basicpy),
           Step(name="N2V2 denoise",    func=denoise_n2v2),
       ],
       name="BaSiC + N2V2",
   )

   pipelines.append(post_pipe)
   orchestrator.run(pipelines=pipelines)

--------------------------------------------------------------------
2. ROI extraction via template matching (Multi-Template-Matching)
--------------------------------------------------------------------

For a lighter dependency we use the
`Multi-Template-Matching <https://github.com/multi-template-matching/MultiTemplateMatching-Python>`_ wrapper around OpenCV.
It can handle multiple templates and returns bounding boxes directly.

Install once with ::

   pip install multitpletematch  # actual PyPI name: multitpletematch

.. code-block:: python

   from pathlib import Path
   import numpy as np
   import cv2
   import multitpletematch as mtm  # MultiTemplateMatching

   templates = [cv2.imread(str(p), cv2.IMREAD_ANYDEPTH)
                for p in Path("templates").glob("*.tif")]

   matcher = mtm.MultiTemplateMatching(method=cv2.TM_CCOEFF_NORMED,
                                       maxOverlap=0.1,
                                       scoreThreshold=0.6)

   def crop_by_template(images, pad=20):
       """Crop around first high-score template match for each image."""
       outs = []
       for im in images:
           bboxes, _ = matcher.matchTemplates(templates, im, N_object=1)
           if not bboxes:
               outs.append(im)  # fallback: no crop
               continue
           x, y, w, h = bboxes[0]['bbox']  # mtm gives (x, y, w, h)
           y1 = max(y - pad, 0)
           x1 = max(x - pad, 0)
           y2 = min(y + h + pad, im.shape[0])
           x2 = min(x + w + pad, im.shape[1])
           outs.append(im[y1:y2, x1:x2])
       return outs

   crop_pipe = Pipeline(
       input_dir=pipelines[1].output_dir,
       output_dir=Path("out/roi_crop"),
       steps=[Step(name="Template crop", func=crop_by_template)],
       name="ROI Cropper",
   )

   orchestrator.run(pipelines=[crop_pipe])

--------------------------------------------------------------------
Navigation
--------------------------------------------------------------------

* Back to :doc:`advanced_usage` for custom factories and multithreading.
* Forward to :doc:`../development/extending` to add new microscope handlers.
