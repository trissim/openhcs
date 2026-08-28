Capture and publish gallery media
=================================

This guide shows OpenHCS maintainers how to capture declared UI surfaces or
real X11 application windows and derive bounded website media from immutable
source captures with ``scripts.capture_media_gallery``. The workflow does not
draw, blur, replace, synthesise, or otherwise fabricate application content.
Make changes to visible UI state in the application before capture.

The workflow keeps four concerns separate:

1. OpenHCS and its integrations create the real session.
2. A lossless PNG or FFV1 Matroska file records that session.
3. Nominal gallery scenario declarations own stable IDs, capture targets, published filenames, copy, layout, and evidence.
4. One typed JSON manifest owns each capture-specific crop, trim, resolution bound, size bound, frame rate, and poster timestamp. Its output inventory must match the selected scenario declaration.

The website cards, RTD workflow figures, and public checksum record are generated
projections of the scenario declarations. Captions, scenario descriptions,
accessibility text, feature claims, and website layout therefore do not drift
into documentation pages or the private transformation manifest.

Host requirements
-----------------

Run the capability report before a capture session:

.. code:: bash

   python -m scripts.capture_media_gallery doctor

The report includes executable paths and versions so the exact toolchain can be preserved with the capture report.

- Building and validating media requires FFmpeg and FFprobe.
- ``capture-scenario-still`` additionally requires a live UI bridge descriptor and the local MCP dependencies.
- ``capture-still`` additionally requires ImageMagick.
- ``record-window`` additionally requires ``xdotool`` and an X11 display.

Wayland-only hosts can record a lossless PNG or Matroska source with their compositor's capture tool, then use the same manifest-based plan, build, and validate commands. Missing optional tools produce an actionable error rather than an import failure.

Capture immutable sources
-------------------------

For a scenario whose nominal target is exposed by the UI bridge, capture the declared target directly. The scenario owns the stable window identity; the command does not repeat a widget name or native window ID:

.. code:: bash

   python -m scripts.capture_media_gallery capture-scenario-still \
     multi-plate-overview \
     --source-root /absolute/private/gallery-captures \
     --output raw/multi-plate-overview.png \
     --descriptor-file-path /absolute/path/to/ui-bridge-descriptor.json

The command resolves its timeout through the same UI bridge timeout authority used by MCP, requests the declared widget snapshot, verifies the returned path, PNG suffix, SHA-256, and dimensions, and atomically moves the accepted source into the capture root. It refuses to overwrite an existing source.

For a human-reviewed native window that does not expose a trusted snapshot leaf, list visible X11 windows and note the ID of the intended window:

.. code:: bash

   wmctrl -lx

Capture a still:

.. code:: bash

   python -m scripts.capture_media_gallery capture-still \
     --source-root /absolute/private/gallery-captures \
     --window-id 0x04600007 \
     --output raw/interface.png

Record a fixed window rectangle as lossless FFV1:

.. code:: bash

   python -m scripts.capture_media_gallery record-window \
     --source-root /absolute/private/gallery-captures \
     --window-id 0x04600007 \
     --output raw/interaction.mkv \
     --duration-seconds 18 \
     --fps 30 \
     --no-mouse \
     --display "${DISPLAY}"

Keep the selected window stationary while recording. The command resolves its geometry once so that another window cannot enter the frame if the target is moved. Use ``--no-mouse`` when every visible interaction is driven through MCP; the capture report records that policy. Keep the default pointer capture for a separately labelled human UI demonstration. Both capture commands refuse to overwrite an existing source.

Declare derivatives once
------------------------

The manifest has no separate format field. The validated filename suffix selects the authoritative WebP, MP4, WebM, or GIF encoding declaration, so a format string cannot drift from its codec.

.. code:: json

   {
     "schema_version": 2,
     "captures": [
       {
         "scenario_id": "multi-plate-overview",
         "source": "raw/interface.png",
         "crop": {
           "x": 20,
           "y": 40,
           "width": 1800,
           "height": 1080
         },
         "outputs": [
           {
             "filename": "multi-plate-overview.webp",
             "max_width": 1600,
             "max_height": 1000,
             "max_bytes": 1200000
           }
         ]
       },
       {
         "scenario_id": "lazy-inheritance",
         "source": "raw/interaction.mkv",
         "crop": {
           "x": 20,
           "y": 40,
           "width": 1800,
           "height": 1080
         },
         "trim": {
           "start_seconds": 3.25,
           "duration_seconds": 10
         },
         "outputs": [
           {
             "filename": "lazy-inheritance-poster.webp",
             "max_width": 1600,
             "max_height": 1000,
             "max_bytes": 1200000,
             "frame_at_seconds": 1.5
           },
           {
             "filename": "lazy-inheritance.webm",
             "max_width": 1600,
             "max_height": 1000,
             "max_bytes": 6000000,
             "fps": 24
           },
           {
             "filename": "lazy-inheritance.mp4",
             "max_width": 1600,
             "max_height": 1000,
             "max_bytes": 8000000,
             "fps": 24
           }
         ]
       }
     ]
   }

Motion trims are limited to 30 seconds. Derived filenames must be lowercase basenames made from letters, numbers, and single hyphens. This keeps URLs stable and prevents an output from escaping the destination root.

Use WebM as the primary website animation, MP4 as the broad compatibility fallback, and the poster WebP for reduced-motion and unloaded states. GIF is optional and intended for sharing where video is unavailable; it is usually larger than either video encoding.

Plan, build, and verify
-----------------------

Inspect exact commands before writing:

.. code:: bash

   python -m scripts.capture_media_gallery plan \
     --manifest /absolute/private/gallery-captures/manifest.json \
     --source-root /absolute/private/gallery-captures \
     --output-root website/assets/gallery

``plan`` validates paths and typed records but does not require source files, probe media, create directories, or write outputs.

Build each derivative atomically and preserve the JSON report:

.. code:: bash

   python -m scripts.capture_media_gallery build \
     --manifest /absolute/private/gallery-captures/manifest.json \
     --source-root /absolute/private/gallery-captures \
     --output-root website/assets/gallery \
     > /absolute/private/gallery-captures/build-report.json

The report records source and derivative SHA-256 hashes, dimensions, byte sizes, codecs, and durations. Existing derivatives are rejected by default. After reviewing the manifest and source, ``--force`` may atomically replace derivatives; it never permits replacing a source capture.

Re-run verification independently before staging website assets:

.. code:: bash

   python -m scripts.capture_media_gallery validate \
     --manifest /absolute/private/gallery-captures/manifest.json \
     --source-root /absolute/private/gallery-captures \
     --output-root website/assets/gallery \
     > /absolute/private/gallery-captures/validation-report.json

Validation fails if an output is missing, uses the wrong codec, exceeds its dimensions or byte budget, or has a duration inconsistent with the declared trim. Encoder versions can change exact compressed bytes, so reproducibility means preserving the manifest, source hash, command plan, toolchain report, and build report together.

Privacy and scientific-integrity review
---------------------------------------

Complete this review before capture:

- Use redistributable example data or a session whose visible metadata is safe to publish.
- Close unrelated windows, terminals, notification surfaces, file managers, and private browser tabs.
- Use public-safe plate names and paths in the actual running application.
- Disable desktop notifications for the recording period.
- Avoid showing credentials, tokens, usernames, hostnames, patient or subject identifiers, unpublished measurements, and private local paths.

Complete this review after capture:

- Inspect every lossless source at full resolution before deriving outputs.
- Inspect each poster, the first and last frame, and the full animation.
- Confirm that cropping excludes only irrelevant window chrome or surrounding desktop space. Cropping is not a substitute for redaction.
- If private content appears inside the intended application frame, correct the real session and capture it again. Do not paint over the source.
- Confirm that the final caption describes only behaviour visible in the media and verified in the running software.
- Keep lossless sources and reports in controlled storage. Publish only reviewed derivatives.

Reproducible capture record
---------------------------

For each accepted media group, retain:

- the source PNG or FFV1 Matroska file;
- the JSON manifest;
- the ``doctor`` toolchain report;
- the dry-run command plan;
- the build and validation reports;
- dataset provenance and redistribution evidence;
- the OpenHCS commit and package version;
- the human privacy and claim-verification sign-off.

These records are capture provenance, not a second registry of OpenHCS scenarios or website claims. Inspect the declaration-owned scenario and target catalog with:

.. code:: bash

   python -m scripts.capture_media_gallery catalog

The checked-in release record is also the zero-dependency website projection.
Each scenario leaf supplies its card copy, accessibility text, representative
static image, layout, capture target, derivative contract, and release evidence.
The website builder reads the generated projection without importing OpenHCS,
while the Sphinx directive reads the same nominal scenario declarations. Neither
deployment surface acquires a second gallery inventory.

After a declaration or accepted asset changes, regenerate the checked-in projection and verify it exactly:

.. code:: bash

   python -m scripts.gallery_catalog
   python -m scripts.gallery_catalog --check
