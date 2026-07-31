# Media gallery capture and delivery

`scripts/capture_media_gallery.py` captures real X11 application windows and
derives bounded website media from immutable source captures. It does not draw,
blur, replace, synthesize, or otherwise fabricate application content. Changes
to visible UI state must happen in the application before capture.

The workflow keeps three concerns separate:

1. OpenHCS and its integrations create the real session.
2. A lossless PNG or FFV1 Matroska file records that session.
3. One typed JSON manifest owns every crop, trim, output name, resolution bound,
   size bound, frame rate, and poster timestamp.

Captions, scenario descriptions, feature claims, and website layout do not
belong in the manifest.

## Host requirements

Run the capability report before a capture session:

```bash
python scripts/capture_media_gallery.py doctor
```

The report includes executable paths and versions so the exact toolchain can be
preserved with the capture report.

- Building and validating media requires FFmpeg and FFprobe.
- `capture-still` additionally requires ImageMagick.
- `record-window` additionally requires `xdotool` and an X11 display.

Wayland-only hosts can record a lossless PNG or Matroska source with their
compositor's capture tool, then use the same manifest-based plan, build, and
validate commands. Missing optional tools produce an actionable error rather
than an import failure.

## Capture immutable sources

List visible X11 windows and note the ID of the intended window:

```bash
wmctrl -lx
```

Capture a still:

```bash
python scripts/capture_media_gallery.py capture-still \
  --source-root /absolute/private/gallery-captures \
  --window-id 0x04600007 \
  --output raw/interface.png
```

Record a fixed window rectangle as lossless FFV1:

```bash
python scripts/capture_media_gallery.py record-window \
  --source-root /absolute/private/gallery-captures \
  --window-id 0x04600007 \
  --output raw/interaction.mkv \
  --duration-seconds 18 \
  --fps 30 \
  --display "${DISPLAY}"
```

Keep the selected window stationary while recording. The command resolves its
geometry once so that another window cannot enter the frame if the target is
moved. Both capture commands refuse to overwrite an existing source.

## Declare derivatives once

The manifest has no separate format field. The validated filename suffix selects
the authoritative WebP, MP4, WebM, or GIF encoding declaration, so a format
string cannot drift from its codec.

```json
{
  "schema_version": 1,
  "captures": [
    {
      "source": "raw/interface.png",
      "crop": {
        "x": 20,
        "y": 40,
        "width": 1800,
        "height": 1080
      },
      "outputs": [
        {
          "filename": "interface-overview.webp",
          "max_width": 1600,
          "max_height": 1000,
          "max_bytes": 1200000
        }
      ]
    },
    {
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
          "filename": "interaction-poster.webp",
          "max_width": 1600,
          "max_height": 1000,
          "max_bytes": 1200000,
          "frame_at_seconds": 1.5
        },
        {
          "filename": "interaction.webm",
          "max_width": 1600,
          "max_height": 1000,
          "max_bytes": 6000000,
          "fps": 24
        },
        {
          "filename": "interaction.mp4",
          "max_width": 1600,
          "max_height": 1000,
          "max_bytes": 8000000,
          "fps": 24
        },
        {
          "filename": "interaction.gif",
          "max_width": 960,
          "max_height": 600,
          "max_bytes": 12000000,
          "fps": 12
        }
      ]
    }
  ]
}
```

Motion trims are limited to 30 seconds. Derived filenames must be lowercase
basenames made from letters, numbers, and single hyphens. This keeps URLs stable
and prevents an output from escaping the destination root.

Use WebM as the primary website animation, MP4 as the broad compatibility
fallback, and the poster WebP for reduced-motion and unloaded states. GIF is
optional and intended for sharing where video is unavailable; it is usually
larger than either video encoding.

## Plan, build, and verify

Inspect exact commands before writing:

```bash
python scripts/capture_media_gallery.py plan \
  --manifest /absolute/private/gallery-captures/manifest.json \
  --source-root /absolute/private/gallery-captures \
  --output-root website/assets/gallery
```

`plan` validates paths and typed records but does not require source files, probe
media, create directories, or write outputs.

Build each derivative atomically and preserve the JSON report:

```bash
python scripts/capture_media_gallery.py build \
  --manifest /absolute/private/gallery-captures/manifest.json \
  --source-root /absolute/private/gallery-captures \
  --output-root website/assets/gallery \
  > /absolute/private/gallery-captures/build-report.json
```

The report records source and derivative SHA-256 hashes, dimensions, byte sizes,
codecs, and durations. Existing derivatives are rejected by default. After
reviewing the manifest and source, `--force` may atomically replace derivatives;
it never permits replacing a source capture.

Re-run verification independently before staging website assets:

```bash
python scripts/capture_media_gallery.py validate \
  --manifest /absolute/private/gallery-captures/manifest.json \
  --source-root /absolute/private/gallery-captures \
  --output-root website/assets/gallery \
  > /absolute/private/gallery-captures/validation-report.json
```

Validation fails if an output is missing, uses the wrong codec, exceeds its
dimensions or byte budget, or has a duration inconsistent with the declared
trim. Encoder versions can change exact compressed bytes, so reproducibility
means preserving the manifest, source hash, command plan, toolchain report, and
build report together.

## Privacy and scientific-integrity review

Complete this review before capture:

- Use redistributable example data or a session whose visible metadata is safe
  to publish.
- Close unrelated windows, terminals, notification surfaces, file managers, and
  private browser tabs.
- Use public-safe plate names and paths in the actual running application.
- Disable desktop notifications for the recording period.
- Avoid showing credentials, tokens, usernames, hostnames, patient or subject
  identifiers, unpublished measurements, and private local paths.

Complete this review after capture:

- Inspect every lossless source at full resolution before deriving outputs.
- Inspect each poster, the first and last frame, and the full animation.
- Confirm that cropping excludes only irrelevant window chrome or surrounding
  desktop space. Cropping is not a substitute for redaction.
- If private content appears inside the intended application frame, correct the
  real session and capture it again. Do not paint over the source.
- Confirm that the final caption describes only behavior visible in the media
  and verified in the running software.
- Keep lossless sources and reports in controlled storage. Publish only reviewed
  derivatives.

## Reproducible capture record

For each accepted media group, retain:

- the source PNG or FFV1 Matroska file;
- the JSON manifest;
- the `doctor` toolchain report;
- the dry-run command plan;
- the build and validation reports;
- dataset provenance and redistribution evidence;
- the OpenHCS commit and package version;
- the human privacy and claim-verification sign-off.

These records are capture provenance, not a second registry of OpenHCS
scenarios or website claims.
