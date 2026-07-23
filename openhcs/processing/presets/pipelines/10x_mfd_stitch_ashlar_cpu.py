# Edit this pipeline and save to apply changes

from openhcs.processing.presets.mfd_specs import MfdPresetKey, build_mfd_preset

pipeline_steps = build_mfd_preset(MfdPresetKey.STITCH_ASHLAR_CPU)
