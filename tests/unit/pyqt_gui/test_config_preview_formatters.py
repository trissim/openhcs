"""Declaration-driven OpenHCS config preview projections."""

from pyqt_reactive.utils.preview_formatters import resolve_preview_label

from openhcs.core.config import (
    LazyNapariStreamingConfig,
    LazyWellFilterConfig,
    NapariStreamingConfig,
    StepMaterializationConfig,
    WellFilterConfig,
    WellFilterMode,
)
from openhcs.pyqt_gui.widgets.config_preview_formatters import (
    format_well_filter_config,
)


def test_well_filter_preview_label_is_owned_by_config_declaration() -> None:
    resolution = resolve_preview_label(WellFilterConfig())

    assert resolution is not None
    assert resolution.owner is WellFilterConfig
    assert resolution.label == "FILT"


def test_plain_well_filter_preview_requires_a_selection() -> None:
    assert format_well_filter_config(WellFilterConfig(), "well_filter") is None
    assert (
        format_well_filter_config(
            WellFilterConfig(
                well_filter=["A01", "A02"],
                well_filter_mode=WellFilterMode.EXCLUDE,
            ),
            "well_filter",
        )
        == "FILT-2"
    )


def test_lazy_plain_well_filter_preserves_base_preview_policy() -> None:
    resolution = resolve_preview_label(LazyWellFilterConfig())

    assert resolution is not None
    assert resolution.owner is WellFilterConfig
    assert format_well_filter_config(LazyWellFilterConfig(), "well_filter") is None


def test_lazy_specialized_config_uses_its_declared_preview_policy() -> None:
    config = LazyNapariStreamingConfig(enabled=True)
    resolution = resolve_preview_label(config)

    assert resolution is not None
    assert resolution.owner is NapariStreamingConfig
    assert format_well_filter_config(config, "napari_streaming_config") == "NAP"


def test_specialized_preview_labels_are_derived_from_subclass_declarations() -> None:
    assert (
        format_well_filter_config(
            NapariStreamingConfig(enabled=True),
            "napari_streaming_config",
        )
        == "NAP"
    )
    assert (
        format_well_filter_config(
            StepMaterializationConfig(enabled=True),
            "step_materialization_config",
        )
        == "MAT"
    )
