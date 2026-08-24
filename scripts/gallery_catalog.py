"""Nominal source declarations for the public OpenHCS application gallery."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import ClassVar

from openhcs.agent.ui_bridge_identities import (
    MainWindowWidgetIdentity,
    PipelineEditorWidgetIdentity,
)
from openhcs.serialization.json import JsonValue, to_jsonable

GALLERY_CARDS_TOKEN = "{{ OPENHCS_GALLERY_CARDS }}"
GALLERY_PROVENANCE_TOKEN = "{{ OPENHCS_GALLERY_PROVENANCE }}"
RELEASE_MEDIA_SCHEMA_VERSION = "openhcs.release-media.v3"
RELEASE_MEDIA_RECORD_NAME = "release-media-record.json"
SCENARIO_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class GalleryCatalogError(RuntimeError):
    """Raised when a gallery declaration or generated projection is invalid."""


@dataclass(frozen=True, slots=True)
class GallerySourceCaptureRequest:
    """Destination and live-connection fields for one immutable source capture."""

    source_root: Path
    output: Path
    descriptor_file_path: Path | None = None
    timeout_ms: int | None = None

    def __post_init__(self) -> None:
        if self.timeout_ms is not None and self.timeout_ms <= 0:
            raise GalleryCatalogError("Gallery capture timeout_ms must be positive.")


@dataclass(frozen=True, slots=True, kw_only=True)
class GallerySourceEvidence:
    """Immutable facts retained from one accepted lossless source capture."""

    sha256: str
    width: int
    height: int
    duration_seconds: float | None = None
    format: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class GallerySourceCaptureResult(GallerySourceEvidence):
    """Accepted source evidence plus its capture-session path."""

    path: str


class GalleryCaptureTargetABC(ABC):
    """Nominal owner of the live surface required by one gallery scenario."""

    human_review_required: ClassVar[bool] = False

    @property
    def target_kind(self) -> str:
        """Derive the serialized target kind from its nominal declaration."""

        declaration_name = type(self).__name__.removesuffix("CaptureTarget")
        return re.sub(r"(?<!^)(?=[A-Z])", "_", declaration_name).lower()

    def release_record(self) -> GalleryCaptureTargetReleaseRecord:
        """Project nominal identity and target fields for publication."""

        return GalleryCaptureTargetReleaseRecord(
            kind=self.target_kind,
            human_review_required=self.human_review_required,
            parameters=to_jsonable(self),
        )


class HumanReviewedCaptureTargetABC(GalleryCaptureTargetABC, ABC):
    """Target branch whose published capture requires human review."""

    human_review_required = True


@dataclass(frozen=True, slots=True)
class UiBridgeWindowCaptureTarget(GalleryCaptureTargetABC):
    """One stable window exposed by the OpenHCS UI bridge."""

    window_id: str


@dataclass(frozen=True, slots=True)
class ApplicationSceneCaptureTarget(HumanReviewedCaptureTargetABC):
    """An application-owned composition of several declared UI windows."""

    window_roles: tuple[str, ...]


class ViewerWindowCaptureTargetABC(GalleryCaptureTargetABC, ABC):
    """Nominal branch for viewer-native windows."""


class HumanReviewedViewerWindowCaptureTargetABC(
    HumanReviewedCaptureTargetABC,
    ViewerWindowCaptureTargetABC,
    ABC,
):
    """A viewer surface that still requires a native human capture check."""

    @property
    def review_reason(self) -> str:
        return (
            f"{self.target_kind.replace('_', ' ')} does not yet expose a native "
            "trusted screenshot leaf."
        )


@dataclass(frozen=True, slots=True)
class FijiViewerWindowCaptureTarget(HumanReviewedViewerWindowCaptureTargetABC):
    """The Fiji window used for native image and ROI review."""


@dataclass(frozen=True, slots=True)
class NapariViewerWindowCaptureTarget(ViewerWindowCaptureTargetABC):
    """The Napari window exposed through its viewer control endpoint."""


class GalleryScientificEvidenceABC(ABC):
    """Nominal scientific-evidence record attached to a scenario leaf."""


@dataclass(frozen=True, slots=True)
class FijiRoiScientificEvidence(GalleryScientificEvidenceABC):
    """Source-plane and ROI archive facts for the accepted Fiji capture."""

    source_plane_sha256: str
    roi_archive_sha256: str
    source_plane_count: int
    roi_count: int


@dataclass(frozen=True, slots=True)
class GalleryPublishedAssetRecord:
    """One published derivative and its content identity."""

    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class GalleryCaptureTargetReleaseRecord:
    """Published identity and parameters for one nominal capture target."""

    kind: str
    human_review_required: bool
    parameters: JsonValue


@dataclass(frozen=True, slots=True)
class GalleryScenarioReleaseRecord:
    """Release evidence projected from one scenario declaration."""

    id: str
    proof: str
    capture_target: GalleryCaptureTargetReleaseRecord
    source: GallerySourceEvidence | None
    scientific_evidence: GalleryScientificEvidenceABC | None
    published: tuple[GalleryPublishedAssetRecord, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class GalleryScenarioABC(ABC):
    """Declaration owner for one public gallery scenario and all its views."""

    scenario_id: str
    label: str
    heading: str
    description: str
    proof: str
    layout_class: str
    alt_text: str
    width: int
    height: int
    capture_target: GalleryCaptureTargetABC
    source_evidence: GallerySourceEvidence | None = None
    scientific_evidence: GalleryScientificEvidenceABC | None = None
    media_card_class: ClassVar[str | None] = None

    @abstractmethod
    def published_paths(self) -> tuple[str, ...]:
        """Return the complete public asset inventory for this scenario."""

    @abstractmethod
    def render_media(self) -> str:
        """Render this scenario's media element without caller-side dispatch."""

    @abstractmethod
    def capture_source(
        self,
        request: GallerySourceCaptureRequest,
    ) -> GallerySourceCaptureResult:
        """Capture a source through this scenario's media contract."""

    def caption_id_attribute(self) -> str:
        """Return an optional caption identity required by the media element."""

        return ""

    def card_class(self) -> str:
        return " ".join(
            class_name
            for class_name in (
                "gallery-card",
                self.media_card_class,
                self.layout_class,
            )
            if class_name is not None
        )

    def render_card(self) -> str:
        """Render the website card from this scenario declaration."""

        return "\n".join(
            (
                f'          <figure class="{escape(self.card_class(), quote=True)}">',
                self.render_media(),
                f"            <figcaption{self.caption_id_attribute()}>",
                f'              <span class="gallery-label">{escape(self.label)}</span>',
                f"              <h3>{escape(self.heading)}</h3>",
                f"              <p>{escape(self.description)}</p>",
                "            </figcaption>",
                "          </figure>",
            )
        )

    def release_record(self, asset_root: Path) -> GalleryScenarioReleaseRecord:
        """Project source, target, proof, and current published checksums."""

        return GalleryScenarioReleaseRecord(
            id=self.scenario_id,
            proof=self.proof,
            capture_target=self.capture_target.release_record(),
            source=self.source_evidence,
            scientific_evidence=self.scientific_evidence,
            published=tuple(
                GalleryPublishedAssetRecord(
                    path=path,
                    sha256=_sha256_file(asset_root / path),
                )
                for path in self.published_paths()
            ),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class StillGalleryScenarioABC(GalleryScenarioABC, ABC):
    """Gallery scenario represented by one full-resolution still image."""

    open_aria_label: str

    def published_paths(self) -> tuple[str, ...]:
        return (f"{self.scenario_id}.webp",)

    def render_media(self) -> str:
        asset_filename = self.published_paths()[0]
        return "\n".join(
            (
                "            <a",
                '              class="gallery-media-link"',
                f'              href="assets/gallery/{escape(asset_filename, quote=True)}"',
                f'              aria-label="{escape(self.open_aria_label, quote=True)}"',
                "            >",
                "              <img",
                f'                src="assets/gallery/{escape(asset_filename, quote=True)}"',
                f'                width="{self.width}"',
                f'                height="{self.height}"',
                '                loading="lazy"',
                '                decoding="async"',
                f'                alt="{escape(self.alt_text, quote=True)}"',
                "              >",
                "            </a>",
            )
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class UiBridgeStillGalleryScenario(StillGalleryScenarioABC):
    """Still scenario captured through one stable UI-bridge window."""

    capture_target: UiBridgeWindowCaptureTarget

    def capture_source(
        self,
        request: GallerySourceCaptureRequest,
    ) -> GallerySourceCaptureResult:
        from scripts.capture_media_gallery import capture_ui_bridge_window_source

        return capture_ui_bridge_window_source(self.capture_target, request)


@dataclass(frozen=True, slots=True, kw_only=True)
class HumanReviewedStillGalleryScenario(StillGalleryScenarioABC):
    """Still scenario whose external viewer capture requires human review."""

    capture_target: HumanReviewedViewerWindowCaptureTargetABC

    def capture_source(
        self,
        request: GallerySourceCaptureRequest,
    ) -> GallerySourceCaptureResult:
        del request
        raise GalleryCatalogError(self.capture_target.review_reason)


@dataclass(frozen=True, slots=True, kw_only=True)
class MotionGalleryScenario(GalleryScenarioABC):
    """Gallery scenario represented by WebM/MP4 video and a WebP poster."""

    download_label: str
    open_aria_label: str
    media_card_class = "gallery-card-motion"

    def capture_source(
        self,
        request: GallerySourceCaptureRequest,
    ) -> GallerySourceCaptureResult:
        del request
        raise GalleryCatalogError(
            f"Gallery scenario {self.scenario_id!r} requires a bounded recording "
            "workflow, not a still-source capture."
        )

    def published_paths(self) -> tuple[str, ...]:
        return (
            f"{self.scenario_id}-poster.webp",
            f"{self.scenario_id}.webm",
            f"{self.scenario_id}.mp4",
        )

    def caption_id_attribute(self) -> str:
        return f' id="{escape(self.scenario_id, quote=True)}-caption"'

    def render_media(self) -> str:
        stem = escape(self.scenario_id, quote=True)
        caption_id = escape(f"{self.scenario_id}-caption", quote=True)
        return "\n".join(
            (
                '            <div class="gallery-motion">',
                "              <video",
                "                controls",
                "                muted",
                "                loop",
                "                playsinline",
                '                preload="metadata"',
                f'                poster="assets/gallery/{stem}-poster.webp"',
                f'                aria-describedby="{caption_id}"',
                "              >",
                f'                <source src="assets/gallery/{stem}.webm" type="video/webm">',
                f'                <source src="assets/gallery/{stem}.mp4" type="video/mp4">',
                f'                <a href="assets/gallery/{stem}.mp4">{escape(self.download_label)}</a>',
                "              </video>",
                "              <a",
                '                class="gallery-motion-fallback gallery-media-link"',
                f'                href="assets/gallery/{stem}-poster.webp"',
                f'                aria-label="{escape(self.open_aria_label, quote=True)}"',
                "              >",
                "                <img",
                f'                  src="assets/gallery/{stem}-poster.webp"',
                f'                  width="{self.width}"',
                f'                  height="{self.height}"',
                '                  loading="lazy"',
                '                  decoding="async"',
                f'                  alt="{escape(self.alt_text, quote=True)}"',
                "                >",
                "                <span>Static view shown because reduced motion is enabled</span>",
                "              </a>",
                "            </div>",
            )
        )


class GalleryScenarioCatalog:
    """Compose one scenario declaration from each nominal MRO branch."""

    @classmethod
    def scenarios(cls) -> tuple[GalleryScenarioABC, ...]:
        """Return scenario declarations directly from the catalog MRO."""

        return tuple(
            declaration
            for owner_type in cls.__mro__
            if isinstance(
                declaration := owner_type.__dict__.get("scenario"),
                GalleryScenarioABC,
            )
        )

    @classmethod
    def for_id(cls, scenario_id: str) -> GalleryScenarioABC:
        """Resolve one scenario from the MRO-composed declaration set."""

        matches = tuple(
            scenario
            for scenario in cls.scenarios()
            if scenario.scenario_id == scenario_id
        )
        if len(matches) != 1:
            raise GalleryCatalogError(
                f"Expected one gallery scenario for {scenario_id!r}; found "
                f"{len(matches)}."
            )
        return matches[0]


class MultiPlateOverviewGalleryScenarioCatalog(GalleryScenarioCatalog):
    """Add the multi-plate workspace scenario to the gallery MRO."""

    scenario = UiBridgeStillGalleryScenario(
        scenario_id="multi-plate-overview",
        label="Workspace",
        heading="Seven assay plates, one workspace",
        description=(
            "Five deterministic examples and two imported CellProfiler workflows "
            "retain independent state in one session."
        ),
        proof=(
            "Seven independently configured assay plates are visible in one workspace."
        ),
        layout_class="gallery-card-wide",
        width=1600,
        height=958,
        alt_text=(
            "OpenHCS Plate Manager with seven assay plates loaded, including "
            "CellProfiler Comet Assay and Wound Healing examples"
        ),
        open_aria_label=(
            "Open the multi-plate workspace screenshot at full resolution"
        ),
        capture_target=UiBridgeWindowCaptureTarget(
            window_id=MainWindowWidgetIdentity.require_value()
        ),
    )


class PipelineEditorGalleryScenarioCatalog(GalleryScenarioCatalog):
    """Add the pipeline-editor scenario to the gallery MRO."""

    scenario = UiBridgeStillGalleryScenario(
        scenario_id="pipeline-editor",
        label="Authoring",
        heading="CellProfiler steps in the same pipeline model",
        description=(
            "The 12-step Comet Assay uses OpenHCS compilation, configuration, "
            "viewers, and results."
        ),
        proof=(
            "The imported CellProfiler Comet Assay is visible as twelve pipeline "
            "steps."
        ),
        layout_class="gallery-card-compact",
        width=942,
        height=900,
        alt_text=(
            "OpenHCS Pipeline Editor showing the 12 imported steps of the "
            "CellProfiler Comet Assay"
        ),
        open_aria_label="Open the pipeline editor screenshot at full resolution",
        capture_target=UiBridgeWindowCaptureTarget(
            window_id=PipelineEditorWidgetIdentity.require_value()
        ),
    )


class LazyInheritanceGalleryScenarioCatalog(GalleryScenarioCatalog):
    """Add the lazy-inheritance scenario to the gallery MRO."""

    scenario = MotionGalleryScenario(
        scenario_id="lazy-inheritance",
        label="Configuration",
        heading="See configuration provenance in both directions",
        description=(
            "Well Filter changes to Image30 in PipelineConfig, then the inherited "
            "step field flashes as the update arrives. Clicking that step label "
            "flashes the owning PipelineConfig field, making the provenance link "
            "explicit in both directions."
        ),
        proof=(
            "Editing the pipeline-owned Well Filter flashes its inherited step field; "
            "clicking the inherited step label then flashes the owning pipeline field."
        ),
        layout_class="gallery-card-wide",
        width=1600,
        height=1000,
        alt_text=(
            "OpenHCS workspace beside pipeline and step settings showing Well Filter "
            "Image30 at pipeline scope and inherited by the step"
        ),
        download_label="Download the lazy-inheritance demonstration.",
        open_aria_label=(
            "Open the lazy-inheritance demonstration poster at full resolution"
        ),
        capture_target=ApplicationSceneCaptureTarget(
            window_roles=("main_window", "pipeline_config", "step_editor"),
        ),
        source_evidence=GallerySourceEvidence(
            sha256=("4057415aed2d48a6a923a063a5fcc5114f89b1fa9886ff491f5e32bd3503d0c5"),
            width=1600,
            height=1000,
            duration_seconds=18.0,
        ),
    )


class FijiReviewGalleryScenarioCatalog(GalleryScenarioCatalog):
    """Add the Fiji-review scenario to the gallery MRO."""

    scenario = HumanReviewedStillGalleryScenario(
        scenario_id="fiji-review",
        label="Fiji",
        heading="Review matching ROIs in Fiji",
        description=(
            "OpenHCS streams one bound Field 1 nuclear plane and its nine "
            "segmentation ROIs together. The native outlines follow the displayed "
            "nuclei."
        ),
        proof=(
            "One bound Field 1 nuclear plane is shown with the corresponding nine "
            "native ROI Manager entries; the visible outlines follow the displayed "
            "nuclei without cross-plane ROI mixing."
        ),
        layout_class="gallery-card-compact",
        width=1310,
        height=930,
        alt_text=(
            "Fiji ImageJ showing one NeuronCyto II Field 1 nuclear plane with its "
            "nine native segmentation ROI entries"
        ),
        open_aria_label="Open the Fiji integration screenshot at full resolution",
        capture_target=FijiViewerWindowCaptureTarget(),
        source_evidence=GallerySourceEvidence(
            sha256=("1fc686a80f78189d0abc6efbc1b9d10865a17812f94b0a63274af1551a6a44bd"),
            width=1310,
            height=930,
            format="PNG",
        ),
        scientific_evidence=FijiRoiScientificEvidence(
            source_plane_sha256=(
                "ddd9f8a9edd0837275d6967fd746bdd424bb7a642139073e443a07eca0271847"
            ),
            roi_archive_sha256=(
                "179772aaed8d21bdc260202ab48925538f125a2ee17b6e2ca4eecaa42fbeff62"
            ),
            source_plane_count=1,
            roi_count=9,
        ),
    )


class ZmqStartupCompileGalleryScenarioCatalog(GalleryScenarioCatalog):
    """Add the endpoint-startup scenario to the gallery MRO."""

    scenario = MotionGalleryScenario(
        scenario_id="zmq-startup-compile",
        label="Local MCP",
        heading="See endpoint startup while the request is running",
        description=(
            "A pointer-free MCP compile request immediately adds the execution "
            "endpoint, streams catalog-preparation phases into the browser and "
            "status bar, and settles connected."
        ),
        proof=(
            "An execution endpoint appears during the compile request and reports "
            "startup phases before settling connected."
        ),
        layout_class="gallery-card-feature",
        width=1600,
        height=1000,
        alt_text=(
            "OpenHCS showing an execution endpoint preparing its function catalog "
            "while a pipeline compiles"
        ),
        download_label="Download the endpoint-startup recording.",
        open_aria_label=(
            "Open the execution-endpoint startup poster at full resolution"
        ),
        capture_target=UiBridgeWindowCaptureTarget(
            window_id=MainWindowWidgetIdentity.require_value()
        ),
        source_evidence=GallerySourceEvidence(
            sha256=("319295c0d98e00b67e7ae04e505705406557fe197a64cc813bc2b25ab8cebacd"),
            width=1600,
            height=1000,
            duration_seconds=36.0,
        ),
    )


class NapariRoiNavigationGalleryScenarioCatalog(GalleryScenarioCatalog):
    """Add the Napari ROI-navigation scenario to the gallery MRO."""

    scenario = MotionGalleryScenario(
        scenario_id="napari-roi-navigation",
        label="Napari",
        heading="ROI selection navigates the Z stack",
        description=(
            "A pointer-free MCP sequence selects measurement rows across a "
            "three-plane result. Napari scrolls the table, highlights the native "
            "ROI, and moves from Z 3/3 to Z 1/3 and back."
        ),
        proof=(
            "MCP row selection scrolls the ROI table, updates native Shapes "
            "selection, and moves between Z 3/3 and Z 1/3."
        ),
        layout_class="gallery-card-feature",
        width=1600,
        height=896,
        alt_text=(
            "Napari showing segmented nuclei, a selected native ROI, its "
            "measurement row, and the current Z plane"
        ),
        download_label="Download the Napari ROI-navigation recording.",
        open_aria_label=("Open the Napari ROI-navigation poster at full resolution"),
        capture_target=NapariViewerWindowCaptureTarget(),
        source_evidence=GallerySourceEvidence(
            sha256=("b8d8154e89474de1b8ff51d1ec099e83fba725068f2a1d0612b9882596954249"),
            width=2304,
            height=1290,
            duration_seconds=30.0,
        ),
    )


class OpenHCSGalleryScenarioCatalog(
    MultiPlateOverviewGalleryScenarioCatalog,
    PipelineEditorGalleryScenarioCatalog,
    LazyInheritanceGalleryScenarioCatalog,
    FijiReviewGalleryScenarioCatalog,
    ZmqStartupCompileGalleryScenarioCatalog,
    NapariRoiNavigationGalleryScenarioCatalog,
):
    """Complete public gallery scenario lattice."""


@dataclass(frozen=True, slots=True)
class GalleryCaptureContract:
    """Shared acquisition contract for the accepted gallery media."""

    surface: str
    visible_interaction_driver: str
    mouse_visible: bool
    source_formats: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GalleryDatasetAttribution:
    """Dataset provenance rendered below the application gallery."""

    dataset_note: str
    attribution_url: str
    attribution_label: str

    def render_provenance(self) -> str:
        return "\n".join(
            (
                '        <aside class="gallery-provenance" aria-label="Gallery dataset provenance">',
                "          <span>Dataset note</span>",
                "          <p>",
                f"            {escape(self.dataset_note)}",
                f'            <a href="{escape(self.attribution_url, quote=True)}">{escape(self.attribution_label)}</a>.',
                "            The pointer-free clips have a",
                f'            <a href="assets/gallery/{RELEASE_MEDIA_RECORD_NAME}">capture and checksum record</a>.',
                "          </p>",
                "        </aside>",
            )
        )


@dataclass(frozen=True, slots=True)
class GalleryReleaseContext:
    """Shared identity and capture contract for one gallery release."""

    captured_at: str
    capture_contract: GalleryCaptureContract


@dataclass(frozen=True, slots=True)
class GalleryReleaseDeclaration(GalleryReleaseContext):
    """Authoritative release context plus website-only attribution."""

    dataset_attribution: GalleryDatasetAttribution

    def project_record(self, asset_root: Path) -> GalleryReleaseRecord:
        """Project this release and every registered scenario into one record."""

        return GalleryReleaseRecord(
            schema_version=RELEASE_MEDIA_SCHEMA_VERSION,
            captured_at=self.captured_at,
            capture_contract=self.capture_contract,
            captures=tuple(
                scenario.release_record(asset_root) for scenario in gallery_scenarios()
            ),
        )


@dataclass(frozen=True, slots=True)
class GalleryReleaseRecord(GalleryReleaseContext):
    """Complete nominal release-media record before JSON serialization."""

    schema_version: str
    captures: tuple[GalleryScenarioReleaseRecord, ...]


GALLERY_RELEASE = GalleryReleaseDeclaration(
    captured_at="2026-08-09",
    capture_contract=GalleryCaptureContract(
        surface="real OpenHCS, Napari, and Fiji/ImageJ X11 windows",
        visible_interaction_driver="local MCP calls and native viewer controls",
        mouse_visible=False,
        source_formats=("FFV1", "PNG"),
    ),
    dataset_attribution=GalleryDatasetAttribution(
        dataset_note=(
            "CellProfiler ExampleCometAssay and ExampleWoundHealing are shown under "
            "BSD-3-Clause. Comet fluorescence images were contributed by Scott "
            "Floyd and Michael Pacold."
        ),
        attribution_url=(
            "https://github.com/CellProfiler/examples/tree/"
            "4972b59e670a4ae96c3d453803c92eeff378d054"
        ),
        attribution_label="Source and attribution",
    ),
)


def gallery_scenarios() -> tuple[GalleryScenarioABC, ...]:
    """Return the validated MRO-composed declarations in presentation order."""

    scenarios = OpenHCSGalleryScenarioCatalog.scenarios()
    scenario_ids = tuple(scenario.scenario_id for scenario in scenarios)
    published_paths = tuple(
        path for scenario in scenarios for path in scenario.published_paths()
    )
    if len(set(scenario_ids)) != len(scenario_ids):
        raise GalleryCatalogError("Gallery scenario ids must be unique.")
    if len(set(published_paths)) != len(published_paths):
        raise GalleryCatalogError("Gallery published asset paths must be unique.")
    for scenario in scenarios:
        scenario_id = scenario.scenario_id
        if SCENARIO_ID_PATTERN.fullmatch(scenario_id) is None:
            raise GalleryCatalogError(f"Invalid gallery scenario id: {scenario_id!r}.")
        if scenario.width <= 0 or scenario.height <= 0:
            raise GalleryCatalogError(
                f"Gallery scenario {scenario_id!r} has invalid dimensions."
            )
    return scenarios


def gallery_published_paths() -> tuple[str, ...]:
    """Return the declaration-derived public gallery asset inventory."""

    return tuple(
        path for scenario in gallery_scenarios() for path in scenario.published_paths()
    )


def render_gallery_cards() -> str:
    """Render every website card from the nominal scenario declarations."""

    return "\n\n".join(scenario.render_card() for scenario in gallery_scenarios())


def project_gallery_markup(index_path: Path) -> None:
    """Replace the two gallery tokens in one staged landing page."""

    document = index_path.read_text(encoding="utf-8")
    replacements = {
        GALLERY_CARDS_TOKEN: render_gallery_cards(),
        GALLERY_PROVENANCE_TOKEN: (
            GALLERY_RELEASE.dataset_attribution.render_provenance()
        ),
    }
    for token, rendered in replacements.items():
        token_count = document.count(token)
        if token_count != 1:
            raise GalleryCatalogError(
                f"Landing page must contain exactly one {token!r}; found {token_count}."
            )
        document = document.replace(token, rendered)
    index_path.write_text(document, encoding="utf-8")


def gallery_release_record(repo_root: Path) -> GalleryReleaseRecord:
    """Project the checked-in release record from declarations and assets."""

    asset_root = repo_root / "website" / "assets" / "gallery"
    return GALLERY_RELEASE.project_record(asset_root)


def gallery_release_record_text(repo_root: Path) -> str:
    """Serialize the release record deterministically."""

    return json.dumps(to_jsonable(gallery_release_record(repo_root)), indent=2) + "\n"


def validate_gallery_assets(repo_root: Path) -> None:
    """Require the asset directory to equal the declaration-derived inventory."""

    asset_root = repo_root / "website" / "assets" / "gallery"
    expected = frozenset((*gallery_published_paths(), RELEASE_MEDIA_RECORD_NAME))
    actual = frozenset(path.name for path in asset_root.iterdir() if path.is_file())
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise GalleryCatalogError(
            "Gallery asset inventory drifted from declarations: "
            f"missing={missing}, unexpected={unexpected}."
        )


def synchronize_gallery_release_record(repo_root: Path, *, check: bool) -> None:
    """Write or verify the checked-in generated release-media record."""

    validate_gallery_assets(repo_root)
    record_path = (
        repo_root / "website" / "assets" / "gallery" / RELEASE_MEDIA_RECORD_NAME
    )
    expected = gallery_release_record_text(repo_root)
    if check:
        actual = record_path.read_text(encoding="utf-8")
        if actual != expected:
            raise GalleryCatalogError(
                f"Generated gallery record is stale: {record_path}. "
                "Run python -m scripts.gallery_catalog."
            )
        return
    record_path.write_text(expected, encoding="utf-8")


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise GalleryCatalogError(f"Missing declared gallery asset: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    """Synchronize or verify the generated gallery release record."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        synchronize_gallery_release_record(arguments.repo_root, check=arguments.check)
    except (GalleryCatalogError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
