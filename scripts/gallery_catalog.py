"""Nominal source declarations for the public OpenHCS application gallery."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from html import escape
from pathlib import Path
from typing import ClassVar

from python_introspect import dataclass_from_mapping

from openhcs.agent.ui_bridge_identities import (
    GlobalConfigWindowIdentity,
    ImageBrowserWindowIdentity,
    LogViewerWindowIdentity,
    MainWindowWidgetIdentity,
    PipelineEditorWidgetIdentity,
    UiStableWindowIdentityDeclaration,
)
from openhcs.serialization.json import JsonValue, to_jsonable

RELEASE_MEDIA_SCHEMA_VERSION = "openhcs.release-media.v7"
RELEASE_MEDIA_RECORD_NAME = "release-media-record.json"
SOURCE_CAPTURE_EVIDENCE_SCHEMA_VERSION = 1
SOURCE_CAPTURE_EVIDENCE_RECORD_NAME = "source-capture-evidence.json"
GALLERY_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GALLERY_ASSET_RELATIVE_ROOT = Path("website/assets/gallery")
SCENARIO_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class GalleryCatalogError(RuntimeError):
    """Raised when a gallery declaration or generated projection is invalid."""


@dataclass(frozen=True, slots=True)
class GalleryImageDimensions:
    """Dimensions read from one generated WebP derivative."""

    width: int
    height: int


def gallery_image_dimensions(path: Path) -> GalleryImageDimensions:
    """Read WebP dimensions from the generated asset that owns them."""

    try:
        payload = path.read_bytes()
    except OSError as error:
        raise GalleryCatalogError(
            f"Cannot read gallery image {path}: {error}"
        ) from error
    if len(payload) < 20 or payload[:4] != b"RIFF" or payload[8:12] != b"WEBP":
        raise GalleryCatalogError(f"Gallery image is not a valid WebP file: {path}")

    offset = 12
    while offset + 8 <= len(payload):
        chunk_type = payload[offset : offset + 4]
        chunk_size = int.from_bytes(payload[offset + 4 : offset + 8], "little")
        chunk = payload[offset + 8 : offset + 8 + chunk_size]
        if len(chunk) != chunk_size:
            raise GalleryCatalogError(f"Gallery WebP chunk is truncated: {path}")
        if chunk_type == b"VP8X" and len(chunk) >= 10:
            return GalleryImageDimensions(
                width=int.from_bytes(chunk[4:7], "little") + 1,
                height=int.from_bytes(chunk[7:10], "little") + 1,
            )
        if chunk_type == b"VP8 " and len(chunk) >= 10 and chunk[3:6] == b"\x9d\x01\x2a":
            return GalleryImageDimensions(
                width=int.from_bytes(chunk[6:8], "little") & 0x3FFF,
                height=int.from_bytes(chunk[8:10], "little") & 0x3FFF,
            )
        if chunk_type == b"VP8L" and len(chunk) >= 5 and chunk[0] == 0x2F:
            packed = int.from_bytes(chunk[1:5], "little")
            return GalleryImageDimensions(
                width=(packed & 0x3FFF) + 1,
                height=((packed >> 14) & 0x3FFF) + 1,
            )
        offset += 8 + chunk_size + (chunk_size & 1)
    raise GalleryCatalogError(f"Gallery WebP has no supported image chunk: {path}")


class GalleryDerivativeRole(Enum):
    """Nominal role and filename suffix for one published derivative."""

    IMAGE = ("image", "")
    POSTER = ("poster", "-poster")
    WEB_VIDEO = ("web_video", "")
    FALLBACK_VIDEO = ("fallback_video", "")

    def __new__(cls, value: str, filename_suffix: str):
        member = object.__new__(cls)
        member._value_ = value
        member.filename_suffix = filename_suffix
        return member

    def path_for(self, scenario_id: str, media_type: GalleryMediaType) -> str:
        """Return this role's declaration-derived asset path."""

        return f"{scenario_id}{self.filename_suffix}{media_type.filename_suffix}"


class GalleryMediaType(Enum):
    """Nominal media type and filename suffix for a gallery derivative."""

    WEBP = ("image/webp", ".webp")
    WEBM = ("video/webm", ".webm")
    MP4 = ("video/mp4", ".mp4")

    def __new__(cls, value: str, filename_suffix: str):
        member = object.__new__(cls)
        member._value_ = value
        member.filename_suffix = filename_suffix
        return member


class GalleryPointerVisibility(Enum):
    """Pointer visibility across the heterogeneous release-media set."""

    HIDDEN = "hidden"
    VISIBLE = "visible"
    MIXED = "mixed"


class GalleryPublicationTarget(Enum):
    """Public projection that may consume a declared media scenario."""

    WEBSITE = "website"
    DOCUMENTATION = "documentation"

    def select(
        self,
        scenarios: Sequence[GalleryScenarioABC],
    ) -> tuple[GalleryScenarioABC, ...]:
        """Select scenarios published to this target in declaration order."""

        return tuple(
            scenario for scenario in scenarios if self in scenario.publication_targets
        )

    def for_id(
        self,
        scenarios: Sequence[GalleryScenarioABC],
        scenario_id: str,
    ) -> GalleryScenarioABC:
        """Resolve one scenario published to this target."""

        matches = tuple(
            scenario
            for scenario in self.select(scenarios)
            if scenario.scenario_id == scenario_id
        )
        if len(matches) != 1:
            raise GalleryCatalogError(
                f"Expected one {self.value} gallery scenario for {scenario_id!r}; "
                f"found {len(matches)}."
            )
        return matches[0]

    @classmethod
    def ordered_values(
        cls,
        targets: frozenset[GalleryPublicationTarget],
    ) -> tuple[str, ...]:
        """Serialize a target set in declaration order."""

        return tuple(target.value for target in cls if target in targets)


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

    def as_evidence(self) -> GallerySourceEvidence:
        """Drop the private session path while retaining immutable source facts."""

        return GallerySourceEvidence(
            sha256=self.sha256,
            width=self.width,
            height=self.height,
            duration_seconds=self.duration_seconds,
            format=self.format,
        )


@dataclass(frozen=True, slots=True)
class GallerySourceEvidenceEntry:
    """Generated source evidence associated with one declared scenario."""

    scenario_id: str
    source: GallerySourceEvidence


@dataclass(frozen=True, slots=True)
class GallerySourceEvidenceRecord:
    """Generated source-capture facts, separate from scenario semantics."""

    captures: tuple[GallerySourceEvidenceEntry, ...] = ()
    schema_version: int = SOURCE_CAPTURE_EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SOURCE_CAPTURE_EVIDENCE_SCHEMA_VERSION:
            raise GalleryCatalogError(
                f"Unsupported source-evidence schema {self.schema_version}; expected "
                f"{SOURCE_CAPTURE_EVIDENCE_SCHEMA_VERSION}."
            )
        scenario_ids = tuple(capture.scenario_id for capture in self.captures)
        if len(set(scenario_ids)) != len(scenario_ids):
            raise GalleryCatalogError(
                "Source evidence contains duplicate scenario ids."
            )

    def source_for(self, scenario_id: str) -> GallerySourceEvidence | None:
        """Resolve optional capture evidence for one scenario."""

        matches = tuple(
            capture.source
            for capture in self.captures
            if capture.scenario_id == scenario_id
        )
        return None if not matches else matches[0]

    def merge(
        self,
        entries: Sequence[GallerySourceEvidenceEntry],
    ) -> GallerySourceEvidenceRecord:
        """Replace supplied scenario evidence while preserving other captures."""

        replacements = {entry.scenario_id: entry for entry in entries}
        retained = tuple(
            entry for entry in self.captures if entry.scenario_id not in replacements
        )
        return GallerySourceEvidenceRecord(
            captures=(*retained, *replacements.values()),
        )


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
    create_if_missing: bool = False


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

    role: str
    media_type: str
    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class GalleryDerivativeExpectation:
    """One publication derivative required by a scenario media contract."""

    role: GalleryDerivativeRole
    media_type: GalleryMediaType

    def path_for(self, scenario_id: str) -> str:
        """Return the derivative path owned by its role and media type."""

        return self.role.path_for(scenario_id, self.media_type)


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
    publication_targets: tuple[str, ...]
    proof: str
    capture_target: GalleryCaptureTargetReleaseRecord
    source: GallerySourceEvidence | None
    scientific_evidence: GalleryScientificEvidenceABC | None
    published: tuple[GalleryPublishedAssetRecord, ...]


@dataclass(frozen=True, slots=True)
class GalleryWebsiteCardReleaseRecord:
    """Website card projected from one publication-enabled scenario."""

    id: str
    rendered_html: str


@dataclass(frozen=True, slots=True)
class GalleryScenarioCatalogRecord:
    """Declaration-derived capture catalog entry for maintainer tooling."""

    scenario_id: str
    publication_targets: tuple[str, ...]
    published_paths: tuple[str, ...]
    capture_target: GalleryCaptureTargetReleaseRecord
    proof: str


class GalleryScenarioDeclarationABC(ABC):
    """Nominal owner that projects one or more gallery scenarios."""

    @abstractmethod
    def scenarios(self) -> tuple[GalleryScenarioABC, ...]:
        """Return the scenarios owned by this declaration."""


@dataclass(frozen=True, slots=True, kw_only=True)
class GalleryScenarioABC(GalleryScenarioDeclarationABC, ABC):
    """Declaration owner for one public gallery scenario and all its views."""

    scenario_id: str
    label: str
    heading: str
    description: str
    proof: str
    layout_class: str
    alt_text: str
    capture_target: GalleryCaptureTargetABC
    publication_targets: frozenset[GalleryPublicationTarget]
    scientific_evidence: GalleryScientificEvidenceABC | None = None
    media_card_class: ClassVar[str | None] = None

    def scenarios(self) -> tuple[GalleryScenarioABC, ...]:
        """Project this leaf declaration as one scenario."""

        return (self,)

    @abstractmethod
    def derivative_expectations(self) -> tuple[GalleryDerivativeExpectation, ...]:
        """Return the complete derivative contract for this scenario."""

    @abstractmethod
    def render_media(self, asset_root: Path) -> str:
        """Render this scenario's media element without caller-side dispatch."""

    @abstractmethod
    def representative_image_path(self) -> str:
        """Return this scenario's still or poster image for static projections."""

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

    def render_card(self, asset_root: Path) -> str:
        """Render the website card from this scenario declaration."""

        return "\n".join(
            (
                f'          <figure class="{escape(self.card_class(), quote=True)}">',
                self.render_media(asset_root),
                f"            <figcaption{self.caption_id_attribute()}>",
                f'              <span class="gallery-label">{escape(self.label)}</span>',
                f"              <h3>{escape(self.heading)}</h3>",
                f"              <p>{escape(self.description)}</p>",
                "            </figcaption>",
                "          </figure>",
            )
        )

    def published_paths(self) -> tuple[str, ...]:
        """Return paths derived from this scenario's derivative contract."""

        return tuple(
            derivative.path_for(self.scenario_id)
            for derivative in self.derivative_expectations()
        )

    def catalog_record(self) -> GalleryScenarioCatalogRecord:
        """Project this declaration for capture-tool discovery."""

        return GalleryScenarioCatalogRecord(
            scenario_id=self.scenario_id,
            publication_targets=GalleryPublicationTarget.ordered_values(
                self.publication_targets
            ),
            published_paths=self.published_paths(),
            capture_target=self.capture_target.release_record(),
            proof=self.proof,
        )

    def derivative_path(self, role: GalleryDerivativeRole) -> str:
        """Return the unique declared derivative path for one role."""

        matches = tuple(
            derivative.path_for(self.scenario_id)
            for derivative in self.derivative_expectations()
            if derivative.role is role
        )
        if len(matches) != 1:
            raise GalleryCatalogError(
                f"Gallery scenario {self.scenario_id!r} declares {len(matches)} "
                f"derivatives for role {role.value!r}; expected one."
            )
        return matches[0]

    def representative_image_dimensions(
        self,
        asset_root: Path,
    ) -> GalleryImageDimensions:
        """Read presentation dimensions from the generated representative asset."""

        return gallery_image_dimensions(asset_root / self.representative_image_path())

    def release_record(
        self,
        asset_root: Path,
        source_evidence: GallerySourceEvidence | None,
    ) -> GalleryScenarioReleaseRecord:
        """Project source, target, proof, and current published checksums."""

        return GalleryScenarioReleaseRecord(
            id=self.scenario_id,
            publication_targets=GalleryPublicationTarget.ordered_values(
                self.publication_targets
            ),
            proof=self.proof,
            capture_target=self.capture_target.release_record(),
            source=source_evidence,
            scientific_evidence=self.scientific_evidence,
            published=tuple(
                GalleryPublishedAssetRecord(
                    role=derivative.role.value,
                    media_type=derivative.media_type.value,
                    path=derivative.path_for(self.scenario_id),
                    sha256=_sha256_file(
                        asset_root / derivative.path_for(self.scenario_id)
                    ),
                )
                for derivative in self.derivative_expectations()
            ),
        )

    def website_card_release_record(
        self,
        asset_root: Path,
    ) -> GalleryWebsiteCardReleaseRecord:
        """Project the declaration-owned website card."""

        return GalleryWebsiteCardReleaseRecord(
            id=self.scenario_id,
            rendered_html=self.render_card(asset_root),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class StillGalleryScenarioABC(GalleryScenarioABC, ABC):
    """Gallery scenario represented by one full-resolution still image."""

    open_aria_label: str

    def derivative_expectations(self) -> tuple[GalleryDerivativeExpectation, ...]:
        return (
            GalleryDerivativeExpectation(
                role=GalleryDerivativeRole.IMAGE,
                media_type=GalleryMediaType.WEBP,
            ),
        )

    def representative_image_path(self) -> str:
        """Return the declared full-resolution still image."""

        return self.derivative_path(GalleryDerivativeRole.IMAGE)

    def render_media(self, asset_root: Path) -> str:
        asset_filename = self.representative_image_path()
        dimensions = self.representative_image_dimensions(asset_root)
        return "\n".join(
            (
                "            <a",
                '              class="gallery-media-link"',
                f'              href="assets/gallery/{escape(asset_filename, quote=True)}"',
                f'              aria-label="{escape(self.open_aria_label, quote=True)}"',
                "            >",
                "              <img",
                f'                src="assets/gallery/{escape(asset_filename, quote=True)}"',
                f'                width="{dimensions.width}"',
                f'                height="{dimensions.height}"',
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
class UiWindowReferenceGalleryScenario(UiBridgeStillGalleryScenario):
    """Documentation reference projected from one stable UI-window identity."""

    @classmethod
    def from_identity(
        cls,
        identity: type[UiStableWindowIdentityDeclaration],
    ) -> UiWindowReferenceGalleryScenario:
        """Build a documentation scenario without copying window identity facts."""

        window_id = identity.require_value()
        title = identity.require_title()
        return cls(
            scenario_id=f"ui-{window_id.replace('_', '-')}",
            label="Window reference",
            heading=title,
            description=(
                f"Reference view of the registered {title} surface in the "
                "OpenHCS desktop."
            ),
            proof=(
                f"The live {title} surface is captured through its registered "
                "stable UI-bridge identity."
            ),
            layout_class="gallery-card-wide",
            alt_text=f"{title} window in the OpenHCS desktop",
            open_aria_label=f"Open the {title} window screenshot at full resolution",
            capture_target=UiBridgeWindowCaptureTarget(
                window_id=window_id,
                create_if_missing=True,
            ),
            publication_targets=frozenset((GalleryPublicationTarget.DOCUMENTATION,)),
        )


@dataclass(frozen=True, slots=True)
class StableUiWindowReferenceGalleryDeclaration(GalleryScenarioDeclarationABC):
    """Derive complete stable-window reference coverage from UI declarations."""

    def scenarios(self) -> tuple[GalleryScenarioABC, ...]:
        return tuple(
            UiWindowReferenceGalleryScenario.from_identity(identity)
            for identity in UiStableWindowIdentityDeclaration.declaration_types()
        )


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

    def derivative_expectations(self) -> tuple[GalleryDerivativeExpectation, ...]:
        return (
            GalleryDerivativeExpectation(
                role=GalleryDerivativeRole.POSTER,
                media_type=GalleryMediaType.WEBP,
            ),
            GalleryDerivativeExpectation(
                role=GalleryDerivativeRole.WEB_VIDEO,
                media_type=GalleryMediaType.WEBM,
            ),
            GalleryDerivativeExpectation(
                role=GalleryDerivativeRole.FALLBACK_VIDEO,
                media_type=GalleryMediaType.MP4,
            ),
        )

    def caption_id_attribute(self) -> str:
        return f' id="{escape(self.scenario_id, quote=True)}-caption"'

    def representative_image_path(self) -> str:
        """Return the declared poster for static projections."""

        return self.derivative_path(GalleryDerivativeRole.POSTER)

    def render_media(self, asset_root: Path) -> str:
        poster_path = escape(self.representative_image_path(), quote=True)
        dimensions = self.representative_image_dimensions(asset_root)
        web_video_path = escape(
            self.derivative_path(GalleryDerivativeRole.WEB_VIDEO), quote=True
        )
        fallback_video_path = escape(
            self.derivative_path(GalleryDerivativeRole.FALLBACK_VIDEO), quote=True
        )
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
                f'                poster="assets/gallery/{poster_path}"',
                f'                aria-describedby="{caption_id}"',
                "              >",
                f'                <source src="assets/gallery/{web_video_path}" type="{GalleryMediaType.WEBM.value}">',
                f'                <source src="assets/gallery/{fallback_video_path}" type="{GalleryMediaType.MP4.value}">',
                f'                <a href="assets/gallery/{fallback_video_path}">{escape(self.download_label)}</a>',
                "              </video>",
                "              <a",
                '                class="gallery-motion-fallback gallery-media-link"',
                f'                href="assets/gallery/{poster_path}"',
                f'                aria-label="{escape(self.open_aria_label, quote=True)}"',
                "              >",
                "                <img",
                f'                  src="assets/gallery/{poster_path}"',
                f'                  width="{dimensions.width}"',
                f'                  height="{dimensions.height}"',
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
            scenario
            for owner_type in cls.__mro__
            if isinstance(
                declaration := owner_type.__dict__.get("declaration"),
                GalleryScenarioDeclarationABC,
            )
            for scenario in declaration.scenarios()
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

    declaration = UiBridgeStillGalleryScenario(
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
        publication_targets=frozenset(GalleryPublicationTarget),
    )


class PipelineEditorGalleryScenarioCatalog(GalleryScenarioCatalog):
    """Add the pipeline-editor scenario to the gallery MRO."""

    declaration = UiBridgeStillGalleryScenario(
        scenario_id="pipeline-editor",
        label="Authoring",
        heading="CellProfiler steps in the same pipeline model",
        description=(
            "The 12-step Comet Assay uses OpenHCS compilation, configuration, "
            "viewers, and results."
        ),
        proof=(
            "The imported CellProfiler Comet Assay is visible as twelve pipeline steps."
        ),
        layout_class="gallery-card-compact",
        alt_text=(
            "OpenHCS Pipeline Editor showing the 12 imported steps of the "
            "CellProfiler Comet Assay"
        ),
        open_aria_label="Open the pipeline editor screenshot at full resolution",
        capture_target=UiBridgeWindowCaptureTarget(
            window_id=PipelineEditorWidgetIdentity.require_value()
        ),
        publication_targets=frozenset((GalleryPublicationTarget.WEBSITE,)),
    )


class FirstPlateWorkflowGalleryScenarioCatalog(GalleryScenarioCatalog):
    """Add the first-plate workflow scenario to documentation media."""

    declaration = UiBridgeStillGalleryScenario(
        scenario_id="first-plate-workflow",
        label="First workflow",
        heading="Start with a complete synthetic plate and pipeline",
        description=(
            "The generated plate is selected in Plate Manager, while Pipeline "
            "Editor keeps the eight supplied steps together after initialisation and "
            "compilation."
        ),
        proof=(
            "One compiled synthetic plate and its complete eight-step pipeline are "
            "visible in the live workspace."
        ),
        layout_class="gallery-card-wide",
        alt_text=(
            "OpenHCS main window with one synthetic plate selected and its eight "
            "pipeline steps visible"
        ),
        open_aria_label="Open the first plate workflow screenshot at full resolution",
        capture_target=UiBridgeWindowCaptureTarget(
            window_id=MainWindowWidgetIdentity.require_value()
        ),
        publication_targets=frozenset((GalleryPublicationTarget.DOCUMENTATION,)),
    )


class SourceImageBrowserGalleryScenarioCatalog(GalleryScenarioCatalog):
    """Add the source-image browser scenario to documentation media."""

    declaration = UiBridgeStillGalleryScenario(
        scenario_id="source-image-browser",
        label="Source images",
        heading="Browse the resolved source-image inventory",
        description=(
            "The generated plate exposes 216 source files in the table, with viewer "
            "configuration and live viewer instances beside the inventory."
        ),
        proof=(
            "The live Image Browser reports 216 loaded source files for the generated "
            "plate and exposes the configured Fiji viewer."
        ),
        layout_class="gallery-card-wide",
        alt_text=(
            "OpenHCS Image Browser listing 216 synthetic source images beside Fiji "
            "viewer configuration"
        ),
        open_aria_label="Open the source image browser screenshot at full resolution",
        capture_target=UiBridgeWindowCaptureTarget(
            window_id=ImageBrowserWindowIdentity.require_value()
        ),
        publication_targets=frozenset((GalleryPublicationTarget.DOCUMENTATION,)),
    )


class GlobalConfigurationGalleryScenarioCatalog(GalleryScenarioCatalog):
    """Add the global-configuration scenario to documentation media."""

    declaration = UiBridgeStillGalleryScenario(
        scenario_id="global-configuration-editor",
        label="Configuration",
        heading="Edit typed configuration with contextual field help",
        description=(
            "GlobalPipelineConfig and UIConfig share the editor, where typed fields, "
            "field-level help controls, provenance feedback, and reset controls "
            "remain together."
        ),
        proof=(
            "The live global configuration editor exposes both configuration roots "
            "with typed values, field-level help, and reset controls."
        ),
        layout_class="gallery-card-compact",
        alt_text=(
            "OpenHCS global configuration editor showing configuration tabs, typed "
            "fields, help controls, and reset controls"
        ),
        open_aria_label=(
            "Open the global configuration editor screenshot at full resolution"
        ),
        capture_target=UiBridgeWindowCaptureTarget(
            window_id=GlobalConfigWindowIdentity.require_value()
        ),
        publication_targets=frozenset((GalleryPublicationTarget.DOCUMENTATION,)),
    )


class ExecutionLogViewerGalleryScenarioCatalog(GalleryScenarioCatalog):
    """Add the execution-log viewer scenario to documentation media."""

    declaration = UiBridgeStillGalleryScenario(
        scenario_id="execution-log-viewer",
        label="Logs",
        heading="Inspect the log for the active execution endpoint",
        description=(
            "The log selector is on the ZMQ execution server, with compilation and "
            "progress records presented using source and level colours."
        ),
        proof=(
            "The live Log Viewer is bound to the active ZMQ execution server and "
            "shows its compilation records."
        ),
        layout_class="gallery-card-wide",
        alt_text=(
            "OpenHCS Log Viewer showing the active ZMQ execution server log, "
            "compilation records, and log controls"
        ),
        open_aria_label="Open the execution log viewer screenshot at full resolution",
        capture_target=UiBridgeWindowCaptureTarget(
            window_id=LogViewerWindowIdentity.require_value()
        ),
        publication_targets=frozenset((GalleryPublicationTarget.DOCUMENTATION,)),
    )


class LazyInheritanceGalleryScenarioCatalog(GalleryScenarioCatalog):
    """Add the lazy-inheritance scenario to the gallery MRO."""

    declaration = MotionGalleryScenario(
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
        publication_targets=frozenset(GalleryPublicationTarget),
    )


class FijiReviewGalleryScenarioCatalog(GalleryScenarioCatalog):
    """Add the Fiji-review scenario to the gallery MRO."""

    declaration = HumanReviewedStillGalleryScenario(
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
        alt_text=(
            "Fiji ImageJ showing one NeuronCyto II Field 1 nuclear plane with its "
            "nine native segmentation ROI entries"
        ),
        open_aria_label="Open the Fiji integration screenshot at full resolution",
        capture_target=FijiViewerWindowCaptureTarget(),
        publication_targets=frozenset(GalleryPublicationTarget),
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

    declaration = MotionGalleryScenario(
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
        publication_targets=frozenset(GalleryPublicationTarget),
    )


class NapariRoiNavigationGalleryScenarioCatalog(GalleryScenarioCatalog):
    """Add the Napari ROI-navigation scenario to the gallery MRO."""

    declaration = MotionGalleryScenario(
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
        alt_text=(
            "Napari showing segmented nuclei, a selected native ROI, its "
            "measurement row, and the current Z plane"
        ),
        download_label="Download the Napari ROI-navigation recording.",
        open_aria_label=("Open the Napari ROI-navigation poster at full resolution"),
        capture_target=NapariViewerWindowCaptureTarget(),
        publication_targets=frozenset(GalleryPublicationTarget),
    )


class StableUiWindowReferenceGalleryScenarioCatalog(GalleryScenarioCatalog):
    """Add registry-derived stable-window reference scenarios to documentation."""

    declaration = StableUiWindowReferenceGalleryDeclaration()


class OpenHCSGalleryScenarioCatalog(
    MultiPlateOverviewGalleryScenarioCatalog,
    PipelineEditorGalleryScenarioCatalog,
    FirstPlateWorkflowGalleryScenarioCatalog,
    SourceImageBrowserGalleryScenarioCatalog,
    GlobalConfigurationGalleryScenarioCatalog,
    ExecutionLogViewerGalleryScenarioCatalog,
    LazyInheritanceGalleryScenarioCatalog,
    FijiReviewGalleryScenarioCatalog,
    ZmqStartupCompileGalleryScenarioCatalog,
    NapariRoiNavigationGalleryScenarioCatalog,
    StableUiWindowReferenceGalleryScenarioCatalog,
):
    """Complete public gallery scenario lattice."""


@dataclass(frozen=True, slots=True)
class GalleryCaptureContract:
    """Shared acquisition contract for the accepted gallery media."""

    surface: str
    visible_interaction_driver: str
    pointer_visibility: GalleryPointerVisibility
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
                "            The gallery media have a",
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

    def project_record(
        self,
        asset_root: Path,
        source_evidence: GallerySourceEvidenceRecord,
    ) -> GalleryReleaseRecord:
        """Project this release and every registered scenario into one record."""

        scenarios = gallery_scenarios()
        return GalleryReleaseRecord(
            schema_version=RELEASE_MEDIA_SCHEMA_VERSION,
            source_capture_evidence_path=SOURCE_CAPTURE_EVIDENCE_RECORD_NAME,
            captured_at=self.captured_at,
            capture_contract=self.capture_contract,
            dataset_attribution=self.dataset_attribution,
            website_provenance_html=self.dataset_attribution.render_provenance(),
            website_cards=tuple(
                scenario.website_card_release_record(asset_root)
                for scenario in GalleryPublicationTarget.WEBSITE.select(scenarios)
            ),
            captures=tuple(
                scenario.release_record(
                    asset_root,
                    source_evidence.source_for(scenario.scenario_id),
                )
                for scenario in scenarios
            ),
        )


@dataclass(frozen=True, slots=True)
class GalleryReleaseRecord(GalleryReleaseContext):
    """Complete nominal release-media record before JSON serialization."""

    schema_version: str
    source_capture_evidence_path: str
    dataset_attribution: GalleryDatasetAttribution
    website_provenance_html: str
    website_cards: tuple[GalleryWebsiteCardReleaseRecord, ...]
    captures: tuple[GalleryScenarioReleaseRecord, ...]


GALLERY_RELEASE = GalleryReleaseDeclaration(
    captured_at="2026-08-28",
    capture_contract=GalleryCaptureContract(
        surface="real OpenHCS, Napari, and Fiji/ImageJ X11 windows",
        visible_interaction_driver=(
            "local MCP calls, native viewer controls, and reviewed UI actions"
        ),
        pointer_visibility=GalleryPointerVisibility.MIXED,
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
        if not scenario.publication_targets:
            raise GalleryCatalogError(
                f"Gallery scenario {scenario_id!r} has no publication target."
            )
    return scenarios


def website_gallery_scenarios() -> tuple[GalleryScenarioABC, ...]:
    """Return scenarios declared for the public website projection."""

    return GalleryPublicationTarget.WEBSITE.select(gallery_scenarios())


def documentation_gallery_scenarios() -> tuple[GalleryScenarioABC, ...]:
    """Return scenarios declared for the documentation projection."""

    return GalleryPublicationTarget.DOCUMENTATION.select(gallery_scenarios())


def ui_window_reference_gallery_scenarios() -> (
    tuple[UiWindowReferenceGalleryScenario, ...]
):
    """Return complete stable-window documentation projected from identities."""

    return tuple(
        scenario
        for scenario in documentation_gallery_scenarios()
        if isinstance(scenario, UiWindowReferenceGalleryScenario)
    )


def documentation_gallery_scenario_for_id(scenario_id: str) -> GalleryScenarioABC:
    """Resolve one declaration available to the documentation projection."""

    return GalleryPublicationTarget.DOCUMENTATION.for_id(
        gallery_scenarios(), scenario_id
    )


def gallery_published_paths() -> tuple[str, ...]:
    """Return the declaration-derived public gallery asset inventory."""

    return tuple(
        path for scenario in gallery_scenarios() for path in scenario.published_paths()
    )


def gallery_asset_root(repo_root: Path = GALLERY_REPOSITORY_ROOT) -> Path:
    """Return the declaration-owned public gallery asset directory."""

    return repo_root / GALLERY_ASSET_RELATIVE_ROOT


def gallery_release_record(repo_root: Path) -> GalleryReleaseRecord:
    """Project the checked-in release record from declarations and assets."""

    return gallery_release_record_for_asset_root(gallery_asset_root(repo_root))


def read_gallery_source_evidence(asset_root: Path) -> GallerySourceEvidenceRecord:
    """Read the generated source-capture evidence projection."""

    path = asset_root / SOURCE_CAPTURE_EVIDENCE_RECORD_NAME
    if not path.is_file():
        return GallerySourceEvidenceRecord()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("source evidence root must be an object")
        return dataclass_from_mapping(GallerySourceEvidenceRecord, payload)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise GalleryCatalogError(
            f"Cannot read source-capture evidence {path}: {error}"
        ) from error


def gallery_release_record_for_asset_root(asset_root: Path) -> GalleryReleaseRecord:
    """Project a release record from declarations, assets, and generated evidence."""

    return GALLERY_RELEASE.project_record(
        asset_root,
        read_gallery_source_evidence(asset_root),
    )


def gallery_release_record_text(repo_root: Path) -> str:
    """Serialize the release record deterministically."""

    return json.dumps(to_jsonable(gallery_release_record(repo_root)), indent=2) + "\n"


def gallery_release_record_text_for_asset_root(asset_root: Path) -> str:
    """Serialize the release record for an explicit gallery asset root."""

    return (
        json.dumps(
            to_jsonable(gallery_release_record_for_asset_root(asset_root)),
            indent=2,
        )
        + "\n"
    )


def validate_gallery_assets(repo_root: Path) -> None:
    """Require the asset directory to equal the declaration-derived inventory."""

    asset_root = gallery_asset_root(repo_root)
    expected = frozenset(
        (
            *gallery_published_paths(),
            RELEASE_MEDIA_RECORD_NAME,
            SOURCE_CAPTURE_EVIDENCE_RECORD_NAME,
        )
    )
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
    asset_root = gallery_asset_root(repo_root)
    synchronize_gallery_release_record_for_asset_root(asset_root, check=check)


def synchronize_gallery_release_record_for_asset_root(
    asset_root: Path,
    *,
    check: bool,
) -> None:
    """Write or verify a record for a complete explicit gallery asset root."""

    record_path = asset_root / RELEASE_MEDIA_RECORD_NAME
    expected = gallery_release_record_text_for_asset_root(asset_root)
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
