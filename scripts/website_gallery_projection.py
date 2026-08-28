"""Dependency-free consumer for the generated website gallery projection."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

GALLERY_CARDS_TOKEN = "{{ OPENHCS_GALLERY_CARDS }}"
GALLERY_PROVENANCE_TOKEN = "{{ OPENHCS_GALLERY_PROVENANCE }}"
SCENARIO_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class WebsiteGalleryProjectionError(RuntimeError):
    """Raised when the checked-in gallery projection is invalid or stale."""


@dataclass(frozen=True, slots=True)
class ProjectedGalleryDerivative:
    """One declared derivative and its accepted content identity."""

    role: str
    media_type: str
    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class ProjectedGalleryCard:
    """Website-ready view of one declaration-owned gallery scenario."""

    scenario_id: str
    rendered_html: str
    derivatives: tuple[ProjectedGalleryDerivative, ...]


@dataclass(frozen=True, slots=True)
class ProjectedGalleryCapture:
    """Published assets projected from one release-media capture."""

    scenario_id: str
    derivatives: tuple[ProjectedGalleryDerivative, ...]


@dataclass(frozen=True, slots=True)
class WebsiteGalleryProjection:
    """Validated website projection loaded without importing OpenHCS."""

    cards: tuple[ProjectedGalleryCard, ...]
    captures: tuple[ProjectedGalleryCapture, ...]
    provenance_html: str

    @property
    def published_paths(self) -> tuple[str, ...]:
        """Return the declaration-ordered derivative inventory."""

        return tuple(
            derivative.path
            for capture in self.captures
            for derivative in capture.derivatives
        )

    @property
    def rendered_cards(self) -> str:
        """Return card markup already rendered by scenario declarations."""

        return "\n\n".join(card.rendered_html for card in self.cards)


def read_website_gallery_projection(record_path: Path) -> WebsiteGalleryProjection:
    """Load and validate the checked-in gallery catalog projection."""

    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WebsiteGalleryProjectionError(
            f"Cannot read gallery projection {record_path}: {error}"
        ) from error
    if not isinstance(record, dict):
        raise WebsiteGalleryProjectionError(
            "Gallery projection root must be an object."
        )
    captures_payload = record.get("captures")
    website_cards_payload = record.get("website_cards")
    provenance_html = record.get("website_provenance_html")
    if not isinstance(captures_payload, list) or not captures_payload:
        raise WebsiteGalleryProjectionError(
            "Gallery projection must contain at least one capture."
        )
    if not isinstance(website_cards_payload, list) or not website_cards_payload:
        raise WebsiteGalleryProjectionError(
            "Gallery projection must contain at least one website card."
        )
    if not isinstance(provenance_html, str) or not provenance_html.strip():
        raise WebsiteGalleryProjectionError(
            "Gallery projection has no rendered website provenance."
        )

    captures = tuple(_read_capture(capture) for capture in captures_payload)
    _require_unique(
        (capture.scenario_id for capture in captures), "capture scenario ids"
    )
    captures_by_id = {capture.scenario_id: capture for capture in captures}
    cards = tuple(_read_card(card, captures_by_id) for card in website_cards_payload)
    _require_unique((card.scenario_id for card in cards), "scenario ids")
    _require_unique(
        (derivative.path for capture in captures for derivative in capture.derivatives),
        "published derivative paths",
    )
    projection = WebsiteGalleryProjection(
        cards=cards,
        captures=captures,
        provenance_html=provenance_html,
    )
    _validate_projected_assets(record_path, projection)
    return projection


def project_gallery_markup(
    index_path: Path,
    projection: WebsiteGalleryProjection,
) -> None:
    """Insert declaration-rendered gallery markup into a staged landing page."""

    document = index_path.read_text(encoding="utf-8")
    replacements = {
        GALLERY_CARDS_TOKEN: projection.rendered_cards,
        GALLERY_PROVENANCE_TOKEN: projection.provenance_html,
    }
    for token, rendered in replacements.items():
        token_count = document.count(token)
        if token_count != 1:
            raise WebsiteGalleryProjectionError(
                f"Landing page must contain exactly one {token!r}; found {token_count}."
            )
        document = document.replace(token, rendered)
    index_path.write_text(document, encoding="utf-8")


def _read_capture(capture: object) -> ProjectedGalleryCapture:
    if not isinstance(capture, dict):
        raise WebsiteGalleryProjectionError("Gallery captures must be objects.")
    scenario_id = _required_string(capture, "id")
    if SCENARIO_ID_PATTERN.fullmatch(scenario_id) is None:
        raise WebsiteGalleryProjectionError(f"Invalid gallery id {scenario_id!r}.")
    published = capture.get("published")
    if not isinstance(published, list) or not published:
        raise WebsiteGalleryProjectionError(
            f"Gallery scenario {scenario_id!r} has no published derivatives."
        )
    derivatives = tuple(
        _read_derivative(item, scenario_id=scenario_id) for item in published
    )
    return ProjectedGalleryCapture(
        scenario_id=scenario_id,
        derivatives=derivatives,
    )


def _read_card(
    card: object,
    captures_by_id: dict[str, ProjectedGalleryCapture],
) -> ProjectedGalleryCard:
    if not isinstance(card, dict):
        raise WebsiteGalleryProjectionError("Gallery website cards must be objects.")
    scenario_id = _required_string(card, "id")
    capture = captures_by_id.get(scenario_id)
    if capture is None:
        raise WebsiteGalleryProjectionError(
            f"Website card {scenario_id!r} has no matching gallery capture."
        )
    return ProjectedGalleryCard(
        scenario_id=scenario_id,
        rendered_html=_required_string(card, "rendered_html"),
        derivatives=capture.derivatives,
    )


def _read_derivative(
    published: object,
    *,
    scenario_id: str,
) -> ProjectedGalleryDerivative:
    if not isinstance(published, dict):
        raise WebsiteGalleryProjectionError(
            f"Gallery scenario {scenario_id!r} has a non-object derivative."
        )
    role = _required_string(published, "role")
    media_type = _required_string(published, "media_type")
    path = _required_string(published, "path")
    sha256 = _required_string(published, "sha256")
    pure_path = PurePosixPath(path)
    if pure_path.name != path or path.startswith("."):
        raise WebsiteGalleryProjectionError(
            f"Gallery scenario {scenario_id!r} has unsafe derivative path {path!r}."
        )
    if not path.startswith((f"{scenario_id}.", f"{scenario_id}-")):
        raise WebsiteGalleryProjectionError(
            f"Gallery derivative {path!r} does not use scenario id {scenario_id!r}."
        )
    if SHA256_PATTERN.fullmatch(sha256) is None:
        raise WebsiteGalleryProjectionError(
            f"Gallery derivative {path!r} has an invalid sha256."
        )
    return ProjectedGalleryDerivative(
        role=role,
        media_type=media_type,
        path=path,
        sha256=sha256,
    )


def _validate_projected_assets(
    record_path: Path,
    projection: WebsiteGalleryProjection,
) -> None:
    asset_root = record_path.parent
    expected = frozenset((*projection.published_paths, record_path.name))
    try:
        actual = frozenset(path.name for path in asset_root.iterdir() if path.is_file())
    except OSError as error:
        raise WebsiteGalleryProjectionError(
            f"Cannot inspect gallery assets at {asset_root}: {error}"
        ) from error
    if actual != expected:
        raise WebsiteGalleryProjectionError(
            "Gallery assets drifted from the checked-in projection: "
            f"missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}."
        )
    for capture in projection.captures:
        for derivative in capture.derivatives:
            artifact_path = asset_root / derivative.path
            with artifact_path.open("rb") as artifact:
                actual_sha256 = hashlib.file_digest(artifact, "sha256").hexdigest()
            if actual_sha256 != derivative.sha256:
                raise WebsiteGalleryProjectionError(
                    f"Gallery derivative checksum drifted: {artifact_path}."
                )


def _required_string(payload: dict[object, object], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise WebsiteGalleryProjectionError(
            f"Gallery projection field {field_name!r} must be a non-empty string."
        )
    return value


def _require_unique(values: Iterable[str], description: str) -> None:
    materialized = tuple(values)
    if len(set(materialized)) != len(materialized):
        raise WebsiteGalleryProjectionError(
            f"Gallery projection contains duplicate {description}."
        )
