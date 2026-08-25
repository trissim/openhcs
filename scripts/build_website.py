#!/usr/bin/env python3
"""Build and validate the static OpenHCS landing-page artifact."""

from __future__ import annotations

import argparse
import ast
import html
import re
import shutil
import tomllib
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

from scripts.website_gallery_projection import (
    GALLERY_CARDS_TOKEN,
    GALLERY_PROVENANCE_TOKEN,
    project_gallery_markup,
    read_website_gallery_projection,
)

HTML_SOURCE_FILES = (
    "index.html",
    "privacy.html",
    "support.html",
    "terms.html",
)
SOURCE_FILES = HTML_SOURCE_FILES + (
    "styles.css",
    "globals.css",
    "assets/logos/bioformats.svg",
    "assets/logos/cellprofiler.png",
    "assets/logos/cupy.svg",
    "assets/logos/fiji.svg",
    "assets/logos/jax.png",
    "assets/logos/napari.svg",
    "assets/logos/pyclesperanto.png",
    "assets/logos/pytorch.svg",
    "assets/logos/tensorflow.svg",
)
ASSET_SOURCES = {
    "assets/logos/openhcs-favicon.svg": (
        "openhcs/resources/assets/openhcs-favicon.svg"
    ),
    "assets/logos/openhcs-horizontal.svg": (
        "openhcs/resources/assets/openhcs-lockup-horizontal.svg"
    ),
    "assets/logos/openhcs-stacked.svg": (
        "openhcs/resources/assets/openhcs-lockup-stacked.svg"
    ),
}
REQUIRED_COPY = (
    "PyPI",
    ".cppipe",
    "Napari",
    "Fiji",
    "MCP",
    "CuPy",
    "PyTorch",
    "JAX",
    "TensorFlow",
    "pyclesperanto",
    "custom functions",
    "Compute with",
    "Your functions",
    "Bio-Formats",
    "Bio-Formats image I/O",
    "openhcs[bioformats]",
)
ALLOWED_REMOTE_SCHEMES = {"data", "https", "mailto"}
RELEASE_VERSION_TOKEN = "{{ OPENHCS_VERSION }}"
CONTACT_EMAIL_TOKEN = "{{ OPENHCS_CONTACT_EMAIL }}"
MCP_CLIENT_MARKS_TOKEN = "{{ MCP_CLIENT_MARKS }}"
MCP_CLIENT_REGISTRATION_SOURCE = "openhcs/mcp/client_registration.py"
MCP_CLIENT_TARGET_ID_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
GALLERY_RECORD_RELATIVE_PATH = "assets/gallery/release-media-record.json"


@dataclass(frozen=True, slots=True)
class WebsiteMcpClient:
    """Website projection of one production-owned MCP registration target."""

    target_id: str
    display_name: str

    @property
    def logo_path(self) -> str:
        """Return the presentation asset derived from the stable target id."""

        return f"assets/logos/client-{self.target_id}.svg"


def read_package_version(repo_root: Path) -> str:
    """Read the literal package version without importing OpenHCS."""

    init_path = repo_root / "openhcs" / "__init__.py"
    module = ast.parse(
        init_path.read_text(encoding="utf-8"),
        filename=str(init_path),
    )
    for statement in module.body:
        if not isinstance(statement, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in statement.targets
        ):
            continue
        if isinstance(statement.value, ast.Constant) and isinstance(
            statement.value.value,
            str,
        ):
            return statement.value.value
    raise ValueError(f"No literal __version__ assignment found in {init_path}")


def read_package_contact_email(repo_root: Path) -> str:
    """Read the first declared project-author email from package metadata."""

    pyproject_path = repo_root / "pyproject.toml"
    with pyproject_path.open("rb") as stream:
        metadata = tomllib.load(stream)
    authors = metadata.get("project", {}).get("authors", ())
    for author in authors:
        if not isinstance(author, dict):
            continue
        email_address = author.get("email")
        if isinstance(email_address, str) and email_address.strip():
            return email_address.strip()
    raise ValueError(f"No project author email found in {pyproject_path}")


def _literal_class_fields(class_node: ast.ClassDef) -> dict[str, object]:
    """Read literal class declarations without importing OpenHCS."""

    fields: dict[str, object] = {}
    for statement in class_node.body:
        field_name: str | None = None
        value: ast.expr | None = None
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name):
                field_name = target.id
                value = statement.value
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            field_name = statement.target.id
            value = statement.value
        if field_name is not None and isinstance(value, ast.Constant):
            fields[field_name] = value.value
    return fields


def read_mcp_client_targets(repo_root: Path) -> tuple[WebsiteMcpClient, ...]:
    """Project registered MCP client identities from their nominal declarations.

    The static site build intentionally parses the declaration module instead of
    importing OpenHCS and its optional dependency graph. Target identity and
    ordering remain owned by ``McpClientRegistrationTarget`` subclasses.
    """

    source_path = repo_root / MCP_CLIENT_REGISTRATION_SOURCE
    module = ast.parse(
        source_path.read_text(encoding="utf-8"),
        filename=str(source_path),
    )
    targets: list[WebsiteMcpClient] = []
    seen_target_ids: set[str] = set()
    for class_node in (node for node in module.body if isinstance(node, ast.ClassDef)):
        fields = _literal_class_fields(class_node)
        target_id = fields.get("target_id")
        display_name = fields.get("display_name")
        if target_id is None and display_name is None:
            continue
        if target_id is None:
            continue
        if (
            not isinstance(target_id, str)
            or MCP_CLIENT_TARGET_ID_PATTERN.fullmatch(target_id) is None
        ):
            raise ValueError(
                f"MCP client target {class_node.name} has an invalid literal target_id: "
                f"{target_id!r}"
            )
        if not isinstance(display_name, str) or not display_name.strip():
            raise ValueError(
                f"MCP client target {class_node.name} has no literal display_name."
            )
        if target_id in seen_target_ids:
            raise ValueError(f"Duplicate MCP client target_id: {target_id!r}")
        seen_target_ids.add(target_id)
        targets.append(
            WebsiteMcpClient(
                target_id=target_id,
                display_name=display_name.strip(),
            )
        )
    if not targets:
        raise ValueError("No registered MCP client targets were found.")
    return tuple(targets)


def project_mcp_client_marks(
    index_path: Path,
    clients: tuple[WebsiteMcpClient, ...],
) -> None:
    """Render the production-owned local-client roster into the landing page."""

    document = index_path.read_text(encoding="utf-8")
    token_count = document.count(MCP_CLIENT_MARKS_TOKEN)
    if token_count != 1:
        raise ValueError(
            "Landing page must contain exactly one "
            f"{MCP_CLIENT_MARKS_TOKEN!r} token; found {token_count}."
        )
    rendered = "\n".join(
        (
            '<li class="client-mark">'
            f'<img src="{html.escape(client.logo_path, quote=True)}" '
            'width="24" height="24" alt="">'
            f"<span>{html.escape(client.display_name)}</span>"
            "</li>"
        )
        for client in clients
    )
    index_path.write_text(
        document.replace(MCP_CLIENT_MARKS_TOKEN, rendered),
        encoding="utf-8",
    )


def project_release_version(index_path: Path, package_version: str) -> None:
    """Project the package-owned version into the staged landing page."""

    html = index_path.read_text(encoding="utf-8")
    token_count = html.count(RELEASE_VERSION_TOKEN)
    if token_count == 0:
        raise ValueError(
            f"Landing page contains no {RELEASE_VERSION_TOKEN!r} release token"
        )
    index_path.write_text(
        html.replace(RELEASE_VERSION_TOKEN, package_version),
        encoding="utf-8",
    )


def project_contact_email(page_path: Path, contact_email: str) -> None:
    """Project the package-author contact into one staged public page."""

    document = page_path.read_text(encoding="utf-8")
    token_count = document.count(CONTACT_EMAIL_TOKEN)
    if token_count == 0:
        raise ValueError(
            f"Public page {page_path.name} contains no "
            f"{CONTACT_EMAIL_TOKEN!r} contact token"
        )
    page_path.write_text(
        document.replace(
            CONTACT_EMAIL_TOKEN,
            html.escape(contact_email, quote=True),
        ),
        encoding="utf-8",
    )


class _ReferenceCollector(HTMLParser):
    """Collect IDs and browser-loaded references from one HTML document."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()
        self.duplicate_ids: set[str] = set()
        self.references: list[tuple[str, str]] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attributes = dict(attrs)
        element_id = attributes.get("id")
        if element_id:
            if element_id in self.ids:
                self.duplicate_ids.add(element_id)
            self.ids.add(element_id)
        for attribute in ("href", "src", "poster"):
            value = attributes.get(attribute)
            if value:
                self.references.append((attribute, value))


def referenced_source_files(source_dir: Path) -> tuple[str, ...]:
    """Return local files selected by the website documents themselves.

    HTML owns which downloadable and browser-loaded media ship. Deriving the
    copy set from those references prevents a second asset inventory and avoids
    publishing unrelated files merely because they share an asset directory.
    """

    source_dir = source_dir.resolve()
    referenced: set[str] = set()
    for relative_name in HTML_SOURCE_FILES:
        document_path = source_dir / relative_name
        collector = _ReferenceCollector()
        collector.feed(document_path.read_text(encoding="utf-8"))
        for _attribute, reference in collector.references:
            parsed = urlsplit(reference)
            if (
                parsed.scheme
                or reference.startswith("//")
                or parsed.path.startswith("/")
                or not parsed.path
            ):
                continue
            candidate = (document_path.parent / unquote(parsed.path)).resolve()
            if not candidate.is_relative_to(source_dir) or not candidate.is_file():
                continue
            referenced.add(candidate.relative_to(source_dir).as_posix())
    return tuple(sorted(referenced))


def _safe_output(repo_root: Path, output_dir: Path) -> Path:
    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    protected = {
        repo_root,
        (repo_root / "website").resolve(),
        (repo_root / "docs").resolve(),
        Path("/").resolve(),
    }
    if output_dir in protected:
        raise ValueError(f"Refusing to replace protected directory: {output_dir}")
    return output_dir


def validate_site(site_dir: Path) -> tuple[str, ...]:
    """Validate public pages, anchors, and local references in a staged site."""

    site_dir = site_dir.resolve()
    index_path = site_dir / "index.html"
    html = index_path.read_text(encoding="utf-8")
    missing_copy = [value for value in REQUIRED_COPY if value not in html]
    if missing_copy:
        raise ValueError(f"Landing page is missing required copy: {missing_copy}")

    errors: list[str] = []
    checked: list[str] = []
    collectors: dict[Path, _ReferenceCollector] = {}
    for relative_name in HTML_SOURCE_FILES:
        document_path = site_dir / relative_name
        if not document_path.is_file():
            errors.append(f"missing public page: {relative_name}")
            continue
        document = document_path.read_text(encoding="utf-8")
        for token in (
            RELEASE_VERSION_TOKEN,
            CONTACT_EMAIL_TOKEN,
            MCP_CLIENT_MARKS_TOKEN,
            GALLERY_CARDS_TOKEN,
            GALLERY_PROVENANCE_TOKEN,
        ):
            if token in document:
                errors.append(
                    f"unprojected website metadata token in {relative_name}: {token}"
                )
        collector = _ReferenceCollector()
        collector.feed(document)
        collectors[document_path.resolve()] = collector
        for duplicate_id in sorted(collector.duplicate_ids):
            errors.append(f"duplicate id in {relative_name}: {duplicate_id!r}")

    for document_path, collector in collectors.items():
        relative_document = document_path.relative_to(site_dir).as_posix()
        for attribute, reference in collector.references:
            parsed = urlsplit(reference)
            if parsed.scheme:
                if parsed.scheme not in ALLOWED_REMOTE_SCHEMES:
                    errors.append(
                        f"unsupported URL scheme in "
                        f"{relative_document} {attribute}={reference!r}"
                    )
                continue
            if reference.startswith("//") or parsed.path.startswith("/"):
                errors.append(
                    f"root-relative URL is not project-Pages-safe in "
                    f"{relative_document}: {reference!r}"
                )
                continue

            if parsed.path:
                relative_path = Path(unquote(parsed.path))
                resolved = (document_path.parent / relative_path).resolve()
            else:
                resolved = document_path
            if not resolved.is_relative_to(site_dir):
                errors.append(
                    f"local URL escapes staged site in "
                    f"{relative_document}: {reference!r}"
                )
                continue
            if not resolved.exists():
                errors.append(
                    f"missing local target in {relative_document} for "
                    f"{attribute}={reference!r}"
                )
                continue
            if parsed.path:
                checked.append(resolved.relative_to(site_dir).as_posix())
            if parsed.fragment:
                target_collector = collectors.get(resolved)
                if (
                    target_collector is None
                    or parsed.fragment not in target_collector.ids
                ):
                    errors.append(
                        f"missing local anchor in {relative_document}: {reference!r}"
                    )

    if errors:
        raise ValueError("Invalid website:\n- " + "\n- ".join(errors))
    return tuple(sorted(set(checked)))


def build_site(repo_root: Path, output_dir: Path) -> tuple[str, ...]:
    """Stage the website and its authoritative assets into ``output_dir``."""

    repo_root = repo_root.resolve()
    source_dir = repo_root / "website"
    gallery_record_path = source_dir / GALLERY_RECORD_RELATIVE_PATH
    gallery_projection = read_website_gallery_projection(gallery_record_path)
    output_dir = _safe_output(repo_root, output_dir)
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(
                f"Website output exists and is not a directory: {output_dir}"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    mcp_clients = read_mcp_client_targets(repo_root)
    source_files = tuple(
        dict.fromkeys(
            (
                *SOURCE_FILES,
                *referenced_source_files(source_dir),
                *(
                    f"assets/gallery/{path}"
                    for path in (
                        *gallery_projection.published_paths,
                        gallery_record_path.name,
                    )
                ),
                *(client.logo_path for client in mcp_clients),
            )
        )
    )
    for relative_name in source_files:
        source = source_dir / relative_name
        destination = output_dir / relative_name
        if not source.is_file():
            raise FileNotFoundError(f"Missing website source: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    for output_name, source_name in ASSET_SOURCES.items():
        source = repo_root / source_name
        destination = output_dir / output_name
        if not source.is_file():
            raise FileNotFoundError(f"Missing website asset authority: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    project_gallery_markup(output_dir / "index.html", gallery_projection)
    project_release_version(
        output_dir / "index.html",
        read_package_version(repo_root),
    )
    project_mcp_client_marks(output_dir / "index.html", mcp_clients)
    contact_email = read_package_contact_email(repo_root)
    for relative_name in HTML_SOURCE_FILES:
        if relative_name == "index.html":
            continue
        project_contact_email(output_dir / relative_name, contact_email)
    (output_dir / ".nojekyll").write_text("", encoding="utf-8")
    return validate_site(output_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="OpenHCS checkout root (default: inferred from this script)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("_site"),
        help="staging directory to replace (default: ./_site)",
    )
    args = parser.parse_args()
    checked = build_site(args.repo_root, args.output)
    print(f"Built validated website at {args.output.resolve()}")
    print("Resolved local targets: " + ", ".join(checked))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
