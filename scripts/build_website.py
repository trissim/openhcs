#!/usr/bin/env python3
"""Build and validate the static OpenHCS landing-page artifact."""

from __future__ import annotations

import argparse
import ast
import html
import shutil
import tomllib
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

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
        for token in (RELEASE_VERSION_TOKEN, CONTACT_EMAIL_TOKEN):
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
    output_dir = _safe_output(repo_root, output_dir)
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(
                f"Website output exists and is not a directory: {output_dir}"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    source_files = tuple(
        dict.fromkeys((*SOURCE_FILES, *referenced_source_files(source_dir)))
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

    project_release_version(
        output_dir / "index.html",
        read_package_version(repo_root),
    )
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
