from __future__ import annotations

import re
from pathlib import Path

import pytest

from scripts.build_website import (
    CONTACT_EMAIL_TOKEN,
    RELEASE_VERSION_TOKEN,
    build_site,
    read_package_contact_email,
    read_package_version,
    validate_site,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_build_site_stages_authoritative_screenshot_and_valid_references(
    tmp_path: Path,
):
    site_dir = tmp_path / "site"

    local_targets = build_site(REPO_ROOT, site_dir)

    assert (site_dir / ".nojekyll").is_file()
    assert (site_dir / "assets/ui.png").read_bytes() == (
        REPO_ROOT / "docs/source/_static/ui.png"
    ).read_bytes()
    for logo_name in (
        "openhcs.svg",
        "bioformats.svg",
        "cellprofiler.png",
        "cupy.svg",
        "fiji.svg",
        "jax.png",
        "napari.svg",
        "pyclesperanto.png",
        "pytorch.svg",
        "tensorflow.svg",
    ):
        if logo_name == "openhcs.svg":
            authority = (
                REPO_ROOT / "openhcs/resources/assets/openhcs-mark.svg"
            )
        else:
            authority = REPO_ROOT / "website/assets/logos" / logo_name
        assert (site_dir / "assets/logos" / logo_name).read_bytes() == (
            authority.read_bytes()
        )
    assert local_targets == (
        "assets/logos/bioformats.svg",
        "assets/logos/cellprofiler.png",
        "assets/logos/cupy.svg",
        "assets/logos/fiji.svg",
        "assets/logos/jax.png",
        "assets/logos/napari.svg",
        "assets/logos/openhcs.svg",
        "assets/logos/pyclesperanto.png",
        "assets/logos/pytorch.svg",
        "assets/logos/tensorflow.svg",
        "assets/ui.png",
        "globals.css",
        "index.html",
        "privacy.html",
        "styles.css",
        "support.html",
        "terms.html",
    )
    assert validate_site(site_dir) == local_targets


def test_shipping_copy_projects_current_release_and_keeps_boundaries_explicit(
    tmp_path: Path,
):
    site_dir = tmp_path / "site"
    build_site(REPO_ROOT, site_dir)

    html = (site_dir / "index.html").read_text(encoding="utf-8")

    package_version = read_package_version(REPO_ROOT)
    assert RELEASE_VERSION_TOKEN not in html
    assert f"OpenHCS {package_version} on PyPI" in html
    assert f"Local MCP in OpenHCS {package_version}" in html
    assert "one-click local agent" in html
    assert "ChatGPT desktop app and Codex app/CLI/IDE" in html
    assert "ChatGPT" in html
    assert "ChatGPT web requires a remote HTTPS" in html
    assert "Secure MCP Tunnel" in html
    assert "shared ChatGPT desktop/Codex configuration" in html
    installer_assets = re.findall(
        r"https://github\.com/OpenHCSDev/OpenHCS/releases/latest/download/" r"([^\"]+)",
        html,
    )
    publish_workflow = (REPO_ROOT / ".github/workflows/publish.yml").read_text(
        encoding="utf-8"
    )
    assert len(installer_assets) == 2
    assert all(asset_name in publish_workflow for asset_name in installer_assets)
    assert installer_assets == [
        "OpenHCS-Windows-Installer.exe",
        "OpenHCS-macOS-Installer.dmg",
    ]
    assert all(not asset_name.endswith(".zip") for asset_name in installer_assets)
    assert "Download for Windows" in html
    assert "Download for macOS" in html
    assert "Download and run — no ZIP to extract" in html
    assert "Open the DMG, then open OpenHCS Installer" in html
    assert "Install-OpenHCS.cmd" not in html
    assert (
        "User-scoped, CPU-only installation with CellProfiler compatibility, local MCP"
        in html
    )
    assert "Napari, and Fiji/Bio-Formats support" in html
    assert "GPU libraries are not included" in html
    assert "Fiji Java components are resolved on first use" in html
    assert "not code-signed" in html and "not notarized" in html
    assert "Native installers follow the latest complete GitHub release" in html
    assert "trail the current PyPI package." in html
    assert f"PyPI {package_version}" in html
    assert 'class="install-routes"' in html
    assert html.index("Download for Windows") < html.index(
        'python -m pip install "openhcs[gui,viz,bioformats,mcp,cellprofiler-compat]"'
    )
    assert "The OpenHCS 0.6 beta includes supported CellProfiler" in html
    assert "Available in the 0.6 beta" in html
    assert f"New in {package_version}" not in html
    assert re.search(r"official registry\s+metadata", html)
    assert "https://openhcs.readthedocs.io/en/latest/" in html
    assert "https://openhcs.readthedocs.io/en/latest/api/" in html
    assert ">Install local MCP</a>" in html
    assert "https://github.com/OpenHCSDev/OpenHCS/releases" in html
    assert 'src="assets/ui.png"' in html
    assert "CellProfiler" in html and package_version in html
    assert "GPU libraries + custom functions" in html
    assert "Compute with" in html
    assert "Your functions" in html
    for backend_name in ("CuPy", "PyTorch", "JAX", "TensorFlow", "pyclesperanto"):
        assert backend_name in html
    assert "automatic memory conversion" in html
    assert "openhcs[gpu]" in html
    assert 'src="assets/logos/cellprofiler.png"' in html
    assert 'src="assets/logos/napari.svg"' in html
    assert 'src="assets/logos/fiji.svg"' in html
    assert 'src="assets/logos/bioformats.svg"' in html
    assert 'src="assets/logos/cupy.svg"' in html
    assert 'src="assets/logos/pytorch.svg"' in html
    assert 'src="assets/logos/jax.png"' in html
    assert 'src="assets/logos/tensorflow.svg"' in html
    assert 'src="assets/logos/pyclesperanto.png"' in html
    assert "https://cellprofiler.org/" in html
    assert "https://napari.org/" in html
    assert "https://imagej.net/software/fiji/" in html
    assert "https://www.openmicroscopy.org/bio-formats/" in html
    assert "https://cupy.dev/" in html
    assert "https://pytorch.org/" in html
    assert "https://docs.jax.dev/" in html
    assert "https://www.tensorflow.org/" in html
    assert "https://clesperanto.github.io/pyclesperanto/" in html
    assert "Bio-Formats image I/O" in html
    assert "openhcs[bioformats]" in html
    assert 'class="works-with"' not in html

    cppipe_row = html.split('class="capability-index">02', 1)[1].split(
        'class="capability-index">03', 1
    )[0]
    viewer_row = html.split('class="capability-index">03', 1)[1].split(
        'class="capability-index">04', 1
    )[0]
    bioformats_row = html.split('class="capability-index">05', 1)[1].split(
        "</article>", 1
    )[0]
    assert 'src="assets/logos/cellprofiler.png"' in cppipe_row
    assert 'src="assets/logos/napari.svg"' in viewer_row
    assert 'src="assets/logos/fiji.svg"' in viewer_row
    assert 'src="assets/logos/bioformats.svg"' in bioformats_row
    assert "./docs/" not in html
    assert "./coverage/" not in html


def test_landing_page_uses_factual_copy_and_readable_proportions():
    html = (REPO_ROOT / "website/index.html").read_text(encoding="utf-8")
    styles = (REPO_ROOT / "website/styles.css").read_text(encoding="utf-8")

    assert "OpenHCS defines and runs microscopy workflows." in html
    assert html.count('src="assets/logos/openhcs.svg"') == 2
    assert 'href="assets/logos/openhcs.svg"' in html
    assert '<span class="brand-mark" aria-hidden="true">H</span>' not in html
    assert 'class="hero-grid"' in html
    assert 'class="release-summary"' in html
    assert "Plate, pipeline, and result management." in html
    assert "Agent access to pipeline and runtime state." in html
    for removed_slogan in (
        "without the black box",
        "See the whole experiment",
        "Visual when you want it",
        "Give your agent the same mental model",
        "Install your way",
        "open by design",
    ):
        assert removed_slogan not in html

    assert "--max: 1280px;" in styles
    assert "font-size: 17px;" in styles
    assert "grid-template-columns: minmax(0, 1.25fr) minmax(20rem, 0.75fr);" in styles
    assert "font-size: clamp(3.4rem, 5.6vw, 5.4rem);" in styles
    assert ".capability-row p { color: var(--muted); font-size: 1rem; }" in styles
    assert (
        ".installer-boundary { padding: 1rem 1.25rem; color: var(--muted); "
        "font-size: 0.74rem;"
    ) in styles


def test_public_policy_pages_are_staged_with_truthful_hosted_boundaries(
    tmp_path: Path,
):
    site_dir = tmp_path / "site"
    build_site(REPO_ROOT, site_dir)

    contact_email = read_package_contact_email(REPO_ROOT)
    index_html = (site_dir / "index.html").read_text(encoding="utf-8")
    privacy_html = (site_dir / "privacy.html").read_text(encoding="utf-8")
    support_html = (site_dir / "support.html").read_text(encoding="utf-8")
    terms_html = (site_dir / "terms.html").read_text(encoding="utf-8")
    privacy_copy = " ".join(privacy_html.split())
    support_copy = " ".join(support_html.split())
    terms_copy = " ".join(terms_html.split())

    assert 'href="support.html"' in index_html
    assert 'href="privacy.html"' in index_html
    assert 'href="terms.html"' in index_html
    for document in (privacy_html, support_html, terms_html):
        assert CONTACT_EMAIL_TOKEN not in document
        assert f"mailto:{contact_email}" in document
        assert "OpenHCS contributors" in document
        assert 'href="privacy.html"' in document
        assert 'href="support.html"' in document
        assert 'href="terms.html"' in document
        assert document.count('src="assets/logos/openhcs.svg"') == 2
        assert 'href="assets/logos/openhcs.svg"' in document

    assert "does not currently operate a public hosted MCP endpoint" in privacy_copy
    assert "does not record bearer tokens or tool arguments" in privacy_copy
    assert "does not require an OpenHCS account or OAuth token" in privacy_copy
    assert "timestamp, authentication mode, declared capability name" in privacy_copy
    assert (
        "Public read-only events do not invent or record a tenant subject"
        in privacy_copy
    )
    assert "private deployment may enable OAuth token introspection" in privacy_copy
    assert "Google Fonts" in privacy_copy
    assert "Files, folders, images, or microscopy datasets" in privacy_copy
    assert "operator, provider list" in privacy_copy
    assert "retention" in privacy_copy

    assert "no public hosted OpenHCS endpoint is currently live" in support_copy
    assert "ChatGPT web cannot" in support_copy
    assert "local OpenHCS application or its STDIO MCP server" in support_copy
    assert "universal, unauthenticated, read-only discovery surface" in support_copy
    assert "private OAuth deployments are a separate operating mode" in support_copy
    assert "Planned ChatGPT web plugin" in support_copy
    assert "Local desktop MCP" in support_copy
    assert "patient identifiers" in support_copy

    assert "No public OpenHCS hosted MCP endpoint is currently operating" in terms_copy
    assert "universal, unauthenticated, read-only access" in terms_copy
    assert "private operator may separately require OAuth" in terms_copy
    assert (
        "not clinical, diagnostic, medical, legal, or regulatory advice" in terms_copy
    )
    assert "patient-identifiable data" in terms_copy
    assert "as is" in terms_copy and "as available" in terms_copy


def test_website_source_and_workflow_follow_package_metadata_authorities():
    source_html = (REPO_ROOT / "website/index.html").read_text(encoding="utf-8")
    workflow = (REPO_ROOT / ".github/workflows/website-pages.yml").read_text(
        encoding="utf-8"
    )

    assert RELEASE_VERSION_TOKEN in source_html
    assert "0.5.21" not in source_html
    assert "0.5.22" not in source_html
    assert workflow.count('      - "openhcs/__init__.py"') == 2
    assert (
        workflow.count(
            '      - "openhcs/resources/assets/openhcs-mark.svg"'
        )
        == 2
    )
    for page_name in ("privacy.html", "support.html", "terms.html"):
        page_source = (REPO_ROOT / "website" / page_name).read_text(encoding="utf-8")
        assert CONTACT_EMAIL_TOKEN in page_source
        assert read_package_contact_email(REPO_ROOT) not in page_source


def test_validation_checks_fragments_across_public_pages(tmp_path: Path):
    site_dir = tmp_path / "site"
    build_site(REPO_ROOT, site_dir)
    privacy_path = site_dir / "privacy.html"
    privacy_path.write_text(
        privacy_path.read_text(encoding="utf-8").replace(
            'href="support.html"',
            'href="support.html#missing-section"',
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing local anchor"):
        validate_site(site_dir)


def test_readme_does_not_link_unpublished_coverage_site():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "trissim.github.io/openhcs/coverage" not in readme
    assert 'src="openhcs/resources/assets/openhcs-mark.svg"' in readme


def test_build_site_refuses_to_replace_source_or_repository_root():
    with pytest.raises(ValueError, match="protected directory"):
        build_site(REPO_ROOT, REPO_ROOT)
    with pytest.raises(ValueError, match="protected directory"):
        build_site(REPO_ROOT, REPO_ROOT / "website")
