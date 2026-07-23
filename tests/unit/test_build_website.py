from __future__ import annotations

import re
from pathlib import Path

import pytest

from scripts.build_website import (
    RELEASE_VERSION_TOKEN,
    build_site,
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
        assert (site_dir / "assets/logos" / logo_name).read_bytes() == (
            REPO_ROOT / "website/assets/logos" / logo_name
        ).read_bytes()
    assert local_targets == (
        "assets/logos/bioformats.svg",
        "assets/logos/cellprofiler.png",
        "assets/logos/cupy.svg",
        "assets/logos/fiji.svg",
        "assets/logos/jax.png",
        "assets/logos/napari.svg",
        "assets/logos/pyclesperanto.png",
        "assets/logos/pytorch.svg",
        "assets/logos/tensorflow.svg",
        "assets/ui.png",
        "globals.css",
        "styles.css",
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
    assert f"OpenHCS {package_version} MCP extra" in html
    assert 'python -m pip install "openhcs[gui,mcp]"' in html
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
    assert 'class="install-routes"' in html
    assert html.index("Download for Windows") < html.index(
        'python -m pip install "openhcs[gui,viz,bioformats,mcp,cellprofiler-compat]"'
    )
    assert "This release adds supported CellProfiler" in html
    assert "Production MCPB signing" in html
    assert "hosted connector remain separate deployment work" in html
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


def test_website_source_and_workflow_follow_package_version_authority():
    source_html = (REPO_ROOT / "website/index.html").read_text(encoding="utf-8")
    workflow = (REPO_ROOT / ".github/workflows/website-pages.yml").read_text(
        encoding="utf-8"
    )

    assert RELEASE_VERSION_TOKEN in source_html
    assert "0.5.21" not in source_html
    assert "0.5.22" not in source_html
    assert workflow.count('      - "openhcs/__init__.py"') == 2


def test_readme_does_not_link_unpublished_coverage_site():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "trissim.github.io/openhcs/coverage" not in readme


def test_build_site_refuses_to_replace_source_or_repository_root():
    with pytest.raises(ValueError, match="protected directory"):
        build_site(REPO_ROOT, REPO_ROOT)
    with pytest.raises(ValueError, match="protected directory"):
        build_site(REPO_ROOT, REPO_ROOT / "website")
