from __future__ import annotations

import hashlib
from html import escape
import json
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

from scripts.build_website import (
    ASSET_SOURCES,
    CONTACT_EMAIL_TOKEN,
    MCP_CLIENT_MARKS_TOKEN,
    RELEASE_VERSION_TOKEN,
    build_site,
    read_mcp_client_targets,
    read_package_contact_email,
    read_package_version,
    validate_site,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


class _GalleryMarkupCollector(HTMLParser):
    """Collect the semantic media surface without a browser dependency."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_gallery = False
        self.figures = 0
        self.figcaptions = 0
        self.images: list[dict[str, str | None]] = []
        self.videos: list[dict[str, str | None]] = []
        self.sources: list[dict[str, str | None]] = []
        self.links: list[dict[str, str | None]] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attributes = dict(attrs)
        if tag == "section" and attributes.get("id") == "gallery":
            self.in_gallery = True
        if not self.in_gallery:
            return
        if tag == "figure":
            self.figures += 1
        elif tag == "figcaption":
            self.figcaptions += 1
        elif tag == "img":
            self.images.append(attributes)
        elif tag == "video":
            self.videos.append(attributes)
        elif tag == "source":
            self.sources.append(attributes)
        elif tag == "a":
            self.links.append(attributes)

    def handle_endtag(self, tag: str) -> None:
        if self.in_gallery and tag == "section":
            self.in_gallery = False


def test_build_site_stages_authoritative_media_and_valid_references(
    tmp_path: Path,
):
    site_dir = tmp_path / "site"

    local_targets = build_site(REPO_ROOT, site_dir)

    assert (site_dir / ".nojekyll").is_file()
    gallery_sources = tuple(sorted((REPO_ROOT / "website/assets/gallery").iterdir()))
    for source in gallery_sources:
        relative_name = source.relative_to(REPO_ROOT / "website")
        assert (site_dir / relative_name).read_bytes() == source.read_bytes()
    agent_sources = tuple(sorted((REPO_ROOT / "website/assets/agent").iterdir()))
    for source in agent_sources:
        relative_name = source.relative_to(REPO_ROOT / "website")
        assert (site_dir / relative_name).read_bytes() == source.read_bytes()
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
        authority = REPO_ROOT / "website/assets/logos" / logo_name
        assert (site_dir / "assets/logos" / logo_name).read_bytes() == (
            authority.read_bytes()
        )
    mcp_clients = read_mcp_client_targets(REPO_ROOT)
    for client in mcp_clients:
        authority = REPO_ROOT / "website" / client.logo_path
        assert (site_dir / client.logo_path).read_bytes() == authority.read_bytes()
    for output_name, source_name in ASSET_SOURCES.items():
        assert (site_dir / output_name).read_bytes() == (
            REPO_ROOT / source_name
        ).read_bytes()
    expected_non_gallery_targets = tuple(
        sorted(
            {
                "assets/logos/bioformats.svg",
                "assets/logos/cellprofiler.png",
                "assets/logos/cupy.svg",
                "assets/logos/fiji.svg",
                "assets/logos/jax.png",
                "assets/logos/napari.svg",
                "assets/logos/openhcs-favicon.svg",
                "assets/logos/openhcs-horizontal.svg",
                "assets/logos/openhcs-stacked.svg",
                "assets/logos/platform-macos.svg",
                "assets/logos/platform-windows.svg",
                "assets/logos/pyclesperanto.png",
                "assets/logos/python.svg",
                "assets/logos/pytorch.svg",
                "assets/logos/tensorflow.svg",
                "globals.css",
                "index.html",
                "privacy.html",
                "styles.css",
                "support.html",
                "terms.html",
                *(
                    str(source.relative_to(REPO_ROOT / "website"))
                    for source in agent_sources
                ),
                *(client.logo_path for client in mcp_clients),
            }
        )
    )
    assert (
        tuple(
            target
            for target in local_targets
            if not target.startswith("assets/gallery/")
        )
        == expected_non_gallery_targets
    )
    collector = _GalleryMarkupCollector()
    collector.feed((site_dir / "index.html").read_text(encoding="utf-8"))
    gallery_references = {
        attributes["src"] for attributes in (*collector.images, *collector.sources)
    }
    gallery_references.add(collector.videos[0]["poster"])
    gallery_references.update(
        link["href"]
        for link in collector.links
        if link.get("href", "").startswith("assets/gallery/")
    )
    assert {
        target for target in local_targets if target.startswith("assets/gallery/")
    } == gallery_references
    assert validate_site(site_dir) == local_targets


def test_shipping_copy_projects_current_release_and_keeps_boundaries_explicit(
    tmp_path: Path,
):
    site_dir = tmp_path / "site"
    build_site(REPO_ROOT, site_dir)

    html = (site_dir / "index.html").read_text(encoding="utf-8")
    normalized_html = " ".join(html.split())

    package_version = read_package_version(REPO_ROOT)
    assert RELEASE_VERSION_TOKEN not in html
    assert f"Install OpenHCS {package_version}" in html
    assert "Desktop installers include the GUI and local MCP setup" in html
    assert "ChatGPT" in html
    assert 'href="support.html#plugin-status"' in html
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
        "User-scoped, CPU-only installers include CellProfiler compatibility"
        in normalized_html
    )
    assert "local MCP, Napari, Fiji, and Bio-Formats" in normalized_html
    assert "GPU libraries are optional and not included" in normalized_html
    assert "Fiji downloads Java on first use" in normalized_html
    assert (
        "The Windows installer is unsigned, and the macOS installer is not notarized"
        in normalized_html
    )
    assert "latest complete GitHub release may trail PyPI" in normalized_html
    assert f"PyPI {package_version}" in html
    assert 'class="install-routes"' in html
    assert html.index("Download for Windows") < html.index(
        'python -m pip install "openhcs[gui,viz,bioformats,mcp,cellprofiler-compat]"'
    )
    assert "The current beta also imports supported" in html
    assert "Available in the current beta" not in html
    assert "0.6 beta" not in html
    assert f"New in {package_version}" not in html
    assert "https://openhcs.readthedocs.io/en/latest/" in html
    assert "https://openhcs.readthedocs.io/en/latest/api/" in html
    assert ">Install local MCP</a>" in html
    assert "https://github.com/OpenHCSDev/OpenHCS/releases" in html
    assert 'id="gallery"' in html
    assert 'src="assets/gallery/' in html
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
    assert "https://cellprofiler.org/citations" in html
    assert "https://doi.org/10.1186/s12859-021-04344-9" in html
    assert "We thank its authors and contributors" in html
    assert "OpenHCS is independent of CellProfiler" in html
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
    assert 'src="assets/logos/platform-windows.svg"' in html
    assert 'src="assets/logos/platform-macos.svg"' in html
    assert 'src="assets/logos/python.svg"' in html

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


def test_mcp_client_marks_project_from_registration_authority(tmp_path: Path):
    source_html = (REPO_ROOT / "website/index.html").read_text(encoding="utf-8")
    clients = read_mcp_client_targets(REPO_ROOT)

    assert source_html.count(MCP_CLIENT_MARKS_TOKEN) == 1
    assert 'class="client-mark"' not in source_html
    assert "illustrative local MCP session" not in source_html
    assert 'class="agent-card"' not in source_html

    site_dir = tmp_path / "site"
    build_site(REPO_ROOT, site_dir)
    html = (site_dir / "index.html").read_text(encoding="utf-8")
    client_section = html.split('<ul class="client-marks"', 1)[1].split("</ul>", 1)[0]

    assert MCP_CLIENT_MARKS_TOKEN not in html
    assert client_section.count('class="client-mark"') == len(clients)
    for client in clients:
        assert f'src="{client.logo_path}"' in client_section
        assert f"<span>{escape(client.display_name)}</span>" in client_section
        assert (site_dir / client.logo_path).is_file()


def test_landing_page_uses_factual_copy_and_readable_proportions():
    html = (REPO_ROOT / "website/index.html").read_text(encoding="utf-8")
    normalized_html = " ".join(html.split())
    styles = (REPO_ROOT / "website/styles.css").read_text(encoding="utf-8")

    assert "From images and a question to a validated workflow." in html
    assert html.count('src="assets/logos/openhcs-horizontal.svg"') == 1
    assert html.count('src="assets/logos/openhcs-stacked.svg"') == 1
    assert 'href="assets/logos/openhcs-favicon.svg"' in html
    assert "<span>OpenHCS</span>" not in html
    assert '<span class="brand-mark" aria-hidden="true">H</span>' not in html
    assert 'class="hero-grid"' in html
    assert 'class="release-summary"' not in html
    assert "execution-progress.webm" not in html
    assert "result-review.webm" not in html
    assert "OpenHCS in use" in html
    assert "Use OpenHCS through a local agent" in html
    assert "without manually constructing the pipeline" in normalized_html
    assert "Every pipeline remains editable in" in normalized_html
    assert "the GUI and as generated Python" in normalized_html
    assert "Auto-configured clients" in html
    assert "These marks describe local setup compatibility" in normalized_html
    assert "exact client, model, and version tested" in normalized_html
    assert 'class="agent-evidence"' in html
    assert 'id="agent-evidence-title"' in html
    assert 'id="agent-workflow-showcase"' in html
    assert 'id="agent-workflow-evidence"' in html
    assert "cold-start-workflow.mp4" in html
    assert "cold-start-workflow-uncut.mp4" in html
    assert "cold-start-workflow-transcript.txt" in html
    assert "cold-start-workflow-final.md" in html
    assert "cold-start-workflow-events.jsonl" in html
    assert "cold-start-workflow-record.json" in html
    assert "cold-start-workflow-pipeline.py" in html
    assert "One prompt; no later human steering" in html
    assert "no shell or repository access" in normalized_html
    assert "NeuronCyto II" in html
    assert "per-neuron morphology analysis" in normalized_html
    assert "24 spatial-graph paths" in normalized_html
    assert "Public neuronal images to inspectable morphology in 0:43" in html
    assert "result interaction was recaptured after visual QA" in normalized_html
    assert "Uncut 10:47 run" in html
    assert "973c51fd0" in html
    assert "0eb5f77c0" in html
    assert "Recordings are being prepared." not in html
    assert "without supervision" not in html
    assert "Open High-Content Screening" in html
    assert "Open-source high-content image analysis" not in html
    assert "Open High-Content Image Analysis" not in html
    assert "Inspect results in Napari" in html
    assert "Inspect results in Fiji" in html
    assert "View images, ROIs, and measurements" in html
    assert "Pipeline review" not in html
    assert "interactive review" not in html
    assert 'class="release-facts"' not in html
    assert 'class="mcp-points"' not in html
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
    assert "grid-template-columns: minmax(0, 1fr);" in styles
    assert "font-size: clamp(3.25rem, 4.8vw, 4.65rem);" in styles
    assert ".hero-intro h1 { max-width: none; }" in styles
    assert ".capability-row p { color: var(--muted); font-size: 1rem; }" in styles
    assert (
        ".installer-boundary { padding: 1rem 1.25rem; color: var(--muted); "
        "font-size: 0.74rem;"
    ) in styles


def test_agent_workflow_evidence_record_matches_published_assets():
    asset_root = REPO_ROOT / "website/assets/agent"
    record = json.loads(
        (asset_root / "cold-start-workflow-record.json").read_text(encoding="utf-8")
    )

    assert record["schema_version"] == "openhcs.agent-workflow-validation.v1"
    assert record["run"]["verdict"] == "passed"
    assert record["run"]["operation_mode"] == "unattended"
    assert all(record["acceptance"].values())
    assert record["trace"]["non_mcp_calls"] == []
    assert record["trace"]["human_interventions"] == []
    assert record["fixture"]["kind"].startswith("public NeuronCyto II")
    assert record["evidence"]["result_summary"]["neurons"] == 9
    assert record["evidence"]["result_summary"]["spatial_graph_path_count"] == 25
    corrected = record["evidence"]["post_qa_corrected_recapture"]
    assert corrected["release_fix_commit"].startswith("0eb5f77c0")
    assert corrected["result_summary"]["neurons"] == 9
    assert corrected["result_summary"]["spatial_graph_path_count"] == 24
    assert corrected["result_summary"]["resolved_crossovers"] == 1
    assert record["evidence"]["output_inventory"]["swc_count"] == 1
    assert record["evidence"]["viewer"]["nonzero_payloads"] == 9
    assert record["evidence"]["post_run_finding"]["release_fix_commit"].startswith(
        "973c51fd0"
    )

    published_artifacts = {
        record["trace"]["event_log_path"]: record["trace"]["event_log_sha256"],
        record["trace"]["transcript_path"]: record["trace"]["transcript_sha256"],
        record["trace"]["final_response_path"]: record["trace"][
            "final_response_sha256"
        ],
        record["evidence"]["pipeline_source_path"]: record["evidence"][
            "pipeline_source_sha256"
        ],
        record["evidence"]["media"]["edited_video_path"]: record["evidence"]["media"][
            "edited_video_sha256"
        ],
        record["evidence"]["media"]["uncut_video_path"]: record["evidence"]["media"][
            "uncut_video_sha256"
        ],
        record["evidence"]["media"]["poster_path"]: record["evidence"]["media"][
            "poster_sha256"
        ],
    }
    for relative_path, expected_hash in published_artifacts.items():
        artifact_path = asset_root / relative_path
        assert artifact_path.is_file()
        with artifact_path.open("rb") as artifact:
            assert hashlib.file_digest(artifact, "sha256").hexdigest() == expected_hash

    events = [
        json.loads(line)
        for line in (asset_root / record["trace"]["event_log_path"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    completed_calls = [
        event["item"]
        for event in events
        if event.get("type") == "item.completed"
        and event.get("item", {}).get("type") == "mcp_tool_call"
    ]
    failed_call_ids = {
        call["id"]
        for call in completed_calls
        if call.get("status") != "completed" or call.get("error")
    }
    result_error_call_ids = {
        call["id"]
        for call in completed_calls
        if isinstance(
            (call.get("result") or {}).get("structured_content"),
            dict,
        )
        and (call.get("result") or {})["structured_content"].get("errors")
    }
    assert record["trace"]["ordered_mcp_call_count"] == len(completed_calls)
    assert record["trace"]["completed_mcp_call_count"] == (
        len(completed_calls) - len(failed_call_ids)
    )
    assert record["trace"]["failed_mcp_call_count"] == len(failed_call_ids)
    assert record["trace"]["result_error_count"] == len(result_error_call_ids)
    assert set(record["trace"]["failed_or_error_call_ids"]) == (
        failed_call_ids | result_error_call_ids
    )


def test_public_pages_use_the_project_name_expansion():
    for page_name in ("index.html", "privacy.html", "support.html", "terms.html"):
        document = (REPO_ROOT / "website" / page_name).read_text(encoding="utf-8")
        assert "Open High-Content Screening" in document
        assert "Open High-Content Image Analysis" not in document


def test_gallery_uses_semantic_accessible_media_and_stable_paths():
    html = (REPO_ROOT / "website/index.html").read_text(encoding="utf-8")
    collector = _GalleryMarkupCollector()
    collector.feed(html)

    assert 'href="#gallery"' in html
    assert 'aria-labelledby="gallery-title"' in html
    assert "The 12-step Comet Assay uses OpenHCS compilation" in html
    assert "time-lapse" not in html
    assert "five-phase compilation" not in html
    assert "OpenHCS sends the image, ROIs, and linked measurements" in html
    assert "BSD-3-Clause" in html
    assert (
        "https://github.com/CellProfiler/examples/tree/"
        "4972b59e670a4ae96c3d453803c92eeff378d054" in html
    )
    assert collector.figures == 6
    assert collector.figcaptions == collector.figures
    assert len(collector.images) == 6
    for image in collector.images:
        assert image["src"].startswith("assets/gallery/")
        assert image["src"].endswith(".webp")
        assert image.get("alt", "").strip()
        assert image.get("loading") == "lazy"
        assert image.get("decoding") == "async"
        assert image.get("width", "").isdigit()
        assert image.get("height", "").isdigit()

    motion_stems = ("lazy-inheritance",)
    assert len(collector.videos) == len(motion_stems)
    for video in collector.videos:
        for boolean_attribute in ("controls", "muted", "loop", "playsinline"):
            assert boolean_attribute in video
        assert "autoplay" not in video
        assert video["preload"] == "metadata"
        assert video["poster"].startswith("assets/gallery/")
        assert video["poster"].endswith("-poster.webp")
        assert video.get("aria-describedby", "").strip()
    assert {
        Path(video["poster"]).stem.removesuffix("-poster") for video in collector.videos
    } == set(motion_stems)
    assert {Path(source["src"]).stem for source in collector.sources} == set(
        motion_stems
    )
    assert [source["type"] for source in collector.sources] == [
        media_type for _ in motion_stems for media_type in ("video/webm", "video/mp4")
    ]

    full_resolution_targets = {
        link["href"]
        for link in collector.links
        if link.get("class") and "gallery-media-link" in link["class"].split()
    }
    assert full_resolution_targets == {image["src"] for image in collector.images}
    assert 'href="assets/gallery/lazy-inheritance.gif"' in html
    for link in collector.links:
        if link.get("class") and "gallery-media-link" in link["class"].split():
            assert link.get("aria-label", "").strip()


def test_gallery_layout_is_responsive_and_has_reduced_motion_fallback():
    styles = (REPO_ROOT / "website/styles.css").read_text(encoding="utf-8")

    assert "grid-template-columns: repeat(12, minmax(0, 1fr));" in styles
    assert "align-items: start;" in styles
    gallery_card_rule = styles.split(".gallery-card {", 1)[1].split("}", 1)[0]
    assert "margin: 0;" in gallery_card_rule
    assert ".gallery-card-wide { grid-column: span 7; }" in styles
    assert ".gallery-card-compact { grid-column: span 5; }" in styles
    assert ".gallery-card-viewer { grid-column: span 6; }" in styles
    assert "@media (max-width: 900px)" in styles
    assert (
        ".gallery-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }" in styles
    )
    assert "@media (max-width: 600px)" in styles
    assert ".gallery-grid { grid-template-columns: 1fr;" in styles
    reduced_motion = styles.split("@media (prefers-reduced-motion: reduce)", 1)[1]
    assert ".gallery-motion video { display: none; }" in reduced_motion
    assert ".gallery-motion-fallback { display: block;" in reduced_motion


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
        assert document.count('src="assets/logos/openhcs-horizontal.svg"') == 1
        assert document.count('src="assets/logos/openhcs-stacked.svg"') == 1
        assert 'href="assets/logos/openhcs-favicon.svg"' in document

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
    assert workflow.count('      - "openhcs/resources/assets/openhcs-*.svg"') == 2
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


def test_validation_checks_video_poster_references(tmp_path: Path):
    site_dir = tmp_path / "site"
    build_site(REPO_ROOT, site_dir)
    index_path = site_dir / "index.html"
    index_path.write_text(
        index_path.read_text(encoding="utf-8").replace(
            'poster="assets/gallery/lazy-inheritance-poster.webp"',
            'poster="assets/gallery/missing-poster.webp"',
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"poster=.*missing-poster"):
        validate_site(site_dir)


def test_readme_does_not_link_unpublished_coverage_site():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "trissim.github.io/openhcs/coverage" not in readme
    assert 'src="openhcs/resources/assets/openhcs-icon-square.svg"' in readme
    assert "<h1>OpenHCS</h1>" in readme


def test_build_site_refuses_to_replace_source_or_repository_root():
    with pytest.raises(ValueError, match="protected directory"):
        build_site(REPO_ROOT, REPO_ROOT)
    with pytest.raises(ValueError, match="protected directory"):
        build_site(REPO_ROOT, REPO_ROOT / "website")
