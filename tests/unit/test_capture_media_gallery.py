from __future__ import annotations

import json
import shutil
import subprocess
from hashlib import sha256
from pathlib import Path

import pytest
from pyqt_reactive.animation import WindowFlashOverlay
from pyqt_reactive.animation.flash_overlay_opengl import WindowFlashOverlayGL
from pyqt_reactive.services.window_snapshot import WindowSnapshotCaptureScope

from openhcs.agent.dto.common import SCHEMA_VERSION, AgentResourceRef
from openhcs.agent.dto.ui_bridge import UiWindowSnapshotResult
from scripts.capture_media_gallery import (
    CaptureManifest,
    CaptureRecord,
    CaptureToolDoctorReport,
    CaptureValidationResult,
    Crop,
    Derivative,
    DerivativeValidationResult,
    GalleryUiBridgeSession,
    HostTool,
    HostToolDeclaration,
    HostToolStatus,
    MediaGalleryError,
    MediaProbe,
    Trim,
    WindowGeometry,
    _capture_target,
    _write_json,
    build_manifest,
    build_transcode_command,
    capture_scenario_still,
    capture_ui_bridge_window_source,
    capture_ui_window_reference_stills,
    capture_window_still,
    doctor,
    load_manifest,
    plan_manifest,
    read_window_geometry,
    record_window,
    resolve_contained_path,
    validate_derivative,
    validate_manifest_outputs,
)
from scripts.gallery_catalog import (
    FunctionSelectorCaptureTarget,
    GallerySourceCaptureRequest,
    GallerySourceCaptureResult,
    ObjectStateCaptureScopeRole,
    PlateManagerActionWindowCaptureTarget,
    SystemMonitorActionWindowCaptureTarget,
    UiBridgeWindowCaptureTarget,
    read_gallery_source_evidence,
    ui_context_reference_gallery_scenarios,
    ui_window_reference_gallery_scenarios,
)


def _still_output(filename: str = "multi-plate-overview.webp") -> Derivative:
    return Derivative(
        filename=filename,
        max_width=1600,
        max_height=1000,
        max_bytes=2_000_000,
    )


def _motion_record(*outputs: Derivative) -> CaptureRecord:
    return CaptureRecord(
        source=Path("raw/session.mkv"),
        scenario_id="lazy-inheritance",
        crop=Crop(x=4, y=6, width=1200, height=760),
        trim=Trim(start_seconds=1.25, duration_seconds=8.0),
        outputs=outputs,
    )


def test_cli_json_projection_descends_from_typed_result(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = DerivativeValidationResult(
        path="example.webp",
        size_bytes=42,
        sha256="a" * 64,
        width=1200,
        height=800,
        codec="webp",
        duration_seconds=None,
    )

    _write_json(result)

    assert json.loads(capsys.readouterr().out) == {
        "codec": "webp",
        "duration_seconds": None,
        "height": 800,
        "path": "example.webp",
        "sha256": "a" * 64,
        "size_bytes": 42,
        "width": 1200,
    }


def test_doctor_descends_from_host_tool_declarations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def inspect(declaration: HostToolDeclaration) -> HostToolStatus:
        return HostToolStatus(
            executable=declaration.executable,
            available=True,
            path=f"/tools/{declaration.executable}",
            version="test version",
        )

    monkeypatch.setattr(HostToolDeclaration, "inspect", inspect)

    report = doctor()

    assert isinstance(report, CaptureToolDoctorReport)
    assert report.schema_version == 2
    assert tuple(status.executable for status in report.tools) == tuple(
        tool.value.executable for tool in HostTool
    )


def test_declared_ui_still_uses_single_mcp_capture_leaf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = []
    expected = GallerySourceCaptureResult(
        path="raw/workspace.png",
        sha256="a" * 64,
        width=1600,
        height=958,
    )

    def capture(target, request):
        observed.append((target, request))
        return expected

    monkeypatch.setattr(
        "scripts.capture_media_gallery.capture_ui_bridge_window_source",
        capture,
    )

    result = capture_scenario_still(
        tmp_path,
        Path("raw/workspace.png"),
        "multi-plate-overview",
    )

    assert result is expected
    assert observed[0][0].window_id == "main_window"
    assert observed[0][1].output == Path("raw/workspace.png")


def test_contextual_ui_scenarios_derive_typed_capture_targets() -> None:
    scenarios = ui_context_reference_gallery_scenarios()

    assert len(scenarios) == 8
    assert any(
        isinstance(scenario.capture_target, FunctionSelectorCaptureTarget)
        for scenario in scenarios
    )
    assert any(
        isinstance(
            scenario.capture_target,
            PlateManagerActionWindowCaptureTarget,
        )
        for scenario in scenarios
    )
    assert any(
        isinstance(
            scenario.capture_target,
            SystemMonitorActionWindowCaptureTarget,
        )
        for scenario in scenarios
    )


def test_contextual_object_state_roles_resolve_live_scope_structure() -> None:
    plate_scope = "/tmp/test-plate"
    step_scope = f"{plate_scope}::functionstep_0"
    function_scope = f"{step_scope}::func_0"
    scope_ids = (plate_scope, step_scope, function_scope)

    assert ObjectStateCaptureScopeRole.PIPELINE.resolve(scope_ids) == plate_scope
    assert ObjectStateCaptureScopeRole.STEP.resolve(scope_ids) == step_scope
    assert ObjectStateCaptureScopeRole.FUNCTION.resolve(scope_ids) == function_scope


def test_contextual_capture_waits_for_declared_flash_type_lifecycle() -> None:
    session = object.__new__(GalleryUiBridgeSession)
    observed = iter(
        (
            frozenset({WindowFlashOverlay.__name__}),
            frozenset({WindowFlashOverlayGL.__name__}),
            frozenset(),
        )
    )
    session.visible_widget_class_names = lambda _window_id: next(observed)
    session.control_timeout_ms = 1

    session.wait_for_flash_feedback("scope")


def test_still_capture_rejects_motion_and_human_review_scenarios(
    tmp_path: Path,
) -> None:
    with pytest.raises(MediaGalleryError, match="bounded recording workflow"):
        capture_scenario_still(
            tmp_path,
            Path("raw/lazy.png"),
            "lazy-inheritance",
        )
    with pytest.raises(MediaGalleryError, match="fiji viewer window"):
        capture_scenario_still(
            tmp_path,
            Path("raw/fiji.png"),
            "fiji-review",
        )


def test_ui_bridge_snapshot_capture_uses_declared_request_and_result_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "captures"
    snapshot_path = source_root / "raw" / "mcp-snapshot.png"
    snapshot_path.parent.mkdir(parents=True)
    snapshot_bytes = b"lossless snapshot bytes"
    snapshot_path.write_bytes(snapshot_bytes)
    expected_sha256 = sha256(snapshot_bytes).hexdigest()
    descriptor_path = tmp_path / "ui-bridge.json"
    observed_connections = []
    observed_requests = []

    class FakeUiBridgeService:
        def __init__(self) -> None:
            pass

        def connection_from_args(self, **kwargs):
            observed_connections.append(kwargs)
            return kwargs

        def snapshot_window(self, request, connection):
            observed_requests.append(request)
            assert connection is observed_connections[-1]
            return UiWindowSnapshotResult(
                schema_version=SCHEMA_VERSION,
                window_id="main_window",
                output_dir_path=str(snapshot_path.parent),
                capture_scope=WindowSnapshotCaptureScope.WIDGET,
                captured=True,
                resource=AgentResourceRef(
                    uri=snapshot_path.as_uri(),
                    title="OpenHCS window snapshot",
                    mime_type="image/png",
                    path=str(snapshot_path),
                    size_bytes=len(snapshot_bytes),
                    sha256=expected_sha256,
                ),
                width=942,
                height=900,
            )

    monkeypatch.setattr(
        "scripts.capture_media_gallery.UiBridgeService",
        FakeUiBridgeService,
    )

    result = capture_ui_bridge_window_source(
        UiBridgeWindowCaptureTarget(
            window_id="main_window",
            create_if_missing=True,
        ),
        GallerySourceCaptureRequest(
            source_root=source_root,
            output=Path("raw/workspace.png"),
            descriptor_file_path=descriptor_path,
        ),
    )

    assert observed_requests[0].window_id == "main_window"
    assert observed_requests[0].open_policy.create_if_missing is True
    assert observed_connections[0]["descriptor_file_path"] == str(descriptor_path)
    assert result.path == "raw/workspace.png"
    assert result.sha256 == expected_sha256
    assert (result.width, result.height) == (942, 900)
    assert (source_root / result.path).read_bytes() == snapshot_bytes


def test_ui_window_reference_batch_derives_every_capture_and_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "sources"
    output_root = tmp_path / "gallery"
    output_root.mkdir()
    scenarios = ui_window_reference_gallery_scenarios()[:2]
    captured_requests: list[tuple[str, GallerySourceCaptureRequest]] = []

    def capture_source(scenario, request: GallerySourceCaptureRequest):
        captured_requests.append((scenario.scenario_id, request))
        target = request.source_root / request.output
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(scenario.scenario_id.encode())
        return GallerySourceCaptureResult(
            path=str(request.output),
            sha256=sha256(target.read_bytes()).hexdigest(),
            width=800,
            height=600,
            format="png",
        )

    def build(manifest, actual_source_root, actual_output_root, *, force):
        assert actual_source_root == source_root
        assert actual_output_root == output_root
        assert force is True
        return tuple(
            CaptureValidationResult(
                scenario_id=record.scenario_id,
                source=GallerySourceCaptureResult(
                    path=str(record.source),
                    sha256=sha256(
                        (source_root / record.source).read_bytes()
                    ).hexdigest(),
                    width=800,
                    height=600,
                    format="png",
                ),
                outputs=(),
            )
            for record in manifest.captures
        )

    synchronized: list[tuple[Path, bool]] = []
    monkeypatch.setattr(
        "scripts.capture_media_gallery.gallery_scenarios",
        lambda: scenarios,
    )
    monkeypatch.setattr(
        "scripts.gallery_catalog.UiWindowReferenceGalleryScenario.capture_source",
        capture_source,
    )
    monkeypatch.setattr("scripts.capture_media_gallery.build_manifest", build)
    monkeypatch.setattr(
        "scripts.capture_media_gallery.synchronize_gallery_release_record_for_asset_root",
        lambda path, *, check: synchronized.append((path, check)),
    )

    result = capture_ui_window_reference_stills(
        source_root,
        output_root,
        descriptor_file_path=tmp_path / "bridge.json",
        timeout_ms=12_000,
        force=True,
    )

    assert tuple(scenario_id for scenario_id, _request in captured_requests) == tuple(
        scenario.scenario_id for scenario in scenarios
    )
    assert tuple(
        request.output for _scenario_id, request in captured_requests
    ) == tuple(Path("raw") / f"{scenario.scenario_id}.png" for scenario in scenarios)
    assert all(request.timeout_ms == 12_000 for _, request in captured_requests)
    assert synchronized == [(output_root, False)]
    evidence = read_gallery_source_evidence(output_root)
    assert tuple(capture.scenario_id for capture in result.captures) == tuple(
        scenario.scenario_id for scenario in scenarios
    )
    assert Path(result.manifest_path).is_file()
    assert Path(result.evidence_path).is_file()
    assert len(evidence.captures) == len(scenarios)


def test_load_manifest_builds_typed_authority_and_rejects_unknown_fields(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "captures": [
                    {
                        "scenario_id": "multi-plate-overview",
                        "source": "raw/interface.png",
                        "crop": {"x": 10, "y": 12, "width": 800, "height": 500},
                        "outputs": [
                            {
                                "filename": "multi-plate-overview.webp",
                                "max_width": 800,
                                "max_height": 500,
                                "max_bytes": 500000,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = load_manifest(manifest_path)

    assert manifest.schema_version == 2
    assert manifest.captures[0].crop == Crop(10, 12, 800, 500)
    assert manifest.captures[0].outputs[0].filename == "multi-plate-overview.webp"

    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["captures"][0]["caption"] = "This belongs to the website."
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(MediaGalleryError, match="unknown fields: caption"):
        load_manifest(manifest_path)


@pytest.mark.parametrize(
    "source",
    (
        Path("../private.png"),
        Path("/tmp/private.png"),
        Path("."),
    ),
)
def test_capture_source_must_be_contained_relative_path(source: Path) -> None:
    with pytest.raises(MediaGalleryError, match="contained relative path"):
        CaptureRecord(
            source=source,
            scenario_id="multi-plate-overview",
            outputs=(_still_output(),),
        )


def test_containment_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)

    with pytest.raises(MediaGalleryError, match="escapes its declared root"):
        resolve_contained_path(
            root,
            Path("escape/private.png"),
            context="Capture source",
        )


@pytest.mark.parametrize(
    "filename",
    (
        "Has Spaces.webp",
        "camelCase.webp",
        "../escape.webp",
        "double--hyphen.webp",
        "unsupported.png",
    ),
)
def test_output_names_are_caption_safe_and_format_owned(filename: str) -> None:
    with pytest.raises(MediaGalleryError):
        _still_output(filename)


def test_motion_manifest_requires_bounded_trim_and_typed_output_fields() -> None:
    motion = Derivative(
        filename="lazy-update.webm",
        max_width=1280,
        max_height=800,
        max_bytes=3_000_000,
        fps=24,
    )
    with pytest.raises(MediaGalleryError, match="explicit bounded trim"):
        CaptureRecord(
            source=Path("raw/session.mkv"),
            scenario_id="lazy-inheritance",
            outputs=(motion,),
        )

    with pytest.raises(MediaGalleryError, match="must declare fps"):
        _motion_record(
            Derivative(
                filename="lazy-update.mp4",
                max_width=1280,
                max_height=800,
                max_bytes=3_000_000,
            )
        )

    with pytest.raises(MediaGalleryError, match="frame_at_seconds"):
        _motion_record(_still_output("lazy-update-poster.webp"))


def test_plan_is_a_write_free_exact_command_projection(tmp_path: Path) -> None:
    webm = Derivative(
        filename="lazy-inheritance.webm",
        max_width=1280,
        max_height=800,
        max_bytes=3_000_000,
        fps=24,
    )
    poster = Derivative(
        filename="lazy-inheritance-poster.webp",
        max_width=1280,
        max_height=800,
        max_bytes=1_000_000,
        frame_at_seconds=2.5,
    )
    mp4 = Derivative(
        filename="lazy-inheritance.mp4",
        max_width=1280,
        max_height=800,
        max_bytes=3_000_000,
        fps=24,
    )
    manifest = CaptureManifest(captures=(_motion_record(poster, webm, mp4),))
    source_root = tmp_path / "sources"
    output_root = tmp_path / "outputs"

    plan = plan_manifest(manifest, source_root, output_root)

    assert not source_root.exists()
    assert not output_root.exists()
    poster_command, webm_command, _mp4_command = plan[0].commands
    assert "-ss" in poster_command
    assert poster_command[poster_command.index("-ss") + 1] == "3.75"
    assert "-t" not in poster_command
    assert "-t" in webm_command
    assert webm_command[webm_command.index("-t") + 1] == "8"
    poster_filters = poster_command[poster_command.index("-vf") + 1]
    assert "crop=1200:760:4:6" in poster_filters
    assert "force_original_aspect_ratio=decrease" in poster_filters
    assert "fps=" not in poster_filters
    webm_filters = webm_command[webm_command.index("-vf") + 1]
    assert "fps=24" in webm_filters


def test_gif_encoding_owns_palette_filter_and_output_mapping() -> None:
    output = Derivative(
        filename="lazy-update.gif",
        max_width=960,
        max_height=600,
        max_bytes=8_000_000,
        fps=12,
    )
    record = _motion_record(output)

    command = build_transcode_command(
        record,
        output,
        Path("/captures/session.mkv"),
        Path("/gallery/lazy-update.gif"),
        ffmpeg_executable="ffmpeg",
    )

    assert "-filter_complex" in command
    filter_graph = command[command.index("-filter_complex") + 1]
    assert "palettegen=max_colors=192" in filter_graph
    assert filter_graph.endswith("[gallery_output]")
    assert command[command.index("-map") + 1] == "[gallery_output]"


def test_manifest_refuses_to_overwrite_source_capture(tmp_path: Path) -> None:
    record = CaptureRecord(
        source=Path("multi-plate-overview.webp"),
        scenario_id="multi-plate-overview",
        outputs=(_still_output(),),
    )
    manifest = CaptureManifest(captures=(record,))

    with pytest.raises(MediaGalleryError, match="overwrite source capture"):
        plan_manifest(manifest, tmp_path, tmp_path)


def test_raw_capture_never_overwrites_existing_source(tmp_path: Path) -> None:
    target = tmp_path / "raw" / "session.png"
    target.parent.mkdir()
    target.write_bytes(b"original")

    with pytest.raises(MediaGalleryError, match="Refusing to overwrite source"):
        _capture_target(tmp_path, Path("raw/session.png"), ".png")
    assert target.read_bytes() == b"original"


def test_capture_still_projects_real_window_without_pixel_editing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[tuple[str, ...]] = []

    def fake_run(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        commands.append(tuple(command))
        Path(command[-1]).write_bytes(b"lossless-png")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(
        "scripts.capture_media_gallery._require_tool",
        lambda _tool: "magick",
    )
    monkeypatch.setattr("scripts.capture_media_gallery.run_checked", fake_run)
    monkeypatch.setattr(
        "scripts.capture_media_gallery.probe_media",
        lambda _path: MediaProbe(1200, 800, "png", None),
    )

    report = capture_window_still(
        tmp_path,
        Path("raw/interface.png"),
        "0x123",
    )

    assert commands[0][1:4] == ("import", "-window", "0x123")
    assert (tmp_path / "raw" / "interface.png").read_bytes() == b"lossless-png"
    assert report.width == 1200


def test_window_geometry_parser_descends_from_dataclass_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "scripts.capture_media_gallery._require_tool",
        lambda _tool: "xdotool",
    )
    monkeypatch.setattr(
        "scripts.capture_media_gallery.run_checked",
        lambda command: subprocess.CompletedProcess(
            command,
            0,
            "X=11\nY=17\nWIDTH=1400\nHEIGHT=900\nSCREEN=0\n",
            "",
        ),
    )

    assert read_window_geometry("0x123") == WindowGeometry(
        x=11,
        y=17,
        width=1400,
        height=900,
    )


def test_record_window_projects_fixed_geometry_and_lossless_ffv1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[tuple[str, ...]] = []

    def fake_run(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        commands.append(tuple(command))
        Path(command[-1]).write_bytes(b"lossless-ffv1")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(
        "scripts.capture_media_gallery._require_tool",
        lambda _tool: "ffmpeg",
    )
    monkeypatch.setattr(
        "scripts.capture_media_gallery.read_window_geometry",
        lambda _window_id: WindowGeometry(x=11, y=17, width=1400, height=900),
    )
    monkeypatch.setattr("scripts.capture_media_gallery.run_checked", fake_run)
    monkeypatch.setattr(
        "scripts.capture_media_gallery.probe_media",
        lambda _path: MediaProbe(1400, 900, "ffv1", 12.0),
    )

    report = record_window(
        tmp_path,
        Path("raw/interaction.mkv"),
        "0x123",
        duration_seconds=12.0,
        fps=30,
        display=":9.0",
        draw_mouse=False,
    )

    command = commands[0]
    assert command[command.index("-f") + 1] == "x11grab"
    assert command[command.index("-draw_mouse") + 1] == "0"
    assert command[command.index("-video_size") + 1] == "1400x900"
    assert command[command.index("-i") + 1] == ":9.0+11,17"
    assert command[command.index("-c:v") + 1] == "ffv1"
    assert (tmp_path / "raw" / "interaction.mkv").read_bytes() == b"lossless-ffv1"
    assert report.duration_seconds == 12.0
    assert report.mouse_visible is False


def test_build_preflights_every_existing_output_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "sources"
    output_root = tmp_path / "outputs"
    source_path = source_root / "raw" / "session.mkv"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"lossless-source")
    output_root.mkdir()
    existing_output = output_root / "lazy-inheritance.webm"
    existing_output.write_bytes(b"accepted-output")
    manifest = CaptureManifest(
        captures=(
            CaptureRecord(
                source=Path("raw/session.mkv"),
                scenario_id="lazy-inheritance",
                trim=Trim(start_seconds=0.0, duration_seconds=8.0),
                outputs=(
                    Derivative(
                        filename="lazy-inheritance-poster.webp",
                        max_width=1600,
                        max_height=1000,
                        max_bytes=2_000_000,
                        frame_at_seconds=1.0,
                    ),
                    Derivative(
                        filename="lazy-inheritance.webm",
                        max_width=1600,
                        max_height=1000,
                        max_bytes=3_000_000,
                        fps=24,
                    ),
                    Derivative(
                        filename="lazy-inheritance.mp4",
                        max_width=1600,
                        max_height=1000,
                        max_bytes=3_000_000,
                        fps=24,
                    ),
                ),
            ),
        )
    )
    monkeypatch.setattr(
        "scripts.capture_media_gallery.probe_media",
        lambda _path: MediaProbe(800, 500, "ffv1", 8.0),
    )

    with pytest.raises(MediaGalleryError, match="Derived outputs already exist"):
        build_manifest(manifest, source_root, output_root)

    assert not (output_root / "lazy-inheritance-poster.webp").exists()
    assert existing_output.read_bytes() == b"accepted-output"


def test_derivative_validation_enforces_size_dimensions_codec_and_duration(
    tmp_path: Path,
) -> None:
    output = Derivative(
        filename="lazy-update.webm",
        max_width=1280,
        max_height=800,
        max_bytes=100,
        fps=20,
    )
    record = _motion_record(output)
    target = tmp_path / output.filename
    target.write_bytes(b"x" * 50)

    report = validate_derivative(
        record,
        output,
        target,
        probe=MediaProbe(
            width=1200,
            height=760,
            codec_name="vp9",
            duration_seconds=8.0,
        ),
    )
    assert report.size_bytes == 50
    assert report.width == 1200

    with pytest.raises(MediaGalleryError, match="above its 1280x800 manifest bounds"):
        validate_derivative(
            record,
            output,
            target,
            probe=MediaProbe(
                width=1300,
                height=760,
                codec_name="vp9",
                duration_seconds=8.0,
            ),
        )
    with pytest.raises(MediaGalleryError, match="expected 'vp9'"):
        validate_derivative(
            record,
            output,
            target,
            probe=MediaProbe(
                width=1200,
                height=760,
                codec_name="h264",
                duration_seconds=8.0,
            ),
        )
    with pytest.raises(MediaGalleryError, match="duration is"):
        validate_derivative(
            record,
            output,
            target,
            probe=MediaProbe(
                width=1200,
                height=760,
                codec_name="vp9",
                duration_seconds=6.0,
            ),
        )


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="FFmpeg is an optional host capture dependency",
)
def test_real_ffmpeg_derivatives_are_bounded_and_revalidate(tmp_path: Path) -> None:
    source_root = tmp_path / "sources"
    output_root = tmp_path / "outputs"
    source_path = source_root / "raw" / "session.mkv"
    source_path.parent.mkdir(parents=True)
    subprocess.run(
        (
            shutil.which("ffmpeg") or "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=20:duration=2",
            "-c:v",
            "ffv1",
            "-threads",
            "1",
            str(source_path),
        ),
        check=True,
    )
    outputs = (
        Derivative(
            filename="lazy-inheritance-poster.webp",
            max_width=240,
            max_height=140,
            max_bytes=500_000,
            frame_at_seconds=0.25,
        ),
        Derivative(
            filename="lazy-inheritance.webm",
            max_width=240,
            max_height=140,
            max_bytes=1_000_000,
            fps=10,
        ),
        Derivative(
            filename="lazy-inheritance.mp4",
            max_width=240,
            max_height=140,
            max_bytes=1_000_000,
            fps=10,
        ),
    )
    manifest = CaptureManifest(
        captures=(
            CaptureRecord(
                source=Path("raw/session.mkv"),
                scenario_id="lazy-inheritance",
                trim=Trim(start_seconds=0.25, duration_seconds=1.0),
                outputs=outputs,
            ),
        )
    )

    build_report = build_manifest(manifest, source_root, output_root)
    validation_report = validate_manifest_outputs(
        manifest,
        source_root,
        output_root,
    )
    rebuilt_report = build_manifest(
        manifest,
        source_root,
        output_root,
        force=True,
    )

    assert {item.path for item in build_report[0].outputs} == {
        output.filename for output in outputs
    }
    assert {item.path: item.sha256 for item in build_report[0].outputs} == {
        item.path: item.sha256 for item in rebuilt_report[0].outputs
    }
    assert validation_report[0].source.sha256
    assert all((output_root / output.filename).is_file() for output in outputs)
