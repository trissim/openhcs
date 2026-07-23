from __future__ import annotations

from openhcs.mcp.dev_client_renderers.pipeline import PipelineArtifactPlanRenderer


def test_artifact_plan_renderer_shows_main_flow_and_viewer_plans() -> None:
    response = {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_inspect_pipeline_source_artifact_plan",
                "mcp_error": False,
                "payloads": [
                    {
                        "plate_path": "/tmp/plate",
                        "axis_count": 1,
                        "step_count": 1,
                        "progress_event_count": 1,
                        "axes": ["A01"],
                        "steps": [
                            {
                                "step_index": 0,
                                "step_name": "Denoise",
                                "axis_id": "A01",
                                "execution_groups": [None],
                                "main_flow_materialization": {
                                    "output_dir": "/tmp/out/checkpoints",
                                    "backend": "disk",
                                    "plate_root": "/tmp/out",
                                    "sub_dir": "checkpoints",
                                },
                                "viewer_streaming": [
                                    {
                                        "config_key": "napari_streaming_config",
                                        "viewer_type": "napari",
                                        "backend": "napari_stream",
                                        "effective_config": {
                                            "enabled": True,
                                            "persistent": True,
                                            "well_filter": ["A01"],
                                            "port": 5555,
                                        },
                                    }
                                ],
                                "artifact_inputs": [],
                                "artifact_outputs": [],
                            }
                        ],
                    }
                ],
            }
        ],
    }

    rendered = PipelineArtifactPlanRenderer.render(response)

    assert "main-flow checkpoint: backend=disk" in rendered
    assert "output_dir=/tmp/out/checkpoints" in rendered
    assert "viewer stream: viewer=napari" in rendered
    assert "config=napari_streaming_config" in rendered
    assert "enabled=True" in rendered
    assert "well_filter=['A01']" in rendered
