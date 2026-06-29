"""Pipeline authoring and execution renderers for the MCP dev client."""

from __future__ import annotations

import json
from collections.abc import Mapping

from openhcs.agent import dto as agent_dto
from openhcs.agent.capabilities import agent_capabilities
from openhcs.agent.dto.common import JsonObject, JsonValue
from openhcs.mcp.dev_client_rendering import (
    McpDevOutputRenderer,
    McpDevPayloadProjection,
)
from openhcs.mcp.dev_client_core import optional_bool, optional_int
from openhcs.mcp.dev_client_renderers.object_state import ObjectStateScopeRenderer
from openhcs.mcp.dev_client_renderers.ui_bridge import CodeDocumentRenderer
from openhcs.mcp.dev_client_renderers.viewer import ViewerValidationRenderer

class PipelineDraftStepRenderer:
    """Compact renderer for one-step pipeline draft smoke checks."""

    @classmethod
    def render(cls, response: JsonObject, *, max_source_chars: int = 2_000) -> str:
        create_payload = cls._payload_for_tool(response, agent_capabilities.create_pipeline.name)
        add_payload = cls._payload_for_tool(response, agent_capabilities.add_function_step.name)
        validate_payload = cls._payload_for_tool(response, agent_capabilities.validate_pipeline.name)
        render_payload = cls._payload_for_tool(response, agent_capabilities.render_pipeline_source.name)
        pipeline_id = (
            create_payload.get("pipeline_id")
            if create_payload is not None
            else None
        )
        pipeline_steps = (
            McpDevPayloadProjection.sequence_of_mappings(add_payload.get("steps"))
            if add_payload is not None
            else ()
        )
        lines = [
            (
                "Pipeline draft: "
                f"id={McpDevPayloadProjection.text(pipeline_id)} "
                f"valid={cls._valid_text(validate_payload)} "
                f"steps={len(pipeline_steps)}"
            )
        ]
        if create_payload is not None:
            lines.append(
                f"Ref: uri={McpDevPayloadProjection.text(create_payload.get('uri'))}"
            )
        cls._append_payload_messages(lines, create_payload, "Create")
        cls._append_payload_messages(lines, add_payload, "Add step")
        cls._append_payload_messages(lines, validate_payload, "Validate")
        cls._append_repair_hints(lines, validate_payload, pipeline_steps)
        if pipeline_steps:
            lines.append("Steps:")
            lines.extend(cls._step_lines(pipeline_steps))
        if render_payload is not None:
            source = render_payload.get("source")
            if isinstance(source, str):
                lines.append(
                    "Source: "
                    f"title={McpDevPayloadProjection.quoted_text(render_payload.get('title'))} "
                    f"bytes={len(source)}"
                )
                lines.append(
                    CodeDocumentRenderer._source_text(
                        source,
                        max_source_chars=max_source_chars,
                    )
                )
            cls._append_payload_messages(lines, render_payload, "Render")
        return "\n".join(lines)

    @staticmethod
    def _payload_for_tool(
        response: JsonObject,
        tool_name: str,
    ) -> Mapping[str, JsonValue] | None:
        return McpDevPayloadProjection.tool_payload(response, tool_name)

    @staticmethod
    def _valid_text(payload: Mapping[str, JsonValue] | None) -> str:
        if payload is None:
            return "<not-run>"
        return McpDevPayloadProjection.text(payload.get("valid"))

    @classmethod
    def _step_lines(cls, steps: tuple[Mapping[str, JsonValue], ...]) -> list[str]:
        lines: list[str] = []
        for step in steps:
            functions = McpDevPayloadProjection.sequence_of_mappings(
                step.get("functions")
            )
            function_ids = tuple(
                McpDevPayloadProjection.text(function.get("function_id"))
                for function in functions
            )
            lines.append(
                "- "
                f"{McpDevPayloadProjection.text(step.get('step_id'))}: "
                f"name={McpDevPayloadProjection.quoted_text(step.get('name'))} "
                f"enabled={McpDevPayloadProjection.text(step.get('enabled'))} "
                f"functions={','.join(function_ids) if function_ids else '<none>'}"
            )
        return lines

    @staticmethod
    def _append_payload_messages(
        lines: list[str],
        payload: Mapping[str, JsonValue] | None,
        label: str,
    ) -> None:
        if payload is None:
            return
        errors = McpDevPayloadProjection.sequence_of_mappings(payload.get("errors"))
        if errors:
            lines.append(f"{label} errors:")
            lines.extend(ViewerValidationRenderer._error_lines(errors))
        warnings = McpDevPayloadProjection.sequence_of_mappings(payload.get("warnings"))
        if warnings:
            lines.append(f"{label} warnings:")
            lines.extend(ViewerValidationRenderer._error_lines(warnings))

    @classmethod
    def _append_repair_hints(
        cls,
        lines: list[str],
        validate_payload: Mapping[str, JsonValue] | None,
        steps: tuple[Mapping[str, JsonValue], ...],
    ) -> None:
        if validate_payload is None or not steps:
            return
        errors = McpDevPayloadProjection.sequence_of_mappings(
            validate_payload.get("errors")
        )
        missing_kwargs = cls._missing_function_kwargs(errors)
        if not missing_kwargs:
            return
        function_id = cls._first_function_id(steps)
        if function_id is None:
            return
        kwargs_shape = ", ".join(
            f'"{name}": <value>'
            for name in missing_kwargs
        )
        lines.append(f"Next: function {function_id}")
        lines.append(
            "Retry shape: "
            f"draft-pipeline-step {function_id} --kwargs '{{{kwargs_shape}}}'"
        )

    @staticmethod
    def _missing_function_kwargs(
        errors: tuple[Mapping[str, JsonValue], ...],
    ) -> tuple[str, ...]:
        for error in errors:
            if error.get("code") != "missing_function_kwargs":
                continue
            for key in ("hint", "message"):
                value = error.get(key)
                if not isinstance(value, str):
                    continue
                names = PipelineDraftStepRenderer._parse_missing_kwargs(value)
                if names:
                    return names
        return ()

    @staticmethod
    def _parse_missing_kwargs(text: str) -> tuple[str, ...]:
        marker = "required agent kwargs:"
        if marker in text:
            tail = text.split(marker, 1)[1]
        elif ": " in text:
            tail = text.rsplit(": ", 1)[1]
        else:
            return ()
        tail = tail.strip().rstrip(".")
        names = tuple(
            name.strip().strip("`'")
            for name in tail.split(",")
            if name.strip()
        )
        return names

    @staticmethod
    def _first_function_id(
        steps: tuple[Mapping[str, JsonValue], ...],
    ) -> str | None:
        for step in steps:
            functions = McpDevPayloadProjection.sequence_of_mappings(
                step.get("functions")
            )
            for function in functions:
                function_id = function.get("function_id")
                if isinstance(function_id, str) and function_id:
                    return function_id
        return None

class PipelineArtifactPlanRenderer(McpDevOutputRenderer):
    """Compact renderer for pycodified pipeline artifact-plan inspection."""

    output_contract = agent_dto.ArtifactPlanInspection

    @classmethod
    def render(cls, response: JsonObject) -> str:
        payload = McpDevPayloadProjection.first_tool_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)
        lines = [
            (
                "Artifact plan: "
                f"plate={McpDevPayloadProjection.text(payload.get('plate_path'))} "
                f"axes={McpDevPayloadProjection.text(payload.get('axis_count'))} "
                f"steps={McpDevPayloadProjection.text(payload.get('step_count'))} "
                f"progress_events={McpDevPayloadProjection.text(payload.get('progress_event_count'))}"
            )
        ]
        ObjectStateScopeRenderer._append_messages(lines, payload)
        axes = payload.get("axes")
        if axes:
            lines.append(
                f"Axes: {ViewerValidationRenderer._sequence_text(axes)}"
            )
        axis_filter = payload.get("axis_filter")
        if axis_filter:
            lines.append(
                f"Axis filter: {ViewerValidationRenderer._sequence_text(axis_filter)}"
            )
        source_workspace = McpDevPayloadProjection.nested_mapping(
            payload,
            "source_workspace",
        )
        if source_workspace:
            lines.extend(cls._source_workspace_lines(source_workspace))
        worker_assignments = McpDevPayloadProjection.nested_mapping(
            payload,
            "worker_assignments",
        )
        if worker_assignments:
            lines.append(f"Workers: {cls._worker_text(worker_assignments)}")
        steps = McpDevPayloadProjection.sequence_of_mappings(payload.get("steps"))
        if steps:
            lines.append("Steps:")
            lines.extend(cls._step_lines(steps))
        return "\n".join(lines)

    @classmethod
    def _source_workspace_lines(
        cls,
        source_workspace: Mapping[str, JsonValue],
    ) -> list[str]:
        lines = [
            (
                "Source workspace (source-bound files): "
                f"files={McpDevPayloadProjection.text(source_workspace.get('file_count'))} "
                f"truncated={McpDevPayloadProjection.text(source_workspace.get('truncated_file_count'))}"
            )
        ]
        axis_file_counts = McpDevPayloadProjection.nested_mapping(
            source_workspace,
            "axis_file_counts",
        )
        if axis_file_counts:
            lines.append(f"  axis files: {cls._mapping_text(axis_file_counts)}")
        if optional_int(source_workspace.get("file_count")) == 0:
            lines.append(
                "  note: no source-bound virtual files were compiled. Standard "
                "microscope input may still be available through the plate handler; "
                "use inspect-plate or selected-plate-images to review raw image "
                "inventory, and configure source bindings for custom source-bound "
                "layouts."
            )
        files = McpDevPayloadProjection.sequence_of_mappings(
            source_workspace.get("files")
        )
        for file_record in files[:5]:
            virtual_path = McpDevPayloadProjection.text(file_record.get("virtual_path"))
            source_path = McpDevPayloadProjection.text(file_record.get("source_path"))
            metadata = McpDevPayloadProjection.nested_mapping(
                file_record,
                "source_metadata",
            )
            line = f"  - {virtual_path}"
            if source_path and source_path != "<none>" and source_path != virtual_path:
                line += f" -> {source_path}"
            if metadata:
                line += f" components={cls._mapping_text(metadata)}"
            lines.append(line)
        if len(files) > 5:
            lines.append("  - ...")
        return lines

    @classmethod
    def _step_lines(
        cls,
        steps: tuple[Mapping[str, JsonValue], ...],
    ) -> list[str]:
        lines: list[str] = []
        for step in steps:
            lines.append(
                "- "
                f"{McpDevPayloadProjection.text(step.get('step_index'))}: "
                f"{McpDevPayloadProjection.text(step.get('step_name'))} "
                f"axis={McpDevPayloadProjection.text(step.get('axis_id'))} "
                f"groups={ViewerValidationRenderer._sequence_text(step.get('execution_groups'))}"
            )
            artifact_inputs = McpDevPayloadProjection.sequence_of_mappings(
                step.get("artifact_inputs")
            )
            for artifact in artifact_inputs:
                lines.append(cls._artifact_input_line(artifact))
            artifacts = McpDevPayloadProjection.sequence_of_mappings(
                step.get("artifact_outputs")
            )
            for artifact in artifacts:
                lines.append(
                    "  artifact "
                    f"{McpDevPayloadProjection.text(artifact.get('name'))}: "
                    f"kind={McpDevPayloadProjection.text(artifact.get('kind'))} "
                    f"path={McpDevPayloadProjection.text(artifact.get('path'))} "
                    f"groups={ViewerValidationRenderer._sequence_text(artifact.get('group_keys'))}"
                )
                materialization = McpDevPayloadProjection.nested_mapping(
                    artifact,
                    "materialization",
                )
                if materialization:
                    lines.extend(cls._materialization_lines(materialization))
            truncated_input_count = step.get("truncated_artifact_input_count")
            if isinstance(truncated_input_count, int) and truncated_input_count > 0:
                lines.append(f"  artifact input ... truncated={truncated_input_count}")
            truncated_count = step.get("truncated_artifact_output_count")
            if isinstance(truncated_count, int) and truncated_count > 0:
                lines.append(f"  artifact ... truncated={truncated_count}")
        return lines

    @staticmethod
    def _artifact_input_line(artifact: Mapping[str, JsonValue]) -> str:
        source_parts: list[str] = []
        source_step_id = artifact.get("source_step_id")
        if source_step_id is not None:
            source_parts.append(
                f"source_step={McpDevPayloadProjection.text(source_step_id)}"
            )
        source_step_scope_id = artifact.get("source_step_scope_id")
        if source_step_scope_id is not None:
            source_parts.append(
                f"source_scope={McpDevPayloadProjection.text(source_step_scope_id)}"
            )
        suffix = f" {' '.join(source_parts)}" if source_parts else ""
        return (
            "  artifact input "
            f"{McpDevPayloadProjection.text(artifact.get('name'))}: "
            f"kind={McpDevPayloadProjection.text(artifact.get('kind'))} "
            f"path={McpDevPayloadProjection.text(artifact.get('path'))} "
            f"groups={ViewerValidationRenderer._sequence_text(artifact.get('group_keys'))}"
            f"{suffix}"
        )

    @classmethod
    def _materialization_lines(
        cls,
        materialization: Mapping[str, JsonValue],
    ) -> list[str]:
        status_parts: list[str] = []
        if optional_bool(materialization.get("disabled")) is True:
            status_parts.append("disabled")
        elif optional_bool(materialization.get("runtime_resolved")) is True:
            status_parts.append("runtime-resolved")
        else:
            status_parts.append("explicit")
        status_parts.append(
            "persistent="
            f"{McpDevPayloadProjection.text(materialization.get('persistent_enabled'))}"
        )
        backend = materialization.get("persistent_backend")
        if backend is not None:
            status_parts.append(f"backend={McpDevPayloadProjection.text(backend)}")
        analysis_output_dir = materialization.get("analysis_output_dir")
        if analysis_output_dir is not None:
            status_parts.append(
                f"analysis_dir={McpDevPayloadProjection.text(analysis_output_dir)}"
            )
        if (
            optional_bool(materialization.get("filename_uses_source_identity"))
            is True
        ):
            status_parts.append("source-identity-filenames")
        if (
            optional_bool(materialization.get("runtime_metadata_can_refine_paths"))
            is True
        ):
            status_parts.append("runtime-metadata-filenames")

        lines = [f"    materialization: {' '.join(status_parts)}"]
        paths = McpDevPayloadProjection.sequence_of_mappings(
            materialization.get("paths")
        )
        for path in paths[:3]:
            group_key = McpDevPayloadProjection.text(path.get("group_key"))
            candidates = cls._candidate_path_text(path.get("candidate_paths"))
            lines.append(
                "      candidates "
                f"group={group_key}: {candidates}"
            )
        if len(paths) > 3:
            lines.append(f"      candidates ... truncated={len(paths) - 3}")
        note = materialization.get("note")
        if note is not None:
            lines.append(f"      note: {McpDevPayloadProjection.text(note)}")
        return lines

    @staticmethod
    def _candidate_path_text(value: JsonValue) -> str:
        if isinstance(value, list | tuple):
            return ", ".join(McpDevPayloadProjection.text(item) for item in value)
        return McpDevPayloadProjection.text(value)

    @staticmethod
    def _mapping_text(mapping: Mapping[str, JsonValue]) -> str:
        return ", ".join(
            f"{key}={McpDevPayloadProjection.text(value)}"
            for key, value in mapping.items()
        )

    @classmethod
    def _worker_text(cls, mapping: Mapping[str, JsonValue]) -> str:
        return ", ".join(
            f"{worker}=[{ViewerValidationRenderer._sequence_text(axes)}]"
            for worker, axes in mapping.items()
        )

class ExecuteSourceRenderer:
    """Compact renderer for source-backed headless execution command results."""

    SESSION_TOOL = agent_capabilities.create_orchestrator_session_from_pipeline_source.name
    SUBMIT_TOOL = agent_capabilities.submit_pipeline_execution.name

    @classmethod
    def render(cls, response: JsonObject) -> str:
        session_payload = McpDevPayloadProjection.tool_payload(
            response,
            cls.SESSION_TOOL,
        )
        submit_payload = McpDevPayloadProjection.tool_payload(
            response,
            cls.SUBMIT_TOOL,
        )
        if session_payload is None and submit_payload is None:
            return json.dumps(response, indent=2, sort_keys=True)

        lines = ["Headless source execution:"]
        if session_payload is not None:
            lines.append(
                "Session: "
                f"id={McpDevPayloadProjection.text(session_payload.get('session_id'))} "
                f"uri={McpDevPayloadProjection.text(session_payload.get('uri'))}"
            )
            ObjectStateScopeRenderer._append_messages(lines, session_payload)
        else:
            lines.append("Session: <not created>")

        if submit_payload is not None:
            lines.append(
                "Job: "
                f"id={McpDevPayloadProjection.text(submit_payload.get('job_id'))} "
                f"kind={McpDevPayloadProjection.text(submit_payload.get('kind'))} "
                f"status={McpDevPayloadProjection.text(submit_payload.get('status'))} "
                "server_execution="
                f"{McpDevPayloadProjection.text(submit_payload.get('server_execution_id'))}"
            )
            response_payload = McpDevPayloadProjection.nested_mapping(
                submit_payload,
                "response",
            )
            if response_payload:
                lines.append(
                    "Response: "
                    + cls._response_summary_text(response_payload)
                )
            ObjectStateScopeRenderer._append_messages(lines, submit_payload)
        else:
            lines.append("Job: <not submitted>")
        return "\n".join(lines)

    @staticmethod
    def _response_summary_text(response_payload: Mapping[str, JsonValue]) -> str:
        summary_keys = (
            "status",
            "state",
            "message",
            "execution_id",
            "server_execution_id",
            "completed",
            "success",
        )
        parts = [
            f"{key}={McpDevPayloadProjection.text(response_payload.get(key))}"
            for key in summary_keys
            if key in response_payload
        ]
        if parts:
            return " ".join(parts)
        return f"keys={ViewerValidationRenderer._sequence_text(sorted(response_payload))}"
