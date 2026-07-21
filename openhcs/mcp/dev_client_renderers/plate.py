"""Plate renderers for the MCP dev client."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar

from openhcs.agent.capabilities import agent_capabilities
from openhcs.agent.dto.common import JsonObject, JsonValue
from openhcs.agent.dto.plate import (
    PlateFileQueryResult,
    PlateFileStreamResult,
    PlateImageSampleResult,
    PlateInspectionIssueCode,
    PlatePathInspectionResult,
    SelectedPlateFileQueryTarget,
    SelectedPlateFileQueryResult,
    SelectedPlateFileStreamResult,
    SelectedPlateImageInspectionResult,
    SelectedPlateImageSampleResult,
    SyntheticPlateGenerationResult,
)
from openhcs.mcp.dev_client_core import optional_int
from openhcs.mcp.dev_client_rendering import (
    McpDevOutputRenderer,
    McpDevPayloadProjection,
    McpDiagnosticRenderer,
)

class PlateImageSampleRenderer(McpDevOutputRenderer):
    """Compact renderer for sampled plate image pixels and statistics."""

    output_contract = PlateImageSampleResult

    @classmethod
    def render(cls, response: JsonObject) -> str:
        sample_payload = cls._first_tool_payload(response)
        if sample_payload is None:
            return json.dumps(response, indent=2, sort_keys=True)
        errors = sample_payload.get("errors")
        if isinstance(errors, list) and errors:
            return "\n".join(cls._error_lines(errors))

        sample_values = sample_payload.get("sample_values")
        sample_value_count = cls._json_value_count(sample_values)
        lines = [
            f"Image: {cls._text(sample_payload.get('virtual_path'))}",
            f"Source: {cls._text(sample_payload.get('source_path'))}",
            (
                "Resolution: "
                f"selected={cls._text(sample_payload.get('selected_resolution_index'))} "
                f"count={cls._text(sample_payload.get('resolution_count'))} "
                f"source_shape={cls._sequence_text(sample_payload.get('shape'))} "
                "resolution_shape="
                f"{cls._sequence_text(sample_payload.get('resolution_shape'))} "
                f"downsample_yx={cls._sequence_text(sample_payload.get('downsample_yx'))}"
            ),
            (
                "Statistics: "
                f"scope={cls._text(sample_payload.get('statistics_scope'))} "
                f"dtype={cls._text(sample_payload.get('dtype'))} "
                f"min={cls._text(sample_payload.get('minimum'))} "
                f"max={cls._text(sample_payload.get('maximum'))} "
                f"mean={cls._mean_text(sample_payload.get('mean'))}"
            ),
            (
                "Sample: "
                f"origin_yx={cls._sequence_text(sample_payload.get('sample_origin_yx'))} "
                f"shape={cls._sequence_text(sample_payload.get('sample_shape'))} "
                f"included={cls._text(sample_payload.get('sample_included'))}"
            ),
        ]
        if sample_payload.get("sample_included") is True:
            if sample_value_count <= 64:
                lines.append("Sample values:")
                lines.append(json.dumps(sample_values, indent=2))
            else:
                lines.append(
                    f"Sample values: {sample_value_count} elements; pass --json to print them."
                )
        else:
            omitted_reason = cls._text(sample_payload.get("sample_omitted_reason"))
            omitted_line = f"Sample values omitted: {omitted_reason}"
            required_elements = cls._shape_element_count(
                sample_payload.get("sample_shape")
            )
            if (
                cls._omitted_by_element_budget(omitted_reason)
                and required_elements is not None
            ):
                omitted_line += (
                    f"; rerun with --max-array-elements {required_elements} "
                    "or smaller --width/--height"
                )
            elif omitted_reason == "array_values_not_requested":
                omitted_line += "; rerun with --include-array-values"
                if required_elements is not None:
                    omitted_line += f" --max-array-elements {required_elements}"
            lines.append(omitted_line)
        return "\n".join(lines)

    @staticmethod
    def _first_tool_payload(payload: JsonObject) -> Mapping[str, JsonValue] | None:
        results = payload.get("results")
        if not isinstance(results, list) or not results:
            return None
        result = results[0]
        if not isinstance(result, Mapping):
            return None
        payloads = result.get("payloads")
        if not isinstance(payloads, list) or not payloads:
            return None
        first_payload = payloads[0]
        if not isinstance(first_payload, Mapping):
            return None
        return first_payload

    @classmethod
    def _error_lines(cls, errors: list[JsonValue]) -> list[str]:
        lines = ["Sample failed:"]
        for error in errors[:3]:
            if isinstance(error, Mapping):
                code = cls._text(error.get("code"))
                message = cls._text(error.get("message"))
                hint = error.get("hint")
                lines.append(f"- {code}: {message}")
                if hint is not None:
                    lines.append(f"  hint: {cls._text(hint)}")
            else:
                lines.append(f"- {cls._text(error)}")
        if len(errors) > 3:
            lines.append(f"... {len(errors) - 3} more errors")
        return lines

    @staticmethod
    def _json_value_count(value: JsonValue) -> int:
        if isinstance(value, list | tuple):
            return sum(PlateImageSampleRenderer._json_value_count(item) for item in value)
        if isinstance(value, Mapping):
            return sum(
                PlateImageSampleRenderer._json_value_count(item)
                for item in value.values()
            )
        if value is None:
            return 0
        return 1

    @staticmethod
    def _shape_element_count(value: JsonValue) -> int | None:
        if not isinstance(value, list | tuple) or not value:
            return None
        element_count = 1
        for item in value:
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                return None
            element_count *= item
        return element_count

    @staticmethod
    def _omitted_by_element_budget(reason: str) -> bool:
        return (
            reason == "max_array_elements_exceeded"
            or "max_array_elements" in reason
        )

    @staticmethod
    def _sequence_text(value: JsonValue) -> str:
        if isinstance(value, list | tuple):
            return "x".join(str(item) for item in value)
        return PlateImageSampleRenderer._text(value)

    @staticmethod
    def _mean_text(value: JsonValue) -> str:
        if isinstance(value, int | float):
            return f"{value:.3f}"
        return PlateImageSampleRenderer._text(value)

    @staticmethod
    def _text(value: JsonValue) -> str:
        if value is None:
            return "<none>"
        return str(value)

class SyntheticPlateGenerationRenderer(McpDevOutputRenderer):
    """Compact renderer for synthetic plate generation results."""

    output_contract = SyntheticPlateGenerationResult

    @classmethod
    def render(cls, response: JsonObject) -> str:
        payload = McpDevPayloadProjection.first_tool_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)
        errors = McpDevPayloadProjection.sequence_of_mappings(payload.get("errors"))
        if errors:
            return "\n".join(
                (
                    "Synthetic plate generation: failed",
                    *PlateInspectionRenderer._error_lines(errors),
                )
            )
        output_dir = McpDevPayloadProjection.text(payload.get("output_dir"))
        sampled_files = cls._text_sequence(payload.get("sampled_image_files"))
        lines = [
            f"Synthetic plate: {output_dir}",
            (
                "Geometry: "
                f"grid={cls._sequence_text(payload.get('grid_size'))} "
                f"tile={cls._sequence_text(payload.get('tile_size'))} "
                f"overlap={McpDevPayloadProjection.text(payload.get('overlap_percent'))}% "
                f"stage_error_px={McpDevPayloadProjection.text(payload.get('stage_error_px'))}"
            ),
            (
                "Content: "
                f"wells={cls._sequence_text(payload.get('wells'))} "
                f"channels={McpDevPayloadProjection.text(payload.get('wavelengths'))} "
                f"z={McpDevPayloadProjection.text(payload.get('z_stack_levels'))} "
                f"cells={McpDevPayloadProjection.text(payload.get('num_cells'))} "
                "shared_fraction="
                f"{McpDevPayloadProjection.text(payload.get('shared_cell_fraction'))}"
            ),
            (
                "Files: "
                f"images={McpDevPayloadProjection.text(payload.get('image_count'))} "
                f"sampled={len(sampled_files)} "
                f"truncated={McpDevPayloadProjection.text(payload.get('truncated_image_count'))}"
            ),
            (
                "Metadata: "
                f"file={McpDevPayloadProjection.text(payload.get('metadata_file_path'))} "
                f"microscope={McpDevPayloadProjection.text(payload.get('detected_microscope_type'))} "
                f"handler={McpDevPayloadProjection.text(payload.get('handler_class'))}"
            ),
        ]
        if sampled_files:
            lines.append("Sample images:")
            lines.extend(f"- {path}" for path in sampled_files[:12])
        lines.append(f"Next: inspect-plate {output_dir}")
        lines.append(f"Next: query-plate-files {output_dir} --limit 10")
        return "\n".join(lines)

    @staticmethod
    def _sequence_text(value: JsonValue) -> str:
        if isinstance(value, list | tuple):
            return "x".join(str(item) for item in value)
        return McpDevPayloadProjection.text(value)

    @staticmethod
    def _text_sequence(value: JsonValue) -> tuple[str, ...]:
        if not isinstance(value, list | tuple):
            return ()
        return tuple(str(item) for item in value)

class PlateInspectionRenderer(McpDevOutputRenderer):
    """Compact renderer for plate inspection results."""

    output_contract = PlatePathInspectionResult

    MAX_TEXT_PREVIEW_LINES: ClassVar[int] = 3
    MAX_CSV_PREVIEW_ROWS: ClassVar[int] = 3
    MAX_CSV_PREVIEW_COLUMNS: ClassVar[int] = 8
    MAX_TEXT_PREVIEW_CHARS: ClassVar[int] = 180
    MAX_CSV_CELL_CHARS: ClassVar[int] = 90
    MAX_CSV_COLUMN_CHARS: ClassVar[int] = 48
    MAX_CSV_ROW_CHARS: ClassVar[int] = 520

    @classmethod
    def render(
        cls,
        response: JsonObject,
        *,
        next_sample_command: str | None = None,
        next_sample_prefix: str = "",
    ) -> str:
        payload = McpDevPayloadProjection.first_tool_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)
        errors = McpDevPayloadProjection.sequence_of_mappings(payload.get("errors"))
        if errors:
            return "\n".join(("Plate inspection: failed", *cls._error_lines(errors)))

        image_files = McpDevPayloadProjection.nested_mapping(payload, "image_files")
        result_files = McpDevPayloadProjection.nested_mapping(payload, "result_files")
        parse_summary = McpDevPayloadProjection.nested_mapping(payload, "parse_summary")
        workspace = McpDevPayloadProjection.nested_mapping(
            payload,
            "workspace_preparation",
        )
        workflow = McpDevPayloadProjection.nested_mapping(
            payload,
            "workflow_advice",
        )
        handler_candidates = McpDevPayloadProjection.sequence_of_mappings(
            payload.get("format_specific_handler_candidates")
        )
        sampled_files = cls._text_sequence(image_files.get("sampled_files"))
        sampled_records = McpDevPayloadProjection.sequence_of_mappings(
            image_files.get("sampled_records")
        )
        result_records = McpDevPayloadProjection.sequence_of_mappings(
            result_files.get("sampled_records")
        )
        result_sample_count = len(result_records) or len(
            cls._text_sequence(result_files.get("sampled_files"))
        )
        sample_count = len(sampled_records) if sampled_records else len(sampled_files)
        lines = [
            f"Plate: {McpDevPayloadProjection.text(payload.get('plate_path'))}",
            (
                "Status: "
                f"{McpDevPayloadProjection.text(payload.get('status'))} "
                f"confidence={McpDevPayloadProjection.text(payload.get('confidence'))} "
                f"microscope={McpDevPayloadProjection.text(payload.get('detected_microscope_type'))}"
            ),
            (
                "Handler: "
                f"{McpDevPayloadProjection.text(payload.get('handler_class'))} "
                f"parser={McpDevPayloadProjection.text(payload.get('parser_class'))}"
            ),
            (
                "Images: "
                f"count={McpDevPayloadProjection.text(image_files.get('count'))} "
                f"sampled={sample_count} "
                f"truncated={McpDevPayloadProjection.text(image_files.get('truncated_file_count'))}"
            ),
            (
                "Results: "
                f"count={McpDevPayloadProjection.text(result_files.get('count'))} "
                f"sampled={result_sample_count} "
                f"scanned={McpDevPayloadProjection.text(result_files.get('scanned_file_count'))} "
                f"truncated={McpDevPayloadProjection.text(result_files.get('truncated_file_count'))}"
            ),
            (
                "Parse: "
                f"attempted={McpDevPayloadProjection.text(parse_summary.get('attempted_file_count'))} "
                f"parsed={McpDevPayloadProjection.text(parse_summary.get('parsed_file_count'))} "
                f"failed={McpDevPayloadProjection.text(parse_summary.get('failed_file_count'))} "
                f"skipped={McpDevPayloadProjection.text(parse_summary.get('skipped_file_count'))}"
            ),
        ]
        grid_dimensions = payload.get("grid_dimensions")
        pixel_size = payload.get("pixel_size")
        lines.append(
            "Geometry: "
            f"grid={cls._sequence_text(grid_dimensions)} "
            f"pixel_size={McpDevPayloadProjection.text(pixel_size)}"
        )
        if workspace:
            lines.append(
                "Workspace: "
                f"{McpDevPayloadProjection.text(workspace.get('operation'))} "
                f"read_only={McpDevPayloadProjection.text(workspace.get('read_only_inspection'))} "
                f"required_before_execution="
                f"{McpDevPayloadProjection.text(workspace.get('required_before_execution'))}"
            )
        if workflow:
            lines.extend(
                (
                    "Routing: "
                    f"scope={McpDevPayloadProjection.text(workflow.get('workflow_scope'))} "
                    f"ingestion={McpDevPayloadProjection.text(workflow.get('ingestion_route'))} "
                    f"owner={McpDevPayloadProjection.text(workflow.get('ingestion_owner'))} "
                    "source_bindings="
                    f"{McpDevPayloadProjection.text(workflow.get('source_binding_role'))}",
                    "UI next: "
                    f"document={McpDevPayloadProjection.text(workflow.get('ui_code_document_id'))} "
                    f"operation={McpDevPayloadProjection.text(workflow.get('ui_operation'))}",
                    f"Advice: {McpDevPayloadProjection.text(workflow.get('message'))}",
                    "Knowledge query: "
                    f"{McpDevPayloadProjection.text(workflow.get('knowledge_query'))}",
                )
            )
        if handler_candidates:
            lines.append("Format-specific handler candidates:")
            lines.extend(
                cls._handler_candidate_line(candidate)
                for candidate in handler_candidates
            )
        components = McpDevPayloadProjection.sequence_of_mappings(
            payload.get("components")
        )
        if components:
            lines.append(cls._axis_summary_line(components))
            lines.append(cls._metadata_sources_line(components))
            lines.append("Components:")
            lines.extend(cls._component_lines(components))
        source_diagnostics = McpDevPayloadProjection.sequence_of_mappings(
            payload.get("source_diagnostics")
        )
        if source_diagnostics:
            lines.append(f"Source diagnostics: {len(source_diagnostics)}")
            lines.extend(cls._source_diagnostic_lines(source_diagnostics))
        modified_line = PlateFileQueryRenderer.records_modified_summary_line(
            "Sampled artifacts modified",
            (*sampled_records, *result_records),
        )
        if modified_line is not None:
            lines.append(modified_line)
        if sampled_records:
            sample_record = cls._preferred_sample_record(sampled_records)
            sample_path = McpDevPayloadProjection.text(
                sample_record.get("virtual_path")
            )
            lines.append("Sample records:")
            lines.extend(cls._record_lines(sampled_records))
            lines.append(
                cls._next_sample_line(
                    payload,
                    sample_path,
                    next_sample_command,
                    next_sample_prefix,
                )
            )
        elif sampled_files:
            lines.append("Sample paths:")
            lines.extend(f"- {sampled_file}" for sampled_file in sampled_files)
            lines.append(
                cls._next_sample_line(
                    payload,
                    sampled_files[0],
                    next_sample_command,
                    next_sample_prefix,
                )
            )
        if result_records:
            lines.append("Result records:")
            lines.extend(cls._result_record_lines(result_records))
        warnings = McpDevPayloadProjection.sequence_of_mappings(payload.get("warnings"))
        if warnings:
            lines.append("Warnings:")
            lines.extend(cls._error_lines(warnings))
        return "\n".join(lines)

    @staticmethod
    def _handler_candidate_line(candidate: Mapping[str, JsonValue]) -> str:
        return (
            "  - "
            f"{McpDevPayloadProjection.text(candidate.get('microscope_type'))} "
            f"parser={McpDevPayloadProjection.text(candidate.get('parser_class'))} "
            f"recognized={McpDevPayloadProjection.text(candidate.get('recognized_file_count'))}/"
            f"{McpDevPayloadProjection.text(candidate.get('tested_file_count'))} "
            f"root={McpDevPayloadProjection.text(candidate.get('root_dir'))} "
            f"metadata_detected={McpDevPayloadProjection.text(candidate.get('metadata_detected'))} "
            f"diagnostic={McpDevPayloadProjection.text(candidate.get('metadata_diagnostic'))}"
        )

    @staticmethod
    def _source_diagnostic_lines(
        diagnostics: tuple[Mapping[str, JsonValue], ...],
    ) -> list[str]:
        """Render one concise line per structured source-level diagnostic."""

        return [
            "- "
            f"{McpDevPayloadProjection.text(diagnostic.get('diagnostic_type'))}: "
            f"{McpDevPayloadProjection.text(diagnostic.get('message'))}"
            for diagnostic in diagnostics
        ]

    @classmethod
    def _record_lines(
        cls,
        records: tuple[Mapping[str, JsonValue], ...],
    ) -> list[str]:
        lines: list[str] = []
        for record in records:
            virtual_path_value = record.get("virtual_path")
            source_path_value = record.get("source_path")
            full_virtual_path_value = record.get("full_virtual_path")
            virtual_path = McpDevPayloadProjection.text(virtual_path_value)
            source_path = McpDevPayloadProjection.text(source_path_value)
            if (
                source_path_value is not None
                and source_path_value
                not in {virtual_path_value, full_virtual_path_value}
            ):
                lines.append(
                    f"- {virtual_path} -> {source_path}"
                    f"{PlateFileQueryRenderer._metadata_suffix(record)}"
                )
            else:
                lines.append(
                    f"- {virtual_path}"
                    f"{PlateFileQueryRenderer._metadata_suffix(record)}"
                )
        return lines

    @staticmethod
    def _preferred_sample_record(
        records: tuple[Mapping[str, JsonValue], ...],
    ) -> Mapping[str, JsonValue]:
        modified_records: list[tuple[str, int, Mapping[str, JsonValue]]] = []
        for index, record in enumerate(records):
            metadata = McpDevPayloadProjection.nested_mapping(record, "metadata")
            modified = metadata.get("modified")
            if modified is not None:
                modified_records.append(
                    (McpDevPayloadProjection.text(modified), index, record)
                )
        if not modified_records:
            return records[0]
        return max(modified_records, key=lambda item: (item[0], -item[1]))[2]

    @staticmethod
    def _result_record_lines(
        records: tuple[Mapping[str, JsonValue], ...],
    ) -> list[str]:
        lines: list[str] = []
        for record in records:
            relative_path = McpDevPayloadProjection.text(record.get("relative_path"))
            file_format = McpDevPayloadProjection.text(record.get("file_format"))
            lines.append(
                f"- {relative_path} type={file_format}"
                f"{PlateFileQueryRenderer._metadata_suffix(record)}"
            )
            preview = McpDevPayloadProjection.nested_mapping(record, "preview")
            lines.extend(PlateInspectionRenderer._result_preview_lines(preview))
        return lines

    @staticmethod
    def _result_preview_lines(
        preview: Mapping[str, JsonValue],
    ) -> list[str]:
        if not preview:
            return []
        omitted_reason = preview.get("omitted_reason")
        if omitted_reason is not None:
            return [
                "  preview omitted: "
                f"{McpDevPayloadProjection.text(omitted_reason)}"
            ]
        roi_count = preview.get("roi_count")
        if roi_count is not None:
            member_text = PlateInspectionRenderer._roi_member_text(preview)
            lines = [
                "  roi preview: "
                f"count={McpDevPayloadProjection.text(roi_count)} "
                f"{member_text}"
                f"area={PlateInspectionRenderer._roi_area_text(preview)}"
            ]
            roi_examples = McpDevPayloadProjection.sequence_of_mappings(
                preview.get("roi_examples")
            )
            for example in roi_examples[:3]:
                lines.append(
                    "  roi example: "
                    f"label={McpDevPayloadProjection.text(example.get('label'))} "
                    f"area={McpDevPayloadProjection.text(example.get('area'))} "
                    f"bbox={PlateInspectionRenderer._json_summary(example.get('bbox'))} "
                    f"centroid={PlateInspectionRenderer._json_summary(example.get('centroid'))}"
                )
            if preview.get("truncated"):
                lines.append("  roi preview: ...")
            return lines
        csv_lines = PlateInspectionRenderer._csv_preview_lines(preview)
        if csv_lines:
            return csv_lines
        text_lines = PlateInspectionRenderer._text_sequence(preview.get("text_lines"))
        if not text_lines:
            return []
        visible_lines = text_lines[: PlateInspectionRenderer.MAX_TEXT_PREVIEW_LINES]
        preview_lines = [
            f"  preview: {PlateInspectionRenderer._bounded_preview_text(line)}"
            for line in visible_lines
        ]
        if preview.get("truncated") or len(text_lines) > len(visible_lines):
            preview_lines.append("  preview: ...")
        return preview_lines

    @staticmethod
    def _csv_preview_lines(preview: Mapping[str, JsonValue]) -> list[str]:
        columns = PlateInspectionRenderer._text_sequence(preview.get("csv_columns"))
        rows = McpDevPayloadProjection.sequence_of_mappings(preview.get("csv_rows"))
        if not PlateInspectionRenderer._valid_csv_columns(columns) or not rows:
            return []
        lines = [
            f"  csv columns: {PlateInspectionRenderer._csv_columns_text(columns)}"
        ]
        visible_rows = rows[: PlateInspectionRenderer.MAX_CSV_PREVIEW_ROWS]
        for row in visible_rows:
            lines.append(
                "  csv row: "
                f"{PlateInspectionRenderer._csv_row_summary(columns, row)}"
            )
        omitted_row_count = len(rows) - len(visible_rows)
        if omitted_row_count > 0:
            lines.append(
                "  csv preview: "
                f"showing {len(visible_rows)}/{len(rows)} rows; "
                f"{omitted_row_count} more in payload"
            )
        if preview.get("truncated"):
            lines.append("  csv preview: ...")
        return lines

    @staticmethod
    def _valid_csv_columns(columns: tuple[str, ...]) -> bool:
        return bool(columns) and all(columns) and len(set(columns)) == len(columns)

    @staticmethod
    def _csv_row_summary(
        columns: tuple[str, ...],
        row: Mapping[str, JsonValue],
    ) -> str:
        compact_cells: list[str] = []
        wide_columns: list[str] = []
        for column in columns:
            value_text = McpDevPayloadProjection.text(row.get(column))
            column_text = PlateInspectionRenderer._bounded_preview_text(
                column,
                PlateInspectionRenderer.MAX_CSV_COLUMN_CHARS,
            )
            if PlateInspectionRenderer._compact_csv_cell(value_text):
                compact_cells.append(f"{column_text}={value_text}")
            else:
                wide_columns.append(column_text)
        row_text = ", ".join(compact_cells) if compact_cells else "<no compact cells>"
        if wide_columns:
            row_text = (
                f"{row_text}; omitted wide cells: {', '.join(wide_columns)}"
            )
        return PlateInspectionRenderer._bounded_preview_text(
            row_text,
            PlateInspectionRenderer.MAX_CSV_ROW_CHARS,
        )

    @staticmethod
    def _csv_columns_text(columns: tuple[str, ...]) -> str:
        visible_columns = columns[: PlateInspectionRenderer.MAX_CSV_PREVIEW_COLUMNS]
        column_text = ", ".join(
            PlateInspectionRenderer._bounded_preview_text(
                column,
                PlateInspectionRenderer.MAX_CSV_COLUMN_CHARS,
            )
            for column in visible_columns
        )
        hidden_count = len(columns) - len(visible_columns)
        if hidden_count > 0:
            column_text = f"{column_text}; {hidden_count} more columns"
        return PlateInspectionRenderer._bounded_preview_text(
            column_text,
            PlateInspectionRenderer.MAX_CSV_ROW_CHARS,
        )

    @staticmethod
    def _compact_csv_cell(value_text: str) -> bool:
        return (
            "\n" not in value_text
            and "\r" not in value_text
            and len(value_text) <= PlateInspectionRenderer.MAX_CSV_CELL_CHARS
        )

    @staticmethod
    def _roi_member_text(preview: Mapping[str, JsonValue]) -> str:
        member_count = optional_int(preview.get("roi_member_count"))
        duplicate_member_count = optional_int(
            preview.get("roi_duplicate_member_count")
        )
        if (
            member_count is None
            or duplicate_member_count is None
            or duplicate_member_count <= 0
        ):
            return ""
        return f"members={member_count} duplicate_members={duplicate_member_count} "

    @staticmethod
    def _roi_area_text(preview: Mapping[str, JsonValue]) -> str:
        minimum = preview.get("roi_area_min")
        maximum = preview.get("roi_area_max")
        mean = preview.get("roi_area_mean")
        if minimum is None and maximum is None and mean is None:
            return "<none>"
        return (
            f"min={McpDevPayloadProjection.text(minimum)},"
            f"mean={PlateInspectionRenderer._float_text(mean)},"
            f"max={McpDevPayloadProjection.text(maximum)}"
        )

    @staticmethod
    def _float_text(value: JsonValue) -> str:
        if isinstance(value, int | float):
            return f"{value:.3f}"
        return McpDevPayloadProjection.text(value)

    @staticmethod
    def _bounded_preview_text(
        text: str,
        max_chars: int = MAX_TEXT_PREVIEW_CHARS,
    ) -> str:
        if len(text) <= max_chars:
            return text
        return f"{text[:max_chars]}..."

    @staticmethod
    def _json_summary(value: JsonValue) -> str:
        if value is None:
            return "<none>"
        return json.dumps(value, sort_keys=True)

    @staticmethod
    def _next_sample_line(
        payload: Mapping[str, JsonValue],
        image_path: str,
        next_sample_command: str | None,
        next_sample_prefix: str = "",
    ) -> str:
        sample_flags = "--height 8 --width 8 --no-array-values"
        if next_sample_command is not None:
            return (
                f"Next: {next_sample_command} "
                f"{next_sample_prefix}{image_path} {sample_flags}"
            )
        return (
            "Next: sample-plate-image "
            f"{McpDevPayloadProjection.text(payload.get('plate_path'))} "
            f"{image_path} {sample_flags}"
        )

    @classmethod
    def _component_lines(
        cls,
        components: tuple[Mapping[str, JsonValue], ...],
    ) -> list[str]:
        lines: list[str] = []
        for component in components:
            values = McpDevPayloadProjection.sequence_of_mappings(
                component.get("values")
            )
            value_text = ", ".join(cls._component_value_text(value) for value in values)
            if not value_text:
                value_text = "<none>"
            lines.append(
                "- "
                f"{McpDevPayloadProjection.text(component.get('component'))}: "
                f"count={McpDevPayloadProjection.text(component.get('count'))} "
                f"source={McpDevPayloadProjection.text(component.get('source'))} "
                f"values={value_text} "
                f"truncated={McpDevPayloadProjection.text(component.get('truncated_value_count'))}"
            )
        return lines

    @classmethod
    def _axis_summary_line(
        cls,
        components: tuple[Mapping[str, JsonValue], ...],
    ) -> str:
        counts = cls._component_counts(components)
        return (
            "Axis sizes: "
            f"wells={cls._axis_count_text(counts, 'well')} "
            f"sites={cls._axis_count_text(counts, 'site')} "
            f"channels={cls._axis_count_text(counts, 'channel')} "
            f"z={cls._axis_count_text(counts, 'z_index')} "
            f"timepoints={cls._axis_count_text(counts, 'timepoint')} "
            f"profile={cls._axis_profile_text(counts)}"
        )

    @classmethod
    def _metadata_sources_line(
        cls,
        components: tuple[Mapping[str, JsonValue], ...],
    ) -> str:
        sources = {
            McpDevPayloadProjection.text(component.get("component")): (
                McpDevPayloadProjection.text(component.get("source"))
            )
            for component in components
        }
        ordered_parts = [
            f"{component}={sources.get(component, '<none>')}"
            for component in ("well", "site", "channel", "z_index", "timepoint")
            if component in sources
        ]
        return f"Metadata sources: {', '.join(ordered_parts) or '<none>'}"

    @staticmethod
    def _component_counts(
        components: tuple[Mapping[str, JsonValue], ...],
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        for component in components:
            component_name = McpDevPayloadProjection.text(component.get("component"))
            count_value = component.get("count")
            if isinstance(count_value, bool):
                continue
            if isinstance(count_value, int):
                counts[component_name] = count_value
        return counts

    @staticmethod
    def _axis_count_text(counts: Mapping[str, int], component: str) -> str:
        count = counts.get(component)
        if count is None:
            return "<unknown>"
        return str(count)

    @classmethod
    def _axis_profile_text(cls, counts: Mapping[str, int]) -> str:
        profile = (
            cls._axis_profile_part(counts, "site", "multi-site", "single-site"),
            cls._axis_profile_part(
                counts,
                "channel",
                "multi-channel",
                "single-channel",
            ),
            cls._axis_profile_part(counts, "z_index", "3D", "2D"),
            cls._axis_profile_part(
                counts,
                "timepoint",
                "time-series",
                "single-timepoint",
            ),
        )
        return ",".join(profile)

    @staticmethod
    def _axis_profile_part(
        counts: Mapping[str, int],
        component: str,
        multiple_label: str,
        singleton_label: str,
    ) -> str:
        count = counts.get(component)
        if count is None:
            return f"unknown-{component}"
        if count > 1:
            return multiple_label
        return singleton_label

    @staticmethod
    def _component_value_text(value: Mapping[str, JsonValue]) -> str:
        key = McpDevPayloadProjection.text(value.get("key"))
        label = value.get("label")
        if label is None or str(label) == "None":
            return key
        return f"{key} ({label})"

    @staticmethod
    def _text_sequence(value: JsonValue) -> tuple[str, ...]:
        if not isinstance(value, list):
            return ()
        return tuple(McpDevPayloadProjection.text(item) for item in value)

    @staticmethod
    def _sequence_text(value: JsonValue) -> str:
        if isinstance(value, list):
            return "x".join(str(item) for item in value)
        return McpDevPayloadProjection.text(value)

    @staticmethod
    def _error_lines(errors: tuple[Mapping[str, JsonValue], ...]) -> tuple[str, ...]:
        return McpDiagnosticRenderer.error_lines(errors)

class PlateFileQueryRenderer(McpDevOutputRenderer):
    """Compact renderer for plate file query results."""

    output_contract = PlateFileQueryResult

    @classmethod
    def render(cls, response: JsonObject) -> str:
        payload = McpDevPayloadProjection.first_tool_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)
        errors = McpDevPayloadProjection.sequence_of_mappings(payload.get("errors"))
        if errors:
            return "\n".join(("Plate file query: failed", *PlateInspectionRenderer._error_lines(errors)))
        records = McpDevPayloadProjection.sequence_of_mappings(payload.get("records"))
        lines = [
            f"Plate file query: {McpDevPayloadProjection.text(payload.get('plate_path'))}",
            (
                "Result: "
                f"returned={McpDevPayloadProjection.text(payload.get('returned_count'))} "
                f"total={McpDevPayloadProjection.text(payload.get('total_count'))} "
                f"offset={McpDevPayloadProjection.text(payload.get('offset'))} "
                f"limit={McpDevPayloadProjection.text(payload.get('limit'))} "
                f"truncated={McpDevPayloadProjection.text(payload.get('truncated_count'))}"
            ),
            (
                "Handler: "
                f"{McpDevPayloadProjection.text(payload.get('handler_class'))} "
                f"parser={McpDevPayloadProjection.text(payload.get('parser_class'))}"
            ),
        ]
        modified_line = cls.records_modified_summary_line(
            "Returned records modified",
            records,
        )
        if modified_line is not None:
            lines.append(modified_line)
        stale_result_line = cls._stale_result_warning_line(records)
        if stale_result_line is not None:
            lines.append(stale_result_line)
        record_root_line = cls._record_root_summary_line(
            payload.get("plate_path"),
            records,
        )
        if record_root_line is not None:
            lines.append(record_root_line)
        if records:
            lines.append("Records:")
            lines.extend(cls._record_lines(records))
        else:
            lines.append("Records: <none>")
        next_page_line = cls._next_page_line(payload)
        if next_page_line is not None:
            lines.append(next_page_line)
        warnings = McpDevPayloadProjection.sequence_of_mappings(payload.get("warnings"))
        if warnings:
            lines.append("Warnings:")
            lines.extend(PlateInspectionRenderer._error_lines(warnings))
        result_query_line = cls._result_query_line(payload, warnings)
        if result_query_line is not None:
            lines.append(result_query_line)
        return "\n".join(lines)

    @staticmethod
    def _next_page_line(payload: Mapping[str, JsonValue]) -> str | None:
        truncated_count = PlateFileQueryRenderer._int_value(
            payload.get("truncated_count")
        )
        if truncated_count is None or truncated_count <= 0:
            return None
        offset = PlateFileQueryRenderer._int_value(payload.get("offset"))
        returned_count = PlateFileQueryRenderer._int_value(
            payload.get("returned_count")
        )
        limit = PlateFileQueryRenderer._int_value(payload.get("limit"))
        if offset is None or returned_count is None or limit is None:
            return None
        return (
            "Next page: rerun with "
            f"--offset {offset + returned_count} --limit {limit}"
        )

    @staticmethod
    def _record_root_summary_line(
        plate_path: JsonValue,
        records: tuple[Mapping[str, JsonValue], ...],
    ) -> str | None:
        if not isinstance(plate_path, str) or not plate_path:
            return None
        query_root = Path(plate_path)
        record_roots: list[str] = []
        for record in records:
            root = PlateFileQueryRenderer._record_root(record)
            if root is None:
                continue
            if root == query_root:
                continue
            root_text = str(root)
            if root_text not in record_roots:
                record_roots.append(root_text)
        if not record_roots:
            return None
        displayed_roots = ", ".join(record_roots[:3])
        omitted_count = len(record_roots) - 3
        if omitted_count > 0:
            displayed_roots = f"{displayed_roots}, ... (+{omitted_count})"
        return (
            f"Record file roots: {displayed_roots} "
            "(differs from query root; inventory may expose materialized outputs)"
        )

    @staticmethod
    def _record_root(record: Mapping[str, JsonValue]) -> Path | None:
        full_path = record.get("full_path")
        relative_path = record.get("relative_path")
        if not isinstance(full_path, str) or not isinstance(relative_path, str):
            return None
        if not full_path or not relative_path:
            return None
        root = Path(full_path)
        for part in Path(relative_path).parts:
            if part in ("", "."):
                continue
            root = root.parent
        return root

    @staticmethod
    def _int_value(value: JsonValue) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        return None

    @staticmethod
    def _result_query_line(
        payload: Mapping[str, JsonValue],
        warnings: tuple[Mapping[str, JsonValue], ...],
    ) -> str | None:
        warning_codes = {warning.get("code") for warning in warnings}
        if PlateInspectionIssueCode.RESULT_FILES_AVAILABLE.value not in warning_codes:
            return None
        plate_path = payload.get("plate_path")
        if not isinstance(plate_path, str) or not plate_path:
            return None
        microscope_type = payload.get("detected_microscope_type")
        microscope_option = (
            f" --microscope-type {microscope_type}"
            if isinstance(microscope_type, str) and microscope_type
            else ""
        )
        return (
            f"Next: query-plate-files {plate_path}{microscope_option} "
            "--kind result --include-previews"
        )

    @staticmethod
    def _record_lines(records: tuple[Mapping[str, JsonValue], ...]) -> list[str]:
        lines: list[str] = []
        for record in records:
            kind = McpDevPayloadProjection.text(record.get("kind"))
            key = McpDevPayloadProjection.text(record.get("key"))
            source_path = record.get("source_path")
            full_path = record.get("full_path")
            file_format = record.get("file_format")
            if source_path is not None:
                lines.append(
                    f"- {kind} {key} -> {McpDevPayloadProjection.text(source_path)}"
                    f"{PlateFileQueryRenderer._metadata_suffix(record)}"
                )
            elif full_path is not None:
                lines.append(
                    "- "
                    f"{kind} {key} "
                    f"type={McpDevPayloadProjection.text(file_format)} "
                    f"-> {McpDevPayloadProjection.text(full_path)}"
                    f"{PlateFileQueryRenderer._metadata_suffix(record)}"
                )
                preview = McpDevPayloadProjection.nested_mapping(record, "preview")
                lines.extend(PlateInspectionRenderer._result_preview_lines(preview))
            else:
                lines.append(
                    f"- {kind} {key}"
                    f"{PlateFileQueryRenderer._metadata_suffix(record)}"
                )
        return lines

    @staticmethod
    def _metadata_suffix(record: Mapping[str, JsonValue]) -> str:
        metadata = McpDevPayloadProjection.nested_mapping(record, "metadata")
        if not metadata:
            return ""
        parts: list[str] = []
        modified = metadata.get("modified")
        if modified is not None:
            parts.append(f"modified={McpDevPayloadProjection.text(modified)}")
        size = metadata.get("size")
        if size is not None:
            parts.append(f"size={McpDevPayloadProjection.text(size)}")
        if not parts:
            return ""
        return " " + " ".join(parts)

    @classmethod
    def _stale_result_warning_line(
        cls,
        records: tuple[Mapping[str, JsonValue], ...],
    ) -> str | None:
        image_modified = cls._modified_values_for_kind(records, "image")
        result_modified = cls._modified_values_for_kind(records, "result")
        if not image_modified or not result_modified:
            return None
        latest_image = max(image_modified)
        older_result_count = sum(
            1 for modified in result_modified if modified < latest_image
        )
        if older_result_count == 0:
            return None
        return (
            "Potential stale results: "
            f"{older_result_count} result artifact(s) are older than the latest "
            f"image record ({latest_image}); confirm they belong to the current "
            "pipeline/run before using them."
        )

    @classmethod
    def records_modified_summary_line(
        cls,
        label: str,
        records: tuple[Mapping[str, JsonValue], ...],
    ) -> str | None:
        modified_values = cls._modified_values(records)
        if not modified_values:
            return None
        distinct_values = tuple(sorted(set(modified_values)))
        latest = distinct_values[-1]
        earliest = distinct_values[0]
        if len(distinct_values) == 1:
            return f"{label}: {latest}"
        older_record_count = sum(1 for value in modified_values if value != latest)
        return (
            f"{label}: mixed latest={latest} earliest={earliest} "
            f"distinct={len(distinct_values)} older_records={older_record_count}"
        )

    @staticmethod
    def _modified_values(
        records: tuple[Mapping[str, JsonValue], ...],
    ) -> tuple[str, ...]:
        values: list[str] = []
        for record in records:
            metadata = McpDevPayloadProjection.nested_mapping(record, "metadata")
            modified = metadata.get("modified")
            if modified is not None:
                values.append(McpDevPayloadProjection.text(modified))
        return tuple(values)

    @classmethod
    def _modified_values_for_kind(
        cls,
        records: tuple[Mapping[str, JsonValue], ...],
        kind: str,
    ) -> tuple[str, ...]:
        matching_records = tuple(
            record
            for record in records
            if McpDevPayloadProjection.text(record.get("kind")) == kind
        )
        return cls._modified_values(matching_records)

class PlateFileStreamRenderer(McpDevOutputRenderer):
    """Compact renderer for plate file stream results."""

    output_contract = PlateFileStreamResult

    MAX_PATH_LINES: ClassVar[int] = 8
    MAX_STATUS_LINES: ClassVar[int] = 5

    @classmethod
    def render(cls, response: JsonObject) -> str:
        payload = McpDevPayloadProjection.first_tool_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)
        return cls.render_payload(payload)

    @classmethod
    def render_payload(cls, payload: Mapping[str, JsonValue]) -> str:
        errors = McpDevPayloadProjection.sequence_of_mappings(payload.get("errors"))
        if errors:
            return "\n".join(
                (
                    "Plate file stream: failed",
                    *PlateInspectionRenderer._error_lines(errors),
                )
            )

        resolved = McpDevPayloadProjection.sequence_of_mappings(
            payload.get("resolved_records")
        )
        skipped = McpDevPayloadProjection.sequence_of_mappings(
            payload.get("skipped_records")
        )
        image_paths = cls._text_sequence(payload.get("streamed_image_paths"))
        roi_paths = cls._text_sequence(payload.get("streamed_roi_paths"))
        requested_paths = cls._text_sequence(payload.get("requested_paths"))
        connection = McpDevPayloadProjection.nested_mapping(payload, "connection")
        lines = [
            f"Plate file stream: {McpDevPayloadProjection.text(payload.get('plate_path'))}",
            (
                "Viewer: "
                f"{McpDevPayloadProjection.text(payload.get('viewer_type'))} "
                f"config={McpDevPayloadProjection.text(payload.get('viewer_config_key'))} "
                f"host={McpDevPayloadProjection.text(connection.get('host'))} "
                f"port={McpDevPayloadProjection.text(connection.get('port'))} "
                f"transport={McpDevPayloadProjection.text(connection.get('transport_mode'))} "
                f"persistent={McpDevPayloadProjection.text(connection.get('persistent'))}"
            ),
            (
                "Files: "
                f"requested={len(requested_paths)} "
                f"resolved={len(resolved)} "
                f"images={len(image_paths)} "
                f"rois={len(roi_paths)} "
                f"skipped={len(skipped)}"
            ),
            (
                "Handler: "
                f"{McpDevPayloadProjection.text(payload.get('handler_class'))} "
                f"parser={McpDevPayloadProjection.text(payload.get('parser_class'))}"
            ),
        ]
        cls._append_path_lines(lines, "Images", image_paths)
        cls._append_path_lines(lines, "ROIs", roi_paths)
        if skipped:
            lines.append("Skipped:")
            lines.extend(PlateFileQueryRenderer._record_lines(skipped[: cls.MAX_PATH_LINES]))
            if len(skipped) > cls.MAX_PATH_LINES:
                lines.append(f"- ... {len(skipped) - cls.MAX_PATH_LINES} more")
        status_messages = cls._text_sequence(payload.get("status_messages"))
        if status_messages:
            lines.append("Status:")
            lines.extend(f"- {message}" for message in status_messages[: cls.MAX_STATUS_LINES])
            if len(status_messages) > cls.MAX_STATUS_LINES:
                lines.append(f"- ... {len(status_messages) - cls.MAX_STATUS_LINES} more")
        warnings = McpDevPayloadProjection.sequence_of_mappings(payload.get("warnings"))
        if warnings:
            lines.append("Warnings:")
            lines.extend(PlateInspectionRenderer._error_lines(warnings))
        cls._append_next_lines(lines, connection, has_rois=bool(roi_paths))
        return "\n".join(lines)

    @classmethod
    def _append_path_lines(
        cls,
        lines: list[str],
        heading: str,
        paths: tuple[str, ...],
    ) -> None:
        if not paths:
            return
        lines.append(f"{heading}:")
        lines.extend(f"- {path}" for path in paths[: cls.MAX_PATH_LINES])
        if len(paths) > cls.MAX_PATH_LINES:
            lines.append(f"- ... {len(paths) - cls.MAX_PATH_LINES} more")

    @classmethod
    def _append_next_lines(
        cls,
        lines: list[str],
        connection: Mapping[str, JsonValue],
        *,
        has_rois: bool,
    ) -> None:
        port = connection.get("port")
        if port is None:
            return
        options = cls._viewer_command_options(connection)
        lines.append("Next:")
        lines.append(f"- validate-viewer {options} --require-nonzero-payloads")
        lines.append(f"- viewer-state {options}")
        if has_rois:
            lines.append(f"- viewer-rois {options} --limit 5")

    @staticmethod
    def _viewer_command_options(connection: Mapping[str, JsonValue]) -> str:
        parts = [f"--port {McpDevPayloadProjection.text(connection.get('port'))}"]
        host = connection.get("host")
        if host not in (None, "", "localhost"):
            parts.append(f"--host {McpDevPayloadProjection.text(host)}")
        transport_mode = connection.get("transport_mode")
        if transport_mode not in (None, ""):
            parts.append(
                f"--transport-mode {McpDevPayloadProjection.text(transport_mode)}"
            )
        return " ".join(parts)

    @staticmethod
    def _text_sequence(value: JsonValue) -> tuple[str, ...]:
        if not isinstance(value, list):
            return ()
        return tuple(McpDevPayloadProjection.text(item) for item in value)

class SelectedPlateImagesRenderer(McpDevOutputRenderer):
    """Compact renderer for selected-plate image inspection."""

    output_contract = SelectedPlateImageInspectionResult

    @classmethod
    def render(cls, response: JsonObject) -> str:
        payload = McpDevPayloadProjection.first_tool_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)
        errors = McpDevPayloadProjection.sequence_of_mappings(payload.get("errors"))
        if errors:
            return "\n".join(
                ("Selected plate images: failed", *PlateInspectionRenderer._error_lines(errors))
            )

        selected_plate = McpDevPayloadProjection.nested_mapping(
            payload,
            "selected_plate",
        )
        target = payload.get("target") or SelectedPlateFileQueryTarget.SELECTED.value
        inspection = McpDevPayloadProjection.nested_mapping(payload, "inspection")
        lines = [
            (
                "Selected plate: "
                f"{McpDevPayloadProjection.text(selected_plate.get('name'))} "
                f"root={McpDevPayloadProjection.text(selected_plate.get('plate_root'))} "
                f"target={McpDevPayloadProjection.text(target)}"
            )
        ]
        if not inspection:
            lines.append("Inspection: <none>")
            return "\n".join(lines)
        lines.append(
            PlateInspectionRenderer.render(
                {
                    "results": [
                        {
                            "tool": agent_capabilities.inspect_plate_path.name,
                            "mcp_error": False,
                            "payloads": [inspection],
                        }
                    ]
                },
                next_sample_command="selected-plate-sample",
                next_sample_prefix=(
                    ""
                    if target == SelectedPlateFileQueryTarget.SELECTED.value
                    else f"--target {target} "
                ),
            )
        )
        return "\n".join(lines)

class SelectedPlateFilesRenderer(McpDevOutputRenderer):
    """Compact renderer for selected-plate file queries."""

    output_contract = SelectedPlateFileQueryResult

    @classmethod
    def render(cls, response: JsonObject) -> str:
        payload = McpDevPayloadProjection.first_tool_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)
        errors = McpDevPayloadProjection.sequence_of_mappings(payload.get("errors"))
        if errors:
            return "\n".join(
                ("Selected plate files: failed", *PlateInspectionRenderer._error_lines(errors))
            )

        selected_plate = McpDevPayloadProjection.nested_mapping(
            payload,
            "selected_plate",
        )
        query = McpDevPayloadProjection.nested_mapping(payload, "query")
        target = McpDevPayloadProjection.text(payload.get("target"))
        lines = [
            (
                "Selected plate: "
                f"{McpDevPayloadProjection.text(selected_plate.get('name'))} "
                f"root={McpDevPayloadProjection.text(selected_plate.get('plate_root'))} "
                f"target={target}"
            )
        ]
        if not query:
            lines.append("Query: <none>")
            return "\n".join(lines)
        lines.append(
            PlateFileQueryRenderer.render(
                {
                    "results": [
                        {
                            "tool": agent_capabilities.query_plate_files.name,
                            "mcp_error": False,
                            "payloads": [query],
                        }
                    ]
                }
            )
        )
        if cls._should_suggest_output(selected_plate, query, target):
            output_root = McpDevPayloadProjection.text(
                selected_plate.get("output_plate_root")
            )
            lines.append(f"Related output: {output_root}")
            lines.append("Next: selected-plate-files --target output --kind result")
        return "\n".join(lines)

    @staticmethod
    def _should_suggest_output(
        selected_plate: Mapping[str, JsonValue],
        query: Mapping[str, JsonValue],
        target: str,
    ) -> bool:
        if target != SelectedPlateFileQueryTarget.SELECTED.value:
            return False
        if query.get("total_count") != 0:
            return False
        if selected_plate.get("output_plate_root") in (None, ""):
            return False
        return query.get("plate_path") == selected_plate.get("plate_root")

class SelectedPlateSampleRenderer(McpDevOutputRenderer):
    """Compact renderer for selected-plate image sampling."""

    output_contract = SelectedPlateImageSampleResult

    @classmethod
    def render(cls, response: JsonObject) -> str:
        payload = McpDevPayloadProjection.first_tool_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)
        selected_plate = McpDevPayloadProjection.nested_mapping(
            payload,
            "selected_plate",
        )
        target = payload.get("target") or SelectedPlateFileQueryTarget.SELECTED.value
        errors = McpDevPayloadProjection.sequence_of_mappings(payload.get("errors"))
        if errors:
            return "\n".join(
                (
                    "Selected plate sample: failed",
                    (
                        "Selected plate: "
                        f"{McpDevPayloadProjection.text(selected_plate.get('name'))} "
                        f"root={McpDevPayloadProjection.text(selected_plate.get('plate_root'))} "
                        f"target={McpDevPayloadProjection.text(target)}"
                    ),
                    *PlateImageSampleRenderer._error_lines(list(errors)),
                )
            )

        sample = McpDevPayloadProjection.nested_mapping(payload, "sample")
        lines = [
            (
                "Selected plate: "
                f"{McpDevPayloadProjection.text(selected_plate.get('name'))} "
                f"root={McpDevPayloadProjection.text(selected_plate.get('plate_root'))} "
                f"target={McpDevPayloadProjection.text(target)}"
            ),
            (
                "Selected image: "
                f"{McpDevPayloadProjection.text(payload.get('image_path'))} "
                f"auto={McpDevPayloadProjection.text(payload.get('auto_selected_image_path'))}"
            ),
        ]
        if not sample:
            lines.append("Sample: <none>")
            return "\n".join(lines)
        lines.append(
            PlateImageSampleRenderer.render(
                {
                    "results": [
                        {
                            "tool": agent_capabilities.sample_plate_image.name,
                            "mcp_error": False,
                            "payloads": [sample],
                        }
                    ]
                }
            )
        )
        return "\n".join(lines)

class SelectedPlateStreamRenderer(McpDevOutputRenderer):
    """Compact renderer for selected-plate file streaming."""

    output_contract = SelectedPlateFileStreamResult

    @classmethod
    def render(cls, response: JsonObject) -> str:
        payload = McpDevPayloadProjection.first_tool_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)
        selected_plate = McpDevPayloadProjection.nested_mapping(
            payload,
            "selected_plate",
        )
        target = payload.get("target") or SelectedPlateFileQueryTarget.SELECTED.value
        errors = McpDevPayloadProjection.sequence_of_mappings(payload.get("errors"))
        if errors and not payload.get("stream"):
            return "\n".join(
                (
                    "Selected plate stream: failed",
                    (
                        "Selected plate: "
                        f"{McpDevPayloadProjection.text(selected_plate.get('name'))} "
                        f"root={McpDevPayloadProjection.text(selected_plate.get('plate_root'))} "
                        f"target={McpDevPayloadProjection.text(target)}"
                    ),
                    *PlateInspectionRenderer._error_lines(errors),
                )
            )
        stream = McpDevPayloadProjection.nested_mapping(payload, "stream")
        lines = [
            (
                "Selected plate: "
                f"{McpDevPayloadProjection.text(selected_plate.get('name'))} "
                f"root={McpDevPayloadProjection.text(selected_plate.get('plate_root'))} "
                f"target={McpDevPayloadProjection.text(target)}"
            )
        ]
        if not stream:
            lines.append("Stream: <none>")
            return "\n".join(lines)
        lines.append(PlateFileStreamRenderer.render_payload(stream))
        return "\n".join(lines)
