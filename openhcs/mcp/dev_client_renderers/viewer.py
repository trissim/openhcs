"""Viewer, runtime-server, and snapshot renderers for the MCP dev client."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from openhcs.agent.dto.common import JsonObject, JsonValue
from openhcs.agent.dto.execution import (
    RuntimeDebugInspectionResult,
    RuntimeExecutionStatus,
    RuntimeServerInfo,
    RuntimeServerScanResult,
)
from openhcs.agent.dto.ui_bridge import UiWindowSnapshotResult
from openhcs.agent.dto.viewer import (
    ViewerWindowImageSampleResult,
    ViewerWindowLayerIsolationResult,
    ViewerWindowNavigationResult,
    ViewerWindowPayloadResult,
    ViewerWindowProbeResult,
    ViewerWindowRoiSummaryResult,
    ViewerWindowSnapshotResult,
    ViewerWindowStateResult,
    ViewerWindowValidationSummaryResult,
)
from openhcs.mcp.dev_client_rendering import (
    CatalogRenderOptions,
    McpDevOutputRenderer,
    McpDevOutputRendererBinding,
    McpDevPayloadProjection,
    McpDiagnosticRenderer,
    ViewerImageSampleRenderOptions,
)

class ViewerValidationRenderer(McpDevOutputRenderer):
    """Compact renderer for viewer validation summaries."""

    output_contract = ViewerWindowValidationSummaryResult

    @classmethod
    def render(cls, response: JsonObject) -> str:
        payload = McpDevPayloadProjection.first_tool_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)
        errors = McpDevPayloadProjection.sequence_of_mappings(payload.get("errors"))
        if "valid" not in payload and errors:
            return "\n".join(("Viewer validation: failed", *cls._error_lines(errors)))
        policy = McpDevPayloadProjection.nested_mapping(payload, "validation_policy")
        connection = McpDevPayloadProjection.nested_mapping(payload, "connection")
        lines = [
            (
                "Viewer validation: "
                f"valid={McpDevPayloadProjection.text(payload.get('valid'))} "
                f"observed={McpDevPayloadProjection.text(payload.get('observed'))} "
                f"layers={McpDevPayloadProjection.text(payload.get('layer_count'))} "
                f"mounted={McpDevPayloadProjection.text(payload.get('mounted_layer_count'))} "
                f"pending={McpDevPayloadProjection.text(payload.get('pending_update_count'))}"
            ),
            (
                "Payloads: "
                f"total={McpDevPayloadProjection.text(payload.get('payload_count'))} "
                f"nonzero={McpDevPayloadProjection.text(payload.get('nonzero_payload_count'))} "
                f"zero={McpDevPayloadProjection.text(payload.get('zero_payload_count'))} "
                f"missing={McpDevPayloadProjection.text(payload.get('missing_payload_coordinate_count'))} "
                f"duplicates={McpDevPayloadProjection.text(payload.get('duplicate_payload_coordinate_count'))}"
            ),
            (
                "Policy: "
                f"expected_layers={McpDevPayloadProjection.text(policy.get('expected_layer_count'))} "
                f"required_axes={cls._sequence_text(policy.get('required_axis_labels'))} "
                "required_components="
                f"{cls._sequence_text(policy.get('required_component_labels'))} "
                f"require_nonzero={McpDevPayloadProjection.text(policy.get('require_nonzero_payloads'))}"
            ),
            (
                "Connection: "
                f"{McpDevPayloadProjection.text(connection.get('host'))}:"
                f"{McpDevPayloadProjection.text(connection.get('port'))} "
                f"transport={McpDevPayloadProjection.text(connection.get('transport_mode'))}"
            ),
        ]
        if errors:
            lines.append("Errors:")
            lines.extend(cls._error_lines(errors))
        warnings = McpDevPayloadProjection.sequence_of_mappings(payload.get("warnings"))
        if warnings:
            lines.append("Warnings:")
            lines.extend(cls._error_lines(warnings))
        layers = McpDevPayloadProjection.sequence_of_mappings(
            payload.get("layer_summaries")
        )
        if layers:
            lines.append("Layers:")
            lines.extend(cls._layer_lines(layers))
        return "\n".join(lines)

    @classmethod
    def _layer_lines(
        cls,
        layers: tuple[Mapping[str, JsonValue], ...],
    ) -> list[str]:
        lines: list[str] = []
        for layer in layers:
            lines.append(
                "- "
                f"{McpDevPayloadProjection.text(layer.get('route_key'))}: "
                f"valid={McpDevPayloadProjection.text(layer.get('valid'))} "
                f"mounted={McpDevPayloadProjection.text(layer.get('mounted'))} "
                f"items={McpDevPayloadProjection.text(layer.get('item_count'))} "
                f"axes={cls._sequence_text(layer.get('axis_labels'))} "
                f"stack={cls._sequence_text(layer.get('stack_axes'))} "
                f"payloads={McpDevPayloadProjection.text(layer.get('payload_count'))} "
                f"nonzero={McpDevPayloadProjection.text(layer.get('nonzero_payload_count'))} "
                f"gaps={McpDevPayloadProjection.text(layer.get('coordinate_gap_count'))} "
                f"missing_axes={cls._sequence_text(layer.get('missing_required_axis_labels'))} "
                "components="
                f"{cls._sequence_text(layer.get('component_labels'))} "
                "missing_components="
                f"{cls._sequence_text(layer.get('missing_required_component_labels'))} "
                "axis_as_components="
                f"{cls._sequence_text(layer.get('axis_labels_present_as_components'))} "
                f"title={McpDevPayloadProjection.quoted_text(layer.get('title'))}"
            )
        return lines

    @staticmethod
    def _sequence_text(value: JsonValue) -> str:
        if not isinstance(value, list):
            return "<none>" if value is None else str(value)
        if not value:
            return "<none>"
        return ",".join(str(item) for item in value)

    @staticmethod
    def _error_lines(errors: tuple[Mapping[str, JsonValue], ...]) -> tuple[str, ...]:
        return McpDiagnosticRenderer.error_lines(errors)


class ViewerStateRenderer(McpDevOutputRenderer):
    """Compact renderer for viewer state and layer component metadata."""

    output_contract = ViewerWindowStateResult

    @classmethod
    def render(cls, response: JsonObject) -> str:
        payload = McpDevPayloadProjection.first_tool_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)
        errors = McpDevPayloadProjection.sequence_of_mappings(payload.get("errors"))
        if "observed" not in payload and errors:
            return "\n".join(("Viewer state: failed", *ViewerValidationRenderer._error_lines(errors)))
        viewer = McpDevPayloadProjection.nested_mapping(payload, "viewer")
        lines = [
            (
                "Viewer state: "
                f"observed={McpDevPayloadProjection.text(payload.get('observed'))} "
                f"type={McpDevPayloadProjection.text(viewer.get('viewer_type'))} "
                f"title={McpDevPayloadProjection.quoted_text(viewer.get('title'))}"
            ),
            (
                "Window: "
                f"layers={McpDevPayloadProjection.text(payload.get('layer_count'))} "
                f"ndim={McpDevPayloadProjection.text(payload.get('viewer_ndim'))} "
                f"axes={ViewerValidationRenderer._sequence_text(payload.get('axis_labels'))} "
                f"current_step={cls._json_summary(payload.get('current_step'))} "
                f"active_route={McpDevPayloadProjection.text(payload.get('active_dimension_label_route'))}"
            ),
            (
                "Components: "
                f"groups={McpDevPayloadProjection.text(payload.get('component_group_count'))} "
                f"items={McpDevPayloadProjection.text(payload.get('component_item_count'))}"
            ),
        ]
        if errors:
            lines.append("Errors:")
            lines.extend(ViewerValidationRenderer._error_lines(errors))
        warnings = McpDevPayloadProjection.sequence_of_mappings(payload.get("warnings"))
        if warnings:
            lines.append("Warnings:")
            lines.extend(ViewerValidationRenderer._error_lines(warnings))
        layers = McpDevPayloadProjection.sequence_of_mappings(payload.get("layers"))
        if layers:
            lines.append("Layers:")
            lines.extend(cls._layer_lines(layers))
        return "\n".join(lines)

    @classmethod
    def _layer_lines(
        cls,
        layers: tuple[Mapping[str, JsonValue], ...],
    ) -> list[str]:
        lines: list[str] = []
        for layer in layers:
            lines.append(
                "- "
                f"{McpDevPayloadProjection.text(layer.get('route_key'))}: "
                f"title={McpDevPayloadProjection.quoted_text(layer.get('title'))} "
                f"visible={McpDevPayloadProjection.text(layer.get('visible'))} "
                f"selected={McpDevPayloadProjection.text(layer.get('selected'))} "
                f"items={McpDevPayloadProjection.text(layer.get('item_count'))} "
                f"types={ViewerValidationRenderer._sequence_text(layer.get('data_types'))} "
                f"axes={ViewerValidationRenderer._sequence_text(layer.get('axis_labels'))} "
                f"stack={ViewerValidationRenderer._sequence_text(layer.get('stack_axes'))} "
                f"shape={cls._json_summary(layer.get('data_shape'))}"
            )
            lines.append(
                "  components: "
                f"{cls._component_values_text(layer.get('component_values'))}"
            )
            axis_values = McpDevPayloadProjection.nested_mapping(
                layer,
                "axis_component_values",
            )
            routed_values = McpDevPayloadProjection.nested_mapping(
                layer,
                "routed_component_values",
            )
            if axis_values:
                lines.append(f"  axis values: {cls._mapping_text(axis_values)}")
            if routed_values:
                lines.append(f"  routed values: {cls._mapping_text(routed_values)}")
            payloads = McpDevPayloadProjection.sequence_of_mappings(
                layer.get("payload_summaries")
            )
            if payloads:
                lines.append(
                    "  payload summaries: "
                    f"{McpDevPayloadProjection.text(layer.get('payload_summary_count'))} "
                    f"truncated={McpDevPayloadProjection.text(layer.get('payload_summaries_truncated'))}"
                )
                for payload in payloads[:3]:
                    lines.append(
                        "  payload "
                        f"type={McpDevPayloadProjection.text(payload.get('data_type'))} "
                        f"shape={cls._json_summary(payload.get('shape'))} "
                        f"dtype={McpDevPayloadProjection.text(payload.get('dtype'))} "
                        f"min={McpDevPayloadProjection.text(payload.get('min'))} "
                        f"max={McpDevPayloadProjection.text(payload.get('max'))} "
                        f"components={cls._mapping_text(McpDevPayloadProjection.nested_mapping(payload, 'components'))} "
                        f"path={ViewerPayloadPathRenderer.render(payload.get('path'))}"
                    )
        return lines

    @classmethod
    def _component_values_text(cls, value: JsonValue) -> str:
        if not isinstance(value, list) or not value:
            return "<none>"
        merged: dict[str, list[str]] = {}
        for item in value:
            if not isinstance(item, Mapping):
                continue
            for key, component_value in item.items():
                values = merged.setdefault(str(key), [])
                text = McpDevPayloadProjection.text(component_value)
                if text not in values:
                    values.append(text)
        if not merged:
            return "<none>"
        return ", ".join(
            f"{key}={','.join(values)}" for key, values in sorted(merged.items())
        )

    @staticmethod
    def _mapping_text(value: Mapping[str, JsonValue]) -> str:
        if not value:
            return "<none>"
        parts: list[str] = []
        for key in sorted(str(key) for key in value):
            item = value.get(key)
            if isinstance(item, list):
                item_text = ",".join(McpDevPayloadProjection.text(part) for part in item)
            else:
                item_text = McpDevPayloadProjection.text(item)
            parts.append(f"{key}={item_text}")
        return ", ".join(parts)

    @staticmethod
    def _json_summary(value: JsonValue) -> str:
        if value is None:
            return "<none>"
        return json.dumps(value, sort_keys=True)


class ViewerPayloadRenderer(McpDevOutputRenderer):
    """Compact renderer for viewer payload inspection records."""

    output_contract = ViewerWindowPayloadResult

    @classmethod
    def render(cls, response: JsonObject) -> str:
        payload = McpDevPayloadProjection.first_tool_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)
        errors = McpDevPayloadProjection.sequence_of_mappings(payload.get("errors"))
        if "observed" not in payload and errors:
            return "\n".join(
                ("Viewer payloads: failed", *ViewerValidationRenderer._error_lines(errors))
            )
        lines = [
            (
                "Viewer payloads: "
                f"observed={McpDevPayloadProjection.text(payload.get('observed'))} "
                f"layers={McpDevPayloadProjection.text(payload.get('layer_count'))}"
            )
        ]
        if errors:
            lines.append("Errors:")
            lines.extend(ViewerValidationRenderer._error_lines(errors))
        warnings = McpDevPayloadProjection.sequence_of_mappings(payload.get("warnings"))
        if warnings:
            lines.append("Warnings:")
            lines.extend(ViewerValidationRenderer._error_lines(warnings))
        layers = McpDevPayloadProjection.sequence_of_mappings(payload.get("layers"))
        if layers:
            lines.append("Layers:")
            lines.extend(cls._layer_lines(layers))
        return "\n".join(lines)

    @classmethod
    def _layer_lines(
        cls,
        layers: tuple[Mapping[str, JsonValue], ...],
    ) -> list[str]:
        lines: list[str] = []
        for layer in layers:
            payloads = McpDevPayloadProjection.sequence_of_mappings(
                layer.get("payloads")
            )
            lines.append(
                "- "
                f"{McpDevPayloadProjection.text(layer.get('route_key'))}: "
                f"title={McpDevPayloadProjection.quoted_text(layer.get('title'))} "
                f"mounted={McpDevPayloadProjection.text(layer.get('mounted'))} "
                f"items={McpDevPayloadProjection.text(layer.get('item_count'))} "
                f"axes={ViewerValidationRenderer._sequence_text(layer.get('axis_labels'))} "
                f"stack={ViewerValidationRenderer._sequence_text(layer.get('stack_axes'))} "
                f"payloads={len(payloads)} "
                f"pending={McpDevPayloadProjection.text(layer.get('pending_update'))}"
            )
            for payload in payloads[:5]:
                lines.append(cls._payload_line(payload))
            if len(payloads) > 5:
                lines.append(f"  ...<truncated {len(payloads) - 5} payloads>")
        return lines

    @classmethod
    def _payload_line(cls, payload: Mapping[str, JsonValue]) -> str:
        summary = McpDevPayloadProjection.nested_mapping(payload, "summary")
        array_summary = McpDevPayloadProjection.nested_mapping(
            payload,
            "array_value_summary",
        )
        array_text = ""
        if array_summary:
            array_text = (
                " array="
                f"included={McpDevPayloadProjection.text(array_summary.get('included'))}"
                f":shape={ViewerStateRenderer._json_summary(array_summary.get('shape'))}"
            )
            omitted_reason = array_summary.get("omitted_reason")
            if omitted_reason is not None:
                array_text += f":reason={McpDevPayloadProjection.text(omitted_reason)}"
        return (
            "  payload "
            f"type={McpDevPayloadProjection.text(payload.get('data_type'))} "
            f"axis={ViewerAxisIndexRenderer.render(payload.get('axis_indices'))} "
            f"aggregate_axis={ViewerAxisIndexRenderer.render(payload.get('aggregate_axis_indices'))} "
            f"components={ViewerStateRenderer._mapping_text(McpDevPayloadProjection.nested_mapping(payload, 'components'))} "
            f"shape={ViewerStateRenderer._json_summary(summary.get('shape'))} "
            f"dtype={McpDevPayloadProjection.text(summary.get('dtype'))} "
            f"{cls._count_text(payload, summary)}"
            f"{array_text} "
            f"path={ViewerPayloadPathRenderer.render(payload.get('path'))}"
        )

    @staticmethod
    def _count_text(
        payload: Mapping[str, JsonValue],
        summary: Mapping[str, JsonValue],
    ) -> str:
        if payload.get("data_type") != "shapes":
            return f"nonzero={McpDevPayloadProjection.text(summary.get('nonzero_count'))}"
        shape_payload_count = summary.get("shape_payload_count")
        if shape_payload_count is None:
            shape_payload_count = summary.get("nonzero_count")
        shape_payloads = McpDevPayloadProjection.sequence_of_mappings(
            payload.get("shape_payloads")
        )
        return (
            f"shape_members={McpDevPayloadProjection.text(shape_payload_count)} "
            f"returned_shapes={len(shape_payloads)} "
            "semantic_rois=use-viewer-rois"
        )


class ViewerPayloadPathRenderer:
    """Render viewer payload paths without implying virtual paths exist on disk."""

    @staticmethod
    def render(value: JsonValue) -> str:
        text = McpDevPayloadProjection.text(value)
        if not isinstance(value, str) or not value:
            return text
        path = Path(value)
        if not path.is_absolute() or path.exists():
            return text
        return f"{text} (streamed/non-materialized)"


class ViewerAxisIndexRenderer:
    """Render viewer payload axis selectors without Python container reprs."""

    @staticmethod
    def render(value: JsonValue) -> str:
        if isinstance(value, Mapping):
            if not value:
                return "<none>"
            return ", ".join(
                f"{key}={McpDevPayloadProjection.text(value.get(key))}"
                for key in sorted(str(key) for key in value)
            )
        return ViewerValidationRenderer._sequence_text(value)


class ViewerRoiSummaryRenderer(McpDevOutputRenderer):
    """Compact renderer for viewer ROI summaries."""

    output_contract = ViewerWindowRoiSummaryResult

    @classmethod
    def render(cls, response: JsonObject) -> str:
        payload = McpDevPayloadProjection.first_tool_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)
        lines = [
            (
                "Viewer ROIs: "
                f"observed={McpDevPayloadProjection.text(payload.get('observed'))} "
                f"route={McpDevPayloadProjection.text(payload.get('route_key'))} "
                f"axis={ViewerAxisIndexRenderer.render(payload.get('axis_indices'))} "
                f"layers={McpDevPayloadProjection.text(payload.get('layer_count'))} "
                f"records={McpDevPayloadProjection.text(payload.get('payload_record_count'))} "
                f"payloads={McpDevPayloadProjection.text(payload.get('roi_payload_count'))}"
            ),
            (
                "ROIs: "
                f"total={McpDevPayloadProjection.text(payload.get('total_roi_count'))} "
                f"returned={McpDevPayloadProjection.text(payload.get('returned_roi_count'))} "
                f"exact={McpDevPayloadProjection.text(payload.get('roi_count_exact'))} "
                f"members={McpDevPayloadProjection.text(payload.get('total_roi_member_count'))}/"
                f"{McpDevPayloadProjection.text(payload.get('returned_roi_member_count'))} "
                f"truncated={McpDevPayloadProjection.text(payload.get('roi_payloads_truncated'))}"
            ),
        ]
        errors = McpDevPayloadProjection.sequence_of_mappings(payload.get("errors"))
        if errors:
            lines.append("Errors:")
            lines.extend(ViewerValidationRenderer._error_lines(errors))
        warnings = McpDevPayloadProjection.sequence_of_mappings(payload.get("warnings"))
        if warnings:
            lines.append("Warnings:")
            lines.extend(ViewerValidationRenderer._error_lines(warnings))
        payload_type_counts = McpDevPayloadProjection.nested_mapping(
            payload,
            "payload_type_counts",
        )
        if payload_type_counts:
            lines.append(
                "Payload types: "
                f"{ViewerStateRenderer._mapping_text(payload_type_counts)}"
            )
        roi_payloads = McpDevPayloadProjection.sequence_of_mappings(
            payload.get("payloads")
        )
        if roi_payloads:
            lines.append("Payloads:")
            lines.extend(cls._payload_lines(roi_payloads))
        elif not errors:
            lines.extend(cls._no_roi_guidance(payload, payload_type_counts))
        return "\n".join(lines)

    @classmethod
    def _no_roi_guidance(
        cls,
        payload: Mapping[str, JsonValue],
        payload_type_counts: Mapping[str, JsonValue],
    ) -> list[str]:
        if cls._numeric_value(payload.get("total_roi_count")) not in (None, 0):
            return []
        route_key = payload.get("route_key")
        lines = [
            (
                "Interpretation: no ROI/shapes payloads were found for the "
                "requested viewer"
                + (" route." if route_key else ".")
            ),
            "Next:",
            "- Run `viewer-state <port>` to list layer route keys, layer types, and visible image layers.",
        ]
        if route_key:
            lines.append(
                "- Check that the selected route is a shapes/ROI layer, or stream ROI artifacts to the viewer."
            )
        else:
            lines.append(
                "- If `viewer-state` shows a shapes layer, rerun `viewer-rois <port> <route_key>` for that route."
            )
        if payload_type_counts:
            lines.append(
                "- If payload types are image-only, this pipeline/viewer state has no streamed ROI layer."
            )
        lines.append(
            "- To validate output artifacts, query selected output files and stream ROI files if they exist."
        )
        return lines

    @staticmethod
    def _numeric_value(value: JsonValue) -> int | float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return value
        return None

    @classmethod
    def _payload_lines(
        cls,
        roi_payloads: tuple[Mapping[str, JsonValue], ...],
    ) -> list[str]:
        lines: list[str] = []
        for roi_payload in roi_payloads:
            title = McpDevPayloadProjection.text(roi_payload.get("layer_title"))
            layer_route = McpDevPayloadProjection.text(
                roi_payload.get("layer_route_key")
                or roi_payload.get("payload_route_key")
            )
            payload_route = cls._route_summary(roi_payload.get("payload_route_key"))
            lines.append(
                "- "
                f"title={McpDevPayloadProjection.quoted_text(title)} "
                f"layer_route={layer_route} "
                f"payload_route={payload_route} "
                f"axis={ViewerAxisIndexRenderer.render(roi_payload.get('axis_indices'))} "
                f"components={ViewerStateRenderer._mapping_text(McpDevPayloadProjection.nested_mapping(roi_payload, 'components'))} "
                f"roi_count={McpDevPayloadProjection.text(roi_payload.get('roi_count'))} "
                f"returned={McpDevPayloadProjection.text(roi_payload.get('returned_roi_count'))} "
                f"exact={McpDevPayloadProjection.text(roi_payload.get('roi_count_exact'))} "
                f"members={McpDevPayloadProjection.text(roi_payload.get('roi_member_count'))}/"
                f"{McpDevPayloadProjection.text(roi_payload.get('returned_roi_member_count'))} "
                f"duplicate_members={McpDevPayloadProjection.text(roi_payload.get('roi_duplicate_member_count'))} "
                f"truncated={McpDevPayloadProjection.text(roi_payload.get('roi_payloads_truncated'))} "
                f"area={cls._stats_text(roi_payload.get('area'))} "
                f"perimeter={cls._stats_text(roi_payload.get('perimeter'))} "
                f"bounds={cls._json_summary(roi_payload.get('bounds_yx'))} "
                f"coords={McpDevPayloadProjection.text(roi_payload.get('coordinate_count'))} "
                f"source_origin={cls._json_summary(roi_payload.get('spatial_origin_yx'))} "
                f"source_shape={cls._json_summary(roi_payload.get('source_spatial_shape_yx'))} "
                f"out_of_bounds={McpDevPayloadProjection.text(roi_payload.get('out_of_source_bounds_count'))}"
            )
            examples = McpDevPayloadProjection.sequence_of_mappings(
                roi_payload.get("example_rois")
            )
            for example in examples[:2]:
                lines.append(
                    "  example "
                    f"label={McpDevPayloadProjection.text(example.get('label'))} "
                    f"area={McpDevPayloadProjection.text(example.get('area'))} "
                    f"centroid={cls._json_summary(example.get('centroid_yx'))} "
                    f"bbox={cls._json_summary(example.get('bbox_yxyx'))}"
                )
        return lines

    @staticmethod
    def _route_summary(value: JsonValue) -> str:
        text = McpDevPayloadProjection.text(value)
        if text == "<none>" or len(text) <= 80:
            return text
        separator_index = text.rfind("::")
        if separator_index >= 0:
            return f"...{text[separator_index:]}"
        return f"{text[:32]}...{text[-32:]}"

    @staticmethod
    def _sequence_text(value: JsonValue) -> str:
        return ViewerValidationRenderer._sequence_text(value)

    @staticmethod
    def _json_summary(value: JsonValue) -> str:
        if value is None:
            return "<none>"
        return json.dumps(value, sort_keys=True)

    @staticmethod
    def _stats_text(value: JsonValue) -> str:
        if not isinstance(value, Mapping):
            return "<none>"
        return (
            "min="
            f"{McpDevPayloadProjection.text(value.get('min'))},"
            "median="
            f"{McpDevPayloadProjection.text(value.get('median'))},"
            "mean="
            f"{McpDevPayloadProjection.text(value.get('mean'))},"
            "max="
            f"{McpDevPayloadProjection.text(value.get('max'))}"
        )


class ViewerImageSampleRenderer(McpDevOutputRenderer):
    """Compact renderer for viewer image samples."""

    output_contract = ViewerWindowImageSampleResult

    @classmethod
    def render_with_options(
        cls,
        response: JsonObject,
        options: ViewerImageSampleRenderOptions,
    ) -> str:
        return cls.render(
            response,
            include_array_values_requested=options.include_array_values_requested,
        )

    @classmethod
    def render(
        cls,
        response: JsonObject,
        *,
        include_array_values_requested: bool | None = None,
    ) -> str:
        payload = McpDevPayloadProjection.first_tool_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)
        lines = [
            (
                "Viewer image sample: "
                f"observed={McpDevPayloadProjection.text(payload.get('observed'))} "
                f"route={McpDevPayloadProjection.text(payload.get('route_key'))} "
                f"axis={ViewerAxisIndexRenderer.render(payload.get('axis_indices'))} "
                f"slices={cls._json_summary(payload.get('array_slices'))}"
            ),
            (
                "Records: "
                f"matched={McpDevPayloadProjection.text(payload.get('record_count'))} "
                f"returned={McpDevPayloadProjection.text(payload.get('returned_record_count'))} "
                f"truncated={McpDevPayloadProjection.text(payload.get('records_truncated_count'))} "
                f"image={McpDevPayloadProjection.text(payload.get('raw_image_record_count'))} "
                f"total_payloads={McpDevPayloadProjection.text(payload.get('total_payload_record_count'))} "
                f"sample_supported={McpDevPayloadProjection.text(payload.get('sample_protocol_supported'))} "
                f"included={McpDevPayloadProjection.text(payload.get('sample_included_count'))} "
                f"omitted={McpDevPayloadProjection.text(payload.get('sample_omitted_count'))}"
            ),
        ]
        candidate_routes = payload.get("candidate_image_route_keys")
        if candidate_routes:
            lines.append(
                "Image routes: "
                f"{ViewerValidationRenderer._sequence_text(candidate_routes)}"
            )
        errors = McpDevPayloadProjection.sequence_of_mappings(payload.get("errors"))
        if errors:
            lines.append("Errors:")
            lines.extend(ViewerValidationRenderer._error_lines(errors))
        warnings = McpDevPayloadProjection.sequence_of_mappings(payload.get("warnings"))
        if warnings:
            lines.append("Warnings:")
            lines.extend(ViewerValidationRenderer._error_lines(warnings))
        records = McpDevPayloadProjection.sequence_of_mappings(payload.get("records"))
        if records:
            lines.append("Image records:")
            lines.extend(
                cls._record_lines(
                    records,
                    include_array_values_requested=include_array_values_requested,
                )
            )
        return "\n".join(lines)

    @classmethod
    def _record_lines(
        cls,
        records: tuple[Mapping[str, JsonValue], ...],
        *,
        include_array_values_requested: bool | None,
    ) -> list[str]:
        lines: list[str] = []
        for record in records:
            array_summary = McpDevPayloadProjection.nested_mapping(
                record,
                "array_value_summary",
            )
            summary = McpDevPayloadProjection.nested_mapping(record, "summary")
            array_reason = array_summary.get("omitted_reason")
            if (
                include_array_values_requested is False
                and array_reason == "max_array_elements_exceeded"
                and array_summary.get("max_array_elements") == 0
            ):
                array_reason = "array_values_not_requested"
            reason_text = ""
            if isinstance(array_reason, str) and array_reason:
                reason_text = f" reason={array_reason}"
            max_array_elements = array_summary.get("max_array_elements")
            max_elements_text = ""
            if max_array_elements is not None:
                max_elements_text = (
                    " max_elements="
                    f"{McpDevPayloadProjection.text(max_array_elements)}"
                )
            sample_shape = array_summary.get("shape")
            rerun_hint = ""
            required_elements = cls._shape_element_count(sample_shape)
            if (
                array_reason == "max_array_elements_exceeded"
                and required_elements is not None
            ):
                rerun_hint = f" rerun_max_elements={required_elements}"
            if array_reason == "array_values_not_requested":
                rerun_hint = " rerun_with=--include-array-values"
                if required_elements is not None:
                    rerun_hint += f" --max-array-elements {required_elements}"
            lines.append(
                "- "
                f"{McpDevPayloadProjection.text(record.get('payload_route_key'))}: "
                f"layer={McpDevPayloadProjection.text(record.get('layer_route_key'))} "
                f"axis={ViewerAxisIndexRenderer.render(record.get('axis_indices'))} "
                f"path={ViewerPayloadPathRenderer.render(record.get('path'))} "
                f"shape={cls._json_summary(summary.get('shape'))} "
                f"dtype={McpDevPayloadProjection.text(summary.get('dtype'))} "
                f"min={McpDevPayloadProjection.text(summary.get('min'))} "
                f"max={McpDevPayloadProjection.text(summary.get('max'))} "
                f"nonzero={McpDevPayloadProjection.text(summary.get('nonzero_count'))} "
                f"included={McpDevPayloadProjection.text(array_summary.get('included'))} "
                f"sample_shape={cls._json_summary(sample_shape)}"
                f"{reason_text}"
                f"{max_elements_text}"
                f"{rerun_hint}"
            )
            array_values = record.get("array_values")
            if array_summary.get("included") is True and cls._json_value_count(array_values) <= 64:
                lines.append(f"  sample values: {json.dumps(array_values)}")
        return lines

    @staticmethod
    def _json_summary(value: JsonValue) -> str:
        if value is None:
            return "<none>"
        return json.dumps(value, sort_keys=True)

    @staticmethod
    def _json_value_count(value: JsonValue) -> int:
        if isinstance(value, list | tuple):
            return sum(ViewerImageSampleRenderer._json_value_count(item) for item in value)
        if isinstance(value, Mapping):
            return sum(
                ViewerImageSampleRenderer._json_value_count(item)
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


class ViewerNavigationRenderer(McpDevOutputRenderer):
    """Compact renderer for viewer navigation and layer isolation results."""

    @classmethod
    def render_bindings(cls) -> tuple[McpDevOutputRendererBinding, ...]:
        return (
            McpDevOutputRendererBinding(
                output_contract=ViewerWindowNavigationResult,
                renderer_type=cls,
                render_function=cls.render_navigation,
            ),
            McpDevOutputRendererBinding(
                output_contract=ViewerWindowLayerIsolationResult,
                renderer_type=cls,
                render_function=cls.render_isolation,
            ),
        )

    @classmethod
    def render_navigation(cls, response: JsonObject) -> str:
        payload = McpDevPayloadProjection.first_tool_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)
        lines = [
            (
                "Viewer navigation: "
                f"observed={McpDevPayloadProjection.text(payload.get('observed'))} "
                f"route={McpDevPayloadProjection.text(payload.get('route_key'))} "
                f"visible={McpDevPayloadProjection.text(payload.get('visible'))} "
                f"selected={McpDevPayloadProjection.text(payload.get('selected'))}"
            ),
            (
                "Position: "
                f"axes={ViewerValidationRenderer._sequence_text(payload.get('axis_labels'))} "
                f"current_step={cls._json_summary(payload.get('current_step'))} "
                f"active_route={McpDevPayloadProjection.text(payload.get('active_dimension_label_route'))}"
            ),
        ]
        cls._append_messages(lines, payload)
        return "\n".join(lines)

    @classmethod
    def render_isolation(cls, response: JsonObject) -> str:
        payload = McpDevPayloadProjection.first_tool_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)
        lines = [
            (
                "Viewer isolation: "
                f"observed={McpDevPayloadProjection.text(payload.get('observed'))} "
                f"applied={McpDevPayloadProjection.text(payload.get('applied'))} "
                f"selected={McpDevPayloadProjection.text(payload.get('selected_route_key'))} "
                f"changed={McpDevPayloadProjection.text(payload.get('changed_route_count'))} "
                f"layers={McpDevPayloadProjection.text(payload.get('layer_count'))}"
            ),
            (
                "Position: "
                f"axes={ViewerValidationRenderer._sequence_text(payload.get('axis_labels'))} "
                f"current_step={cls._json_summary(payload.get('current_step'))} "
                f"active_route={McpDevPayloadProjection.text(payload.get('active_dimension_label_route'))}"
            ),
            f"Visible: {ViewerValidationRenderer._sequence_text(payload.get('visible_route_keys'))}",
            f"Hidden: {ViewerValidationRenderer._sequence_text(payload.get('hidden_route_keys'))}",
        ]
        missing_routes = payload.get("missing_route_keys")
        if isinstance(missing_routes, list) and missing_routes:
            lines.append(
                "Missing routes: "
                f"{ViewerValidationRenderer._sequence_text(missing_routes)}"
            )
        available_layers = McpDevPayloadProjection.sequence_of_mappings(
            payload.get("available_layers")
        )
        visible_layers = McpDevPayloadProjection.sequence_of_mappings(
            payload.get("visible_layers")
        )
        if available_layers:
            lines.append("Available layers:")
            lines.extend(cls._layer_lines(available_layers))
        if visible_layers:
            lines.append("Visible layers:")
            lines.extend(cls._layer_lines(visible_layers))
        cls._append_messages(lines, payload)
        return "\n".join(lines)

    @staticmethod
    def _layer_lines(layers: tuple[Mapping[str, JsonValue], ...]) -> list[str]:
        return [
            "- "
            f"{McpDevPayloadProjection.text(layer.get('route_key'))}: "
            f"visible={McpDevPayloadProjection.text(layer.get('visible'))} "
            f"selected={McpDevPayloadProjection.text(layer.get('selected'))} "
            f"title={McpDevPayloadProjection.quoted_text(layer.get('title'))}"
            for layer in layers
        ]

    @staticmethod
    def _append_messages(lines: list[str], payload: Mapping[str, JsonValue]) -> None:
        errors = McpDevPayloadProjection.sequence_of_mappings(payload.get("errors"))
        if errors:
            lines.append("Errors:")
            lines.extend(ViewerValidationRenderer._error_lines(errors))
        warnings = McpDevPayloadProjection.sequence_of_mappings(payload.get("warnings"))
        if warnings:
            lines.append("Warnings:")
            lines.extend(ViewerValidationRenderer._error_lines(warnings))

    @staticmethod
    def _json_summary(value: JsonValue) -> str:
        if value is None:
            return "<none>"
        return json.dumps(value, sort_keys=True)


class RuntimeServerRenderer(McpDevOutputRenderer):
    """Compact renderer for runtime server scan/info/status payloads."""

    @classmethod
    def render_bindings(cls) -> tuple[McpDevOutputRendererBinding, ...]:
        return (
            McpDevOutputRendererBinding(
                output_contract=RuntimeServerScanResult,
                renderer_type=cls,
                render_function=cls.render_scan,
            ),
            McpDevOutputRendererBinding(
                output_contract=RuntimeServerInfo,
                renderer_type=cls,
                render_function=cls.render_info,
            ),
            McpDevOutputRendererBinding(
                output_contract=RuntimeExecutionStatus,
                renderer_type=cls,
                render_function=cls.render_status,
            ),
        )

    @classmethod
    def render_scan(cls, response: JsonObject) -> str:
        payload = McpDevPayloadProjection.first_tool_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)

        servers = McpDevPayloadProjection.sequence_of_mappings(payload.get("servers"))
        lines = [
            (
                "Runtime scan: "
                f"ports={ViewerValidationRenderer._sequence_text(payload.get('ports'))} "
                f"timeout_ms={McpDevPayloadProjection.text(payload.get('timeout_ms'))} "
                f"servers={len(servers)}"
            )
        ]
        cls._append_messages(lines, payload)
        if servers:
            lines.append("Servers:")
            lines.extend(cls._server_lines(servers))
        return "\n".join(lines)

    @classmethod
    def render_info(cls, response: JsonObject) -> str:
        payload = McpDevPayloadProjection.first_tool_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)
        lines = ["Runtime server:"]
        lines.extend(cls._server_lines((payload,)))
        cls._append_messages(lines, payload)
        return "\n".join(lines)

    @classmethod
    def render_status(cls, response: JsonObject) -> str:
        payload = McpDevPayloadProjection.first_tool_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)
        response_payload = McpDevPayloadProjection.nested_mapping(payload, "response")
        connection = McpDevPayloadProjection.nested_mapping(payload, "connection")
        executions = response_payload.get("executions")
        execution_count = len(executions) if isinstance(executions, list) else 0
        running = response_payload.get("running_executions")
        running_count = len(running) if isinstance(running, list) else 0
        queued = response_payload.get("queued_executions")
        queued_count = len(queued) if isinstance(queued, list) else 0
        lines = [
            (
                "Runtime execution status: "
                f"status={McpDevPayloadProjection.text(payload.get('status'))} "
                f"execution_id={McpDevPayloadProjection.text(payload.get('execution_id'))} "
                f"port={McpDevPayloadProjection.text(connection.get('port'))}"
            ),
            (
                "Executions: "
                f"known={execution_count} "
                f"active={McpDevPayloadProjection.text(response_payload.get('active_executions'))} "
                f"running={running_count} "
                f"queued={queued_count} "
                f"uptime={cls._seconds_text(response_payload.get('uptime'))}"
            ),
        ]
        cls._append_messages(lines, payload)
        return "\n".join(lines)

    @classmethod
    def _server_lines(
        cls,
        servers: tuple[Mapping[str, JsonValue], ...],
    ) -> list[str]:
        lines: list[str] = []
        for server in servers:
            connection = McpDevPayloadProjection.nested_mapping(server, "connection")
            lines.append(
                "- "
                f"port={McpDevPayloadProjection.text(connection.get('port'))} "
                f"server={McpDevPayloadProjection.text(server.get('server'))} "
                f"reachable={McpDevPayloadProjection.text(server.get('reachable'))} "
                f"ready={McpDevPayloadProjection.text(server.get('ready'))} "
                f"control={McpDevPayloadProjection.text(server.get('control_port'))} "
                f"active={McpDevPayloadProjection.text(server.get('active_executions'))} "
                f"running={cls._count_text(server.get('running_executions'))} "
                f"queued={cls._count_text(server.get('queued_executions'))} "
                f"workers={cls._count_text(server.get('workers'))} "
                f"uptime={cls._seconds_text(server.get('uptime'))} "
                f"log={McpDevPayloadProjection.text(server.get('log_file_path'))}"
            )
            errors = McpDevPayloadProjection.sequence_of_mappings(server.get("errors"))
            if errors:
                lines.extend(f"  {line}" for line in ViewerValidationRenderer._error_lines(errors))
        return lines

    @staticmethod
    def _count_text(value: JsonValue) -> str:
        if isinstance(value, list):
            return str(len(value))
        return McpDevPayloadProjection.text(value)

    @staticmethod
    def _seconds_text(value: JsonValue) -> str:
        if isinstance(value, int | float):
            return f"{float(value):.1f}s"
        return McpDevPayloadProjection.text(value)

    @staticmethod
    def _append_messages(lines: list[str], payload: Mapping[str, JsonValue]) -> None:
        errors = McpDevPayloadProjection.sequence_of_mappings(payload.get("errors"))
        if errors:
            lines.append("Errors:")
            lines.extend(ViewerValidationRenderer._error_lines(errors))
        warnings = McpDevPayloadProjection.sequence_of_mappings(payload.get("warnings"))
        if warnings:
            lines.append("Warnings:")
            lines.extend(ViewerValidationRenderer._error_lines(warnings))


class RuntimeDebugInspectionRenderer(McpDevOutputRenderer):
    """Compact renderer for one renderer-independent paused-worker view."""

    output_contract = RuntimeDebugInspectionResult
    render_options_type = CatalogRenderOptions
    MAX_CELL_CHARS = 160
    MAX_TEXT_CHARS = 500

    @classmethod
    def render_with_options(
        cls,
        response: JsonObject,
        options: CatalogRenderOptions,
    ) -> str:
        return cls.render(
            response,
            contains=options.contains,
            limit=options.limit,
        )

    @classmethod
    def render(
        cls,
        response: JsonObject,
        *,
        contains: str | None = None,
        limit: int = 20,
    ) -> str:
        error_lines = McpDiagnosticRenderer.response_error_lines(response)
        if error_lines:
            return "\n".join(("Runtime debug: unavailable", *error_lines))

        payload = McpDevPayloadProjection.first_tool_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)
        connection = McpDevPayloadProjection.nested_mapping(payload, "connection")
        view_model = McpDevPayloadProjection.nested_mapping(payload, "view_model")
        sections = McpDevPayloadProjection.sequence_of_mappings(
            view_model.get("sections")
        )
        matching_sections = tuple(
            section
            for section in sections
            if cls._section_matches(section, contains)
        )
        total_items = sum(cls._section_item_count(section) for section in sections)
        matched_items = sum(
            cls._matched_item_count(section, contains)
            for section in matching_sections
        )
        remaining = max(limit, 0)
        shown_items = 0
        shown_sections = 0
        body_lines: list[str] = []
        for section in matching_sections:
            section_lines, section_shown_items = cls._section_lines(
                section,
                contains=contains,
                remaining=remaining,
            )
            if not section_lines:
                continue
            shown_sections += 1
            shown_items += section_shown_items
            remaining -= section_shown_items
            body_lines.extend(section_lines)

        endpoint = (
            f"{McpDevPayloadProjection.text(connection.get('host'))}:"
            f"{McpDevPayloadProjection.text(connection.get('port'))}"
        )
        lines = [
            (
                "Runtime debug: "
                f"session={McpDevPayloadProjection.text(payload.get('debug_session_id'))} "
                f"endpoint={endpoint} "
                f"transport={McpDevPayloadProjection.text(connection.get('transport_mode'))} "
                f"persistent={McpDevPayloadProjection.text(connection.get('persistent'))} "
                f"title={McpDevPayloadProjection.quoted_text(view_model.get('title'))}"
            ),
            (
                "Sections: "
                f"total={len(sections)} matched={len(matching_sections)} "
                f"shown={shown_sections}"
            ),
            (
                "Items: "
                f"total={total_items} matched={matched_items} shown={shown_items} "
                f"truncated={max(matched_items - shown_items, 0)} limit={max(limit, 0)}"
            ),
        ]
        if contains:
            lines.append(f"Filter: contains={contains}")
        lines.extend(body_lines)
        return "\n".join(lines)

    @classmethod
    def _section_lines(
        cls,
        section: Mapping[str, JsonValue],
        *,
        contains: str | None,
        remaining: int,
    ) -> tuple[list[str], int]:
        table = cls._table(section)
        rows = cls._rows(table)
        matching_rows = cls._matching_rows(section, rows, contains)
        visible_rows = matching_rows[:remaining]
        text = section.get("text")
        text_value = text if isinstance(text, str) and text else None
        text_matches = cls._text_matches(section, text_value, contains)
        show_text = text_matches and len(visible_rows) < remaining
        shown_items = len(visible_rows) + int(show_text)
        matching_items = len(matching_rows) + int(text_matches)
        if matching_items > 0 and shown_items == 0:
            return [], 0

        lines = [
            (
                "Section: "
                f"kind={cls._bounded_cell(section.get('kind'))} "
                f"title={McpDevPayloadProjection.quoted_text(section.get('title'))} "
                f"items={cls._section_item_count(section)} "
                f"matched={matching_items} shown={shown_items} "
                f"truncated={max(matching_items - shown_items, 0)}"
            )
        ]
        if table is not None:
            columns = cls._columns(table)
            lines.append(
                f"Columns ({len(columns)}): "
                + (" | ".join(columns) if columns else "<none>")
            )
            lines.append(
                "Rows: "
                f"total={len(rows)} matched={len(matching_rows)} "
                f"shown={len(visible_rows)} "
                f"truncated={max(len(matching_rows) - len(visible_rows), 0)}"
            )
            lines.extend(cls._row_line(row) for row in visible_rows)
            if not rows:
                empty_message = table.get("empty_message")
                if isinstance(empty_message, str) and empty_message:
                    lines.append(
                        f"Empty: {McpDevPayloadProjection.quoted_text(empty_message)}"
                    )
        if text_value is not None and text_matches:
            compact_text = " ".join(text_value.split())
            visible_text = cls._bounded_text(compact_text) if show_text else ""
            shown_chars = len(visible_text)
            lines.append(
                "Text: "
                f"chars={len(compact_text)} shown={shown_chars} "
                f"truncated={max(len(compact_text) - shown_chars, 0)}"
            )
            if visible_text:
                lines.append(f"- {visible_text}")
        return lines, shown_items

    @staticmethod
    def _table(
        section: Mapping[str, JsonValue],
    ) -> Mapping[str, JsonValue] | None:
        table = section.get("table")
        return table if isinstance(table, Mapping) else None

    @classmethod
    def _columns(cls, table: Mapping[str, JsonValue] | None) -> tuple[str, ...]:
        if table is None:
            return ()
        columns = table.get("columns")
        if not isinstance(columns, list):
            return ()
        return tuple(cls._bounded_cell(column) for column in columns)

    @classmethod
    def _rows(
        cls,
        table: Mapping[str, JsonValue] | None,
    ) -> tuple[tuple[str, ...], ...]:
        if table is None:
            return ()
        rows = table.get("rows")
        if not isinstance(rows, list):
            return ()
        return tuple(
            tuple(cls._bounded_cell(cell) for cell in row)
            for row in rows
            if isinstance(row, list)
        )

    @classmethod
    def _matching_rows(
        cls,
        section: Mapping[str, JsonValue],
        rows: tuple[tuple[str, ...], ...],
        contains: str | None,
    ) -> tuple[tuple[str, ...], ...]:
        if not contains or cls._section_metadata_matches(section, contains):
            return rows
        needle = contains.casefold()
        return tuple(row for row in rows if needle in " | ".join(row).casefold())

    @classmethod
    def _text_matches(
        cls,
        section: Mapping[str, JsonValue],
        text: str | None,
        contains: str | None,
    ) -> bool:
        if text is None:
            return False
        if not contains or cls._section_metadata_matches(section, contains):
            return True
        return contains.casefold() in text.casefold()

    @classmethod
    def _section_matches(
        cls,
        section: Mapping[str, JsonValue],
        contains: str | None,
    ) -> bool:
        if not contains or cls._section_metadata_matches(section, contains):
            return True
        table = cls._table(section)
        rows = cls._rows(table)
        text = section.get("text")
        text_value = text if isinstance(text, str) and text else None
        return bool(cls._matching_rows(section, rows, contains)) or cls._text_matches(
            section,
            text_value,
            contains,
        )

    @classmethod
    def _section_metadata_matches(
        cls,
        section: Mapping[str, JsonValue],
        contains: str,
    ) -> bool:
        table = cls._table(section)
        metadata = " ".join(
            (
                McpDevPayloadProjection.text(section.get("kind")),
                McpDevPayloadProjection.text(section.get("title")),
                " ".join(cls._columns(table)),
                McpDevPayloadProjection.text(
                    None if table is None else table.get("empty_message")
                ),
            )
        )
        return contains.casefold() in metadata.casefold()

    @classmethod
    def _section_item_count(cls, section: Mapping[str, JsonValue]) -> int:
        rows = cls._rows(cls._table(section))
        text = section.get("text")
        return len(rows) + int(isinstance(text, str) and bool(text))

    @classmethod
    def _matched_item_count(
        cls,
        section: Mapping[str, JsonValue],
        contains: str | None,
    ) -> int:
        rows = cls._rows(cls._table(section))
        text = section.get("text")
        text_value = text if isinstance(text, str) and text else None
        return len(cls._matching_rows(section, rows, contains)) + int(
            cls._text_matches(section, text_value, contains)
        )

    @classmethod
    def _row_line(cls, row: tuple[str, ...]) -> str:
        return "- " + " | ".join(row)

    @classmethod
    def _bounded_cell(cls, value: JsonValue) -> str:
        text = (
            str.__str__(value)
            if isinstance(value, str)
            else McpDevPayloadProjection.text(value)
        )
        if len(text) <= cls.MAX_CELL_CHARS:
            return text
        return f"{text[: cls.MAX_CELL_CHARS - 3]}..."

    @classmethod
    def _bounded_text(cls, value: str) -> str:
        if len(value) <= cls.MAX_TEXT_CHARS:
            return value
        return f"{value[: cls.MAX_TEXT_CHARS - 3]}..."


class ViewerProbeRenderer(McpDevOutputRenderer):
    """Compact renderer for cheap viewer reachability probes."""

    output_contract = ViewerWindowProbeResult

    @classmethod
    def render(cls, response: JsonObject) -> str:
        payload = McpDevPayloadProjection.first_tool_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)
        viewer = McpDevPayloadProjection.nested_mapping(payload, "viewer")
        connection = McpDevPayloadProjection.nested_mapping(payload, "connection")
        lines = [
            (
                "Viewer probe: "
                f"reachable={McpDevPayloadProjection.text(payload.get('reachable'))} "
                f"observed={McpDevPayloadProjection.text(payload.get('observed'))} "
                f"type={McpDevPayloadProjection.text(viewer.get('viewer_type'))} "
                f"title={McpDevPayloadProjection.quoted_text(viewer.get('title'))}"
            ),
            (
                "Window: "
                f"port={McpDevPayloadProjection.text(connection.get('port'))} "
                f"layers={McpDevPayloadProjection.text(payload.get('layer_count'))} "
                f"component_groups={McpDevPayloadProjection.text(payload.get('component_group_count'))} "
                f"component_items={McpDevPayloadProjection.text(payload.get('component_item_count'))}"
            ),
        ]
        RuntimeServerRenderer._append_messages(lines, payload)
        return "\n".join(lines)


class ViewerSnapshotRenderer(McpDevOutputRenderer):
    """Compact renderer for viewer snapshot resources."""

    output_contract = ViewerWindowSnapshotResult

    @classmethod
    def render(cls, response: JsonObject) -> str:
        payload = McpDevPayloadProjection.first_tool_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)
        viewer = McpDevPayloadProjection.nested_mapping(payload, "viewer")
        resource = McpDevPayloadProjection.nested_mapping(payload, "resource")
        lines = [
            (
                "Viewer snapshot: "
                f"captured={McpDevPayloadProjection.text(payload.get('captured'))} "
                f"type={McpDevPayloadProjection.text(viewer.get('viewer_type'))} "
                f"title={McpDevPayloadProjection.quoted_text(viewer.get('title'))} "
                f"scope={McpDevPayloadProjection.text(payload.get('capture_scope'))}"
            ),
            (
                "Image: "
                f"size={McpDevPayloadProjection.text(payload.get('width'))}x"
                f"{McpDevPayloadProjection.text(payload.get('height'))} "
                f"bytes={McpDevPayloadProjection.text(resource.get('size_bytes'))} "
                f"mime={McpDevPayloadProjection.text(resource.get('mime_type'))}"
            ),
            (
                "Resource: "
                f"path={McpDevPayloadProjection.text(resource.get('path'))} "
                f"uri={McpDevPayloadProjection.text(resource.get('uri'))} "
                f"sha256={McpDevPayloadProjection.text(resource.get('sha256'))}"
            ),
        ]
        RuntimeServerRenderer._append_messages(lines, payload)
        return "\n".join(lines)


class WindowSnapshotRenderer(McpDevOutputRenderer):
    """Compact renderer for UI bridge window snapshot resources."""

    output_contract = UiWindowSnapshotResult

    @classmethod
    def render(cls, response: JsonObject) -> str:
        payload = McpDevPayloadProjection.first_tool_payload(response)
        if payload is None:
            return json.dumps(response, indent=2, sort_keys=True)

        summary = McpDevPayloadProjection.nested_mapping(payload, "summary")
        resource = McpDevPayloadProjection.nested_mapping(payload, "resource")
        lines = [
            (
                "Window snapshot: "
                f"captured={McpDevPayloadProjection.text(payload.get('captured'))} "
                f"window={McpDevPayloadProjection.text(payload.get('window_id'))} "
                f"title={McpDevPayloadProjection.quoted_text(summary.get('title'))} "
                f"kind={McpDevPayloadProjection.text(summary.get('window_kind'))} "
                f"scope={McpDevPayloadProjection.text(payload.get('capture_scope'))}"
            ),
            (
                "Status: "
                f"visible={McpDevPayloadProjection.text(summary.get('visible'))} "
                f"dirty={McpDevPayloadProjection.text(summary.get('dirty'))} "
                "dirty_fields="
                f"{McpDevPayloadProjection.text(summary.get('dirty_field_count'))} "
                "default_diff="
                f"{McpDevPayloadProjection.text(summary.get('signature_diff'))} "
                "default_diff_fields="
                f"{McpDevPayloadProjection.text(summary.get('signature_diff_field_count'))} "
                f"markers={cls._marker_text(summary)}"
            ),
            (
                "Image: "
                f"size={McpDevPayloadProjection.text(payload.get('width'))}x"
                f"{McpDevPayloadProjection.text(payload.get('height'))} "
                f"bytes={McpDevPayloadProjection.text(resource.get('size_bytes'))} "
                f"mime={McpDevPayloadProjection.text(resource.get('mime_type'))}"
            ),
            (
                "Resource: "
                f"path={McpDevPayloadProjection.text(resource.get('path'))} "
                f"uri={McpDevPayloadProjection.text(resource.get('uri'))} "
                f"sha256={McpDevPayloadProjection.text(resource.get('sha256'))}"
            ),
        ]
        object_state_scope_id = summary.get("object_state_scope_id")
        if isinstance(object_state_scope_id, str) and object_state_scope_id:
            lines.append(f"ObjectState: scope={object_state_scope_id}")
        managed_action_ids = summary.get("managed_action_ids")
        if isinstance(managed_action_ids, list) and managed_action_ids:
            lines.append(
                "Actions: "
                + ",".join(
                    McpDevPayloadProjection.text(action_id)
                    for action_id in managed_action_ids
                )
            )
        RuntimeServerRenderer._append_messages(lines, payload)
        return "\n".join(lines)

    @staticmethod
    def _marker_text(summary: Mapping[str, JsonValue]) -> str:
        markers = summary.get("semantic_markers")
        if not isinstance(markers, list) or not markers:
            return "-"
        marker_text = "".join(
            McpDevPayloadProjection.text(marker)
            for marker in markers
            if McpDevPayloadProjection.text(marker)
        )
        return marker_text or "-"
