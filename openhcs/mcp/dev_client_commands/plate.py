"""Plate and selected-plate command declarations."""

from __future__ import annotations

import argparse

from openhcs.agent.capabilities import agent_capabilities
from openhcs.agent.dto.common import JsonValue
from openhcs.agent.dto.plate import (
    PlateFileQueryRequest,
    PlateFileStreamRequest,
    PlateImageSampleRequest,
    PlatePathInspectionRequest,
    SelectedPlateFileQueryRequest,
    SelectedPlateFileQueryTarget,
    SelectedPlateFileStreamRequest,
    SelectedPlateImageInspectionRequest,
    SelectedPlateImageSampleRequest,
    SyntheticPlateGenerationRequest,
)
from openhcs.core.plate_file_inventory import PlateFileInventoryQuery
from openhcs.core.synthetic_plate_generation import SYNTHETIC_PLATE_GENERATION_PROFILE
from openhcs.mcp.dev_client_commanding import SingleToolCommandSpec
from openhcs.mcp.dev_client_core import (
    McpToolArgumentAuthority,
    add_request_field_option,
    add_ui_connection_options,
    plate_file_stream_kind_argument,
    request_field_bool_default,
    ui_connection_arguments,
)

class GenerateSyntheticPlateCommandSpec(SingleToolCommandSpec):
    capability = agent_capabilities.generate_synthetic_plate
    default_timeout_seconds = 30.0

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "output_dir",
            help="Synthetic plate output directory under OPENHCS_AGENT_WRITE_ROOTS.",
        )
        add_request_field_option(
            parser,
            SyntheticPlateGenerationRequest,
            "grid_rows",
            "--grid-rows",
        )
        add_request_field_option(
            parser,
            SyntheticPlateGenerationRequest,
            "grid_cols",
            "--grid-cols",
        )
        add_request_field_option(
            parser,
            SyntheticPlateGenerationRequest,
            "tile_width",
            "--tile-width",
        )
        add_request_field_option(
            parser,
            SyntheticPlateGenerationRequest,
            "tile_height",
            "--tile-height",
        )
        add_request_field_option(
            parser,
            SyntheticPlateGenerationRequest,
            "overlap_percent",
            "--overlap-percent",
        )
        add_request_field_option(
            parser,
            SyntheticPlateGenerationRequest,
            "stage_error_px",
            "--stage-error-px",
        )
        add_request_field_option(
            parser,
            SyntheticPlateGenerationRequest,
            "wavelengths",
            "--wavelengths",
        )
        add_request_field_option(
            parser,
            SyntheticPlateGenerationRequest,
            "z_stack_levels",
            "--z-stack-levels",
        )
        add_request_field_option(
            parser,
            SyntheticPlateGenerationRequest,
            "num_cells",
            "--num-cells",
        )
        add_request_field_option(
            parser,
            SyntheticPlateGenerationRequest,
            "shared_cell_fraction",
            "--shared-cell-fraction",
        )
        parser.add_argument(
            "--well",
            dest="wells",
            action="append",
            help="Well ID to generate. Repeat for multiple wells.",
        )
        add_request_field_option(
            parser,
            SyntheticPlateGenerationRequest,
            "format",
            "--format",
            choices=tuple(
                item.value
                for item in SYNTHETIC_PLATE_GENERATION_PROFILE.supported_formats
            ),
        )
        add_request_field_option(
            parser,
            SyntheticPlateGenerationRequest,
            "openhcs_format",
            "--openhcs-format",
            action="store_true",
            help="Also write openhcs_metadata.json for the generated plate.",
        )
        add_request_field_option(
            parser,
            SyntheticPlateGenerationRequest,
            "include_all_components",
            "--omit-singleton-components",
            dest="include_all_components",
            action="store_false",
            help="Let the generator omit singleton filename components.",
        )
        add_request_field_option(
            parser,
            SyntheticPlateGenerationRequest,
            "random_seed",
            "--random-seed",
        )
        add_request_field_option(
            parser,
            SyntheticPlateGenerationRequest,
            "sample_file_limit",
            "--sample-file-limit",
        )
        parser.add_argument(
            "--json",
            action="store_true",
            help="Render the complete MCP JSON response instead of a compact summary.",
        )

    def tool_arguments(
        self,
        args: argparse.Namespace,
    ) -> dict[str, JsonValue]:
        request = SyntheticPlateGenerationRequest.from_fields(
            output_dir=args.output_dir,
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
            tile_width=args.tile_width,
            tile_height=args.tile_height,
            overlap_percent=args.overlap_percent,
            stage_error_px=args.stage_error_px,
            wavelengths=args.wavelengths,
            z_stack_levels=args.z_stack_levels,
            num_cells=args.num_cells,
            shared_cell_fraction=args.shared_cell_fraction,
            wells=args.wells,
            format=args.format,
            openhcs_format=args.openhcs_format,
            include_all_components=args.include_all_components,
            random_seed=args.random_seed,
            sample_file_limit=args.sample_file_limit,
        )
        return McpToolArgumentAuthority.from_payload(request.as_tool_arguments())

class InspectPlateCommandSpec(SingleToolCommandSpec):
    capability = agent_capabilities.inspect_plate_path

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "plate_path",
            help="Local plate folder path under OPENHCS_AGENT_READ_ROOTS.",
        )
        add_request_field_option(
            parser,
            PlatePathInspectionRequest,
            "microscope_type",
            "--microscope-type",
            help="Microscope type to use, or auto for handler detection.",
        )
        add_request_field_option(
            parser,
            PlatePathInspectionRequest,
            "pattern_format",
            "--pattern-format",
            help="Optional filename pattern format forwarded to the handler parser.",
        )
        add_request_field_option(
            parser,
            PlatePathInspectionRequest,
            "max_sample_files",
            "--max-sample-files",
        )
        add_request_field_option(
            parser,
            PlatePathInspectionRequest,
            "max_component_values",
            "--max-component-values",
        )
        add_request_field_option(
            parser,
            PlatePathInspectionRequest,
            "max_parse_failure_samples",
            "--max-parse-failure-samples",
        )
        add_request_field_option(
            parser,
            PlatePathInspectionRequest,
            "max_files_to_parse",
            "--max-files-to-parse",
        )
        parser.add_argument(
            "--json",
            action="store_true",
            help="Render the complete MCP JSON response instead of a compact summary.",
        )

    def tool_arguments(
        self,
        args: argparse.Namespace,
    ) -> dict[str, JsonValue]:
        request = PlatePathInspectionRequest.from_fields(
            plate_path=args.plate_path,
            microscope_type=args.microscope_type,
            pattern_format=args.pattern_format,
            max_sample_files=args.max_sample_files,
            max_component_values=args.max_component_values,
            max_parse_failure_samples=args.max_parse_failure_samples,
            max_files_to_parse=args.max_files_to_parse,
        )
        return McpToolArgumentAuthority.from_payload(request.as_tool_arguments())

class QueryPlateFilesCommandSpec(SingleToolCommandSpec):
    capability = agent_capabilities.query_plate_files

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "plate_path",
            help="Local plate folder path under OPENHCS_AGENT_READ_ROOTS.",
        )
        add_request_field_option(
            parser,
            PlateFileQueryRequest,
            "microscope_type",
            "--microscope-type",
            help="Microscope type to use, or auto for handler detection.",
        )
        add_request_field_option(
            parser,
            PlateFileQueryRequest,
            "pattern_format",
            "--pattern-format",
            help="Optional filename pattern format forwarded to the handler parser.",
        )
        add_request_field_option(
            parser,
            PlateFileQueryRequest,
            "kind",
            "--kind",
            choices=PlateFileInventoryQuery.kind_choices(),
            help="File kind to return.",
        )
        add_request_field_option(
            parser,
            PlateFileQueryRequest,
            "path_contains",
            "--path-contains",
        )
        add_request_field_option(parser, PlateFileQueryRequest, "well", "--well")
        add_request_field_option(parser, PlateFileQueryRequest, "offset", "--offset")
        add_request_field_option(parser, PlateFileQueryRequest, "limit", "--limit")
        add_request_field_option(
            parser,
            PlateFileQueryRequest,
            "include_previews",
            "--include-previews",
            dest="include_previews",
            action="store_true",
            help="Include bounded previews for text-like result artifacts.",
        )
        parser.add_argument(
            "--no-previews",
            dest="include_previews",
            action="store_false",
            help="Do not include bounded previews for text-like result artifacts.",
        )
        add_request_field_option(
            parser,
            PlateFileQueryRequest,
            "max_preview_lines",
            "--max-preview-lines",
        )
        add_request_field_option(
            parser,
            PlateFileQueryRequest,
            "max_preview_bytes",
            "--max-preview-bytes",
        )
        parser.add_argument(
            "--json",
            action="store_true",
            help="Render the complete MCP JSON response instead of a compact summary.",
        )

    def tool_arguments(
        self,
        args: argparse.Namespace,
    ) -> dict[str, JsonValue]:
        request = PlateFileQueryRequest.from_fields(
            plate_path=args.plate_path,
            microscope_type=args.microscope_type,
            pattern_format=args.pattern_format,
            kind=args.kind,
            path_contains=args.path_contains,
            well=args.well,
            offset=args.offset,
            limit=args.limit,
            include_previews=args.include_previews,
            max_preview_lines=args.max_preview_lines,
            max_preview_bytes=args.max_preview_bytes,
        )
        return McpToolArgumentAuthority.from_payload(request.as_tool_arguments())

class SamplePlateImageCommandSpec(SingleToolCommandSpec):
    capability = agent_capabilities.sample_plate_image

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "plate_path",
            help="Local plate folder path under OPENHCS_AGENT_READ_ROOTS.",
        )
        parser.add_argument(
            "image_path",
            help="Virtual path, full virtual path, source path, or unique basename.",
        )
        add_request_field_option(
            parser,
            PlateImageSampleRequest,
            "microscope_type",
            "--microscope-type",
            help="Microscope type to use, or auto for handler detection.",
        )
        add_request_field_option(
            parser,
            PlateImageSampleRequest,
            "pattern_format",
            "--pattern-format",
            help="Optional filename pattern format forwarded to the handler parser.",
        )
        add_request_field_option(parser, PlateImageSampleRequest, "y", "--y")
        add_request_field_option(parser, PlateImageSampleRequest, "x", "--x")
        add_request_field_option(
            parser,
            PlateImageSampleRequest,
            "height",
            "--height",
        )
        add_request_field_option(parser, PlateImageSampleRequest, "width", "--width")
        add_request_field_option(
            parser,
            PlateImageSampleRequest,
            "resolution_index",
            "--resolution-index",
            help=(
                "Exact native resolution index (0 is full resolution). Omit for "
                "bounded automatic native-level selection."
            ),
        )
        add_request_field_option(
            parser,
            PlateImageSampleRequest,
            "max_auto_resolution_size",
            "--max-auto-resolution-size",
            help="Largest spatial edge preferred during automatic level selection.",
        )
        add_request_field_option(
            parser,
            PlateImageSampleRequest,
            "max_array_elements",
            "--max-array-elements",
            help="Largest sampled element count returned with pixel values.",
        )
        parser.add_argument(
            "--no-array-values",
            dest="no_array_values",
            action="store_true",
            default=not request_field_bool_default(
                PlateImageSampleRequest,
                "include_array_values",
            ),
            help="Return image stats and sample bounds without sample pixel values.",
        )
        parser.add_argument(
            "--include-array-values",
            dest="no_array_values",
            action="store_false",
            default=not request_field_bool_default(
                PlateImageSampleRequest,
                "include_array_values",
            ),
            help="Include sampled pixel values in the response.",
        )
        parser.add_argument(
            "--json",
            action="store_true",
            help="Render the complete MCP JSON response instead of a compact summary.",
        )

    def tool_arguments(
        self,
        args: argparse.Namespace,
    ) -> dict[str, JsonValue]:
        request = PlateImageSampleRequest.from_fields(
            plate_path=args.plate_path,
            image_path=args.image_path,
            microscope_type=args.microscope_type,
            pattern_format=args.pattern_format,
            y=args.y,
            x=args.x,
            height=args.height,
            width=args.width,
            resolution_index=args.resolution_index,
            max_auto_resolution_size=args.max_auto_resolution_size,
            include_array_values=not args.no_array_values,
            max_array_elements=args.max_array_elements,
        )
        return McpToolArgumentAuthority.from_payload(request.as_tool_arguments())

class PlateFileStreamCommandOptions:
    """Shared CLI options for plate file streaming commands."""

    @staticmethod
    def configure_parser(
        parser: argparse.ArgumentParser,
        *,
        include_plate_path: bool,
        include_selected_target: bool,
        include_viewer_connection_aliases: bool = False,
        request_type: type = PlateFileStreamRequest,
    ) -> None:
        if include_plate_path:
            parser.add_argument(
                "plate_path",
                help="Local plate folder path under OPENHCS_AGENT_READ_ROOTS.",
            )
        parser.add_argument(
            "file_paths",
            nargs="*",
            help=(
                "Virtual image paths, source paths, result paths, or unique basenames. "
                "If omitted, a bounded query selects streamable records."
            ),
        )
        add_request_field_option(
            parser,
            request_type,
            "microscope_type",
            "--microscope-type",
            help="Microscope type to use, or auto for handler detection.",
        )
        add_request_field_option(
            parser,
            request_type,
            "pattern_format",
            "--pattern-format",
            help="Optional filename pattern format forwarded to the handler parser.",
        )
        parser.add_argument(
            "--kind",
            choices=PlateFileInventoryQuery.kind_choices(),
            default=None,
            help=(
                "File kind to resolve. Defaults to all for explicit paths and image "
                "for query-based streaming."
            ),
        )
        if include_selected_target:
            add_request_field_option(
                parser,
                request_type,
                "target",
                "--target",
                choices=tuple(target.value for target in SelectedPlateFileQueryTarget),
                help="Plate root to stream from the selected PlateManager row.",
            )
        parser.add_argument("--path-contains")
        parser.add_argument("--well")
        add_request_field_option(
            parser,
            request_type,
            "limit",
            "--limit",
        )
        add_request_field_option(
            parser,
            request_type,
            "viewer_config_key",
            "--viewer-config-key",
            help="Streaming config key such as napari_streaming_config.",
        )
        viewer_host_options = ["--viewer-host"]
        viewer_port_options = ["--viewer-port"]
        viewer_transport_options = ["--viewer-transport-mode"]
        if include_viewer_connection_aliases:
            viewer_host_options.append("--host")
            viewer_port_options.append("--port")
            viewer_transport_options.append("--transport-mode")
        add_request_field_option(
            parser,
            request_type,
            "host",
            *viewer_host_options,
            dest="viewer_host",
        )
        add_request_field_option(
            parser,
            request_type,
            "port",
            *viewer_port_options,
            dest="viewer_port",
        )
        add_request_field_option(
            parser,
            request_type,
            "transport_mode",
            *viewer_transport_options,
            dest="viewer_transport_mode",
        )
        persistence = parser.add_mutually_exclusive_group()
        add_request_field_option(
            persistence,
            request_type,
            "persistent",
            "--persistent",
            dest="persistent",
            action="store_true",
        )
        persistence.add_argument(
            "--non-persistent",
            dest="persistent",
            action="store_false",
        )
        parser.add_argument("--fresh-viewer", action="store_true")
        parser.add_argument(
            "--json",
            action="store_true",
            help="Render the complete MCP JSON response instead of a compact summary.",
        )

class StreamPlateFilesCommandSpec(SingleToolCommandSpec):
    capability = agent_capabilities.stream_plate_files_to_viewer
    default_timeout_seconds = 60.0

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        PlateFileStreamCommandOptions.configure_parser(
            parser,
            include_plate_path=True,
            include_selected_target=False,
            include_viewer_connection_aliases=True,
        )

    def tool_arguments(
        self,
        args: argparse.Namespace,
    ) -> dict[str, JsonValue]:
        request = PlateFileStreamRequest.from_fields(
            plate_path=args.plate_path,
            file_paths=list(args.file_paths),
            microscope_type=args.microscope_type,
            pattern_format=args.pattern_format,
            kind=plate_file_stream_kind_argument(
                PlateFileStreamRequest,
                args.kind,
                args.file_paths,
            ),
            path_contains=args.path_contains,
            well=args.well,
            limit=args.limit,
            viewer_config_key=args.viewer_config_key,
            host=args.viewer_host,
            port=args.viewer_port,
            transport_mode=args.viewer_transport_mode,
            persistent=args.persistent,
            fresh_viewer=args.fresh_viewer,
        )
        return McpToolArgumentAuthority.from_payload(request.as_tool_arguments())

class SelectedPlateImagesCommandSpec(SingleToolCommandSpec):
    capability = agent_capabilities.ui_inspect_selected_plate_images

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        add_request_field_option(
            parser,
            SelectedPlateImageInspectionRequest,
            "microscope_type",
            "--microscope-type",
            help="Microscope type to use, or auto for handler detection.",
        )
        add_request_field_option(
            parser,
            SelectedPlateImageInspectionRequest,
            "pattern_format",
            "--pattern-format",
            help="Optional filename pattern format forwarded to the handler parser.",
        )
        add_request_field_option(
            parser,
            SelectedPlateImageInspectionRequest,
            "target",
            "--target",
            choices=tuple(target.value for target in SelectedPlateFileQueryTarget),
            help="Plate root to inspect from the selected PlateManager row.",
        )
        add_request_field_option(
            parser,
            SelectedPlateImageInspectionRequest,
            "max_sample_files",
            "--max-sample-files",
        )
        add_request_field_option(
            parser,
            SelectedPlateImageInspectionRequest,
            "max_component_values",
            "--max-component-values",
        )
        add_request_field_option(
            parser,
            SelectedPlateImageInspectionRequest,
            "max_parse_failure_samples",
            "--max-parse-failure-samples",
        )
        add_request_field_option(
            parser,
            SelectedPlateImageInspectionRequest,
            "max_files_to_parse",
            "--max-files-to-parse",
        )
        parser.add_argument(
            "--json",
            action="store_true",
            help="Render the complete MCP JSON response instead of a compact summary.",
        )
        add_ui_connection_options(parser)

    def tool_arguments(
        self,
        args: argparse.Namespace,
    ) -> dict[str, JsonValue]:
        request = SelectedPlateImageInspectionRequest.from_fields(
            microscope_type=args.microscope_type,
            pattern_format=args.pattern_format,
            target=args.target,
            max_sample_files=args.max_sample_files,
            max_component_values=args.max_component_values,
            max_parse_failure_samples=args.max_parse_failure_samples,
            max_files_to_parse=args.max_files_to_parse,
        )
        payload = request.as_tool_arguments()
        payload["connection"] = ui_connection_arguments(
            args,
            timeout_ms=args.timeout_ms,
        )
        return McpToolArgumentAuthority.from_payload(payload)

class SelectedPlateFilesCommandSpec(SingleToolCommandSpec):
    capability = agent_capabilities.ui_query_selected_plate_files

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        add_request_field_option(
            parser,
            SelectedPlateFileQueryRequest,
            "microscope_type",
            "--microscope-type",
            help="Microscope type to use, or auto for handler detection.",
        )
        add_request_field_option(
            parser,
            SelectedPlateFileQueryRequest,
            "pattern_format",
            "--pattern-format",
            help="Optional filename pattern format forwarded to the handler parser.",
        )
        add_request_field_option(
            parser,
            SelectedPlateFileQueryRequest,
            "kind",
            "--kind",
            choices=PlateFileInventoryQuery.kind_choices(),
            help="File kind to return.",
        )
        add_request_field_option(
            parser,
            SelectedPlateFileQueryRequest,
            "target",
            "--target",
            choices=tuple(target.value for target in SelectedPlateFileQueryTarget),
            help="Plate root to query from the selected PlateManager row.",
        )
        add_request_field_option(
            parser,
            SelectedPlateFileQueryRequest,
            "path_contains",
            "--path-contains",
        )
        add_request_field_option(
            parser,
            SelectedPlateFileQueryRequest,
            "well",
            "--well",
        )
        add_request_field_option(
            parser,
            SelectedPlateFileQueryRequest,
            "offset",
            "--offset",
        )
        add_request_field_option(
            parser,
            SelectedPlateFileQueryRequest,
            "limit",
            "--limit",
        )
        add_request_field_option(
            parser,
            SelectedPlateFileQueryRequest,
            "include_previews",
            "--include-previews",
            dest="include_previews",
            action="store_true",
            help="Include bounded previews for text-like result artifacts.",
        )
        parser.add_argument(
            "--no-previews",
            dest="include_previews",
            action="store_false",
            help="Do not include bounded previews for text-like result artifacts.",
        )
        add_request_field_option(
            parser,
            SelectedPlateFileQueryRequest,
            "max_preview_lines",
            "--max-preview-lines",
        )
        add_request_field_option(
            parser,
            SelectedPlateFileQueryRequest,
            "max_preview_bytes",
            "--max-preview-bytes",
        )
        parser.add_argument(
            "--json",
            action="store_true",
            help="Render the complete MCP JSON response instead of a compact summary.",
        )
        add_ui_connection_options(parser)

    def tool_arguments(
        self,
        args: argparse.Namespace,
    ) -> dict[str, JsonValue]:
        request = SelectedPlateFileQueryRequest.from_fields(
            microscope_type=args.microscope_type,
            pattern_format=args.pattern_format,
            kind=args.kind,
            target=args.target,
            path_contains=args.path_contains,
            well=args.well,
            offset=args.offset,
            limit=args.limit,
            include_previews=args.include_previews,
            max_preview_lines=args.max_preview_lines,
            max_preview_bytes=args.max_preview_bytes,
        )
        payload = request.as_tool_arguments()
        payload["connection"] = ui_connection_arguments(
            args,
            timeout_ms=args.timeout_ms,
        )
        return McpToolArgumentAuthority.from_payload(payload)

class SelectedPlateSampleCommandSpec(SingleToolCommandSpec):
    capability = agent_capabilities.ui_sample_selected_plate_image

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "image_path",
            nargs="?",
            help=(
                "Virtual path, full virtual path, source path, or unique basename. "
                "If omitted, the first listed virtual image is sampled."
            ),
        )
        add_request_field_option(
            parser,
            SelectedPlateImageSampleRequest,
            "microscope_type",
            "--microscope-type",
            help="Microscope type to use, or auto for handler detection.",
        )
        add_request_field_option(
            parser,
            SelectedPlateImageSampleRequest,
            "pattern_format",
            "--pattern-format",
            help="Optional filename pattern format forwarded to the handler parser.",
        )
        add_request_field_option(
            parser,
            SelectedPlateImageSampleRequest,
            "target",
            "--target",
            choices=tuple(target.value for target in SelectedPlateFileQueryTarget),
            help="Plate root to sample from the selected PlateManager row.",
        )
        add_request_field_option(parser, SelectedPlateImageSampleRequest, "y", "--y")
        add_request_field_option(parser, SelectedPlateImageSampleRequest, "x", "--x")
        add_request_field_option(
            parser,
            SelectedPlateImageSampleRequest,
            "height",
            "--height",
        )
        add_request_field_option(
            parser,
            SelectedPlateImageSampleRequest,
            "width",
            "--width",
        )
        add_request_field_option(
            parser,
            SelectedPlateImageSampleRequest,
            "resolution_index",
            "--resolution-index",
            help=(
                "Exact native resolution index (0 is full resolution). Omit for "
                "bounded automatic native-level selection."
            ),
        )
        add_request_field_option(
            parser,
            SelectedPlateImageSampleRequest,
            "max_auto_resolution_size",
            "--max-auto-resolution-size",
            help="Largest spatial edge preferred during automatic level selection.",
        )
        add_request_field_option(
            parser,
            SelectedPlateImageSampleRequest,
            "max_array_elements",
            "--max-array-elements",
            help="Largest sampled element count returned with pixel values.",
        )
        parser.add_argument(
            "--no-array-values",
            dest="no_array_values",
            action="store_true",
            default=not request_field_bool_default(
                SelectedPlateImageSampleRequest,
                "include_array_values",
            ),
            help="Return image stats and sample bounds without sample pixel values.",
        )
        parser.add_argument(
            "--include-array-values",
            dest="no_array_values",
            action="store_false",
            default=not request_field_bool_default(
                SelectedPlateImageSampleRequest,
                "include_array_values",
            ),
            help="Include sampled pixel values in the response.",
        )
        parser.add_argument(
            "--json",
            action="store_true",
            help="Render the complete MCP JSON response instead of a compact summary.",
        )
        add_ui_connection_options(parser)

    def tool_arguments(
        self,
        args: argparse.Namespace,
    ) -> dict[str, JsonValue]:
        request = SelectedPlateImageSampleRequest.from_fields(
            image_path=args.image_path,
            microscope_type=args.microscope_type,
            pattern_format=args.pattern_format,
            target=args.target,
            y=args.y,
            x=args.x,
            height=args.height,
            width=args.width,
            resolution_index=args.resolution_index,
            max_auto_resolution_size=args.max_auto_resolution_size,
            include_array_values=not args.no_array_values,
            max_array_elements=args.max_array_elements,
        )
        payload = request.as_tool_arguments()
        payload["connection"] = ui_connection_arguments(
            args,
            timeout_ms=args.timeout_ms,
        )
        return McpToolArgumentAuthority.from_payload(payload)

class SelectedPlateStreamCommandSpec(SingleToolCommandSpec):
    capability = agent_capabilities.ui_stream_selected_plate_files_to_viewer
    default_timeout_seconds = 60.0

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        PlateFileStreamCommandOptions.configure_parser(
            parser,
            include_plate_path=False,
            include_selected_target=True,
            request_type=SelectedPlateFileStreamRequest,
        )
        add_ui_connection_options(parser)

    def tool_arguments(
        self,
        args: argparse.Namespace,
    ) -> dict[str, JsonValue]:
        request = SelectedPlateFileStreamRequest.from_fields(
            file_paths=list(args.file_paths),
            microscope_type=args.microscope_type,
            pattern_format=args.pattern_format,
            kind=plate_file_stream_kind_argument(
                SelectedPlateFileStreamRequest,
                args.kind,
                args.file_paths,
            ),
            target=args.target,
            path_contains=args.path_contains,
            well=args.well,
            limit=args.limit,
            viewer_config_key=args.viewer_config_key,
            host=args.viewer_host,
            port=args.viewer_port,
            transport_mode=args.viewer_transport_mode,
            persistent=args.persistent,
            fresh_viewer=args.fresh_viewer,
        )
        payload = request.as_tool_arguments()
        payload["connection"] = ui_connection_arguments(
            args,
            timeout_ms=args.timeout_ms,
        )
        return McpToolArgumentAuthority.from_payload(payload)
