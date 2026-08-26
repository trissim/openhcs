from dataclasses import dataclass
from enum import Enum

from openhcs.agent.dto.common import SCHEMA_VERSION
from openhcs.agent.dto.functions import (
    FunctionArtifactSpec,
    FunctionCatalogEntry,
    FunctionDetail,
    FunctionParameterSource,
    FunctionParameterSpec,
    FunctionRuntimeContractSummary,
)
from openhcs.agent.dto.ui_bridge import (
    UiObjectStateFieldHelpRequest,
    UiObjectStateFieldSummary,
    UiObjectStateScopeCatalog,
    UiObjectStateScopeIdentity,
    UiObjectStateScopeSummary,
    UiSemanticAddress,
)
from openhcs.agent.services.object_state_field_help_service import (
    ObjectStateFieldHelpService,
)


class EnumHelpMode(Enum):
    ENABLED = "enabled"
    INHERIT = None


@dataclass
class EnumHelpConfig:
    mode: EnumHelpMode | None = EnumHelpMode.ENABLED


def runtime_artifact_parameter_example(
    image_stack,
    grid_dimensions: tuple[int, int],
    overlap_ratio: float = 0.1,
):
    """Compute tile positions.

    Args:
        image_stack: Runtime image stack.
        grid_dimensions: Logical tile grid.
        overlap_ratio: Expected overlap fraction.
    """


def test_object_state_field_help_projects_declaration_owned_enum_inputs() -> None:
    target_name = f"{EnumHelpConfig.__module__}.{EnumHelpConfig.__qualname__}"

    class _UiBridgeService:
        def list_object_state_scopes(self, request, connection):
            assert connection == "ui-connection"
            assert request.field_paths == ("mode",)
            return UiObjectStateScopeCatalog(
                schema_version=SCHEMA_VERSION,
                object_state_token=1,
                current_branch="main",
                current_snapshot_index=0,
                scopes=(
                    UiObjectStateScopeSummary(
                        schema_version=SCHEMA_VERSION,
                        identity=UiObjectStateScopeIdentity(
                            object_state_scope_id="enum-config",
                        ),
                        object_type=target_name,
                        parameter_count=1,
                        dirty_field_count=0,
                        signature_diff_field_count=0,
                        fields=(
                            UiObjectStateFieldSummary(
                                schema_version=SCHEMA_VERSION,
                                address=UiSemanticAddress(
                                    object_state_scope_id="enum-config",
                                    field_path="mode",
                                ),
                                field_name="mode",
                                container_path="",
                                object_state_path_type=target_name,
                                raw_value_type="EnumHelpMode",
                                resolved_value_type="EnumHelpMode",
                                dirty=False,
                                signature_diff=False,
                                last_changed=False,
                            ),
                        ),
                    ),
                ),
            )

    result = ObjectStateFieldHelpService(_UiBridgeService()).describe(
        UiObjectStateFieldHelpRequest(
            object_state_scope_id="enum-config",
            field_path="mode",
        ),
        "ui-connection",
    )

    assert result.errors == ()
    assert result.enum_values == ("enabled", "INHERIT")


def test_object_state_field_help_describes_structured_callable_values():
    class _FunctionCatalogService:
        def __init__(self):
            self.import_paths = []

        def get_by_import_path(
            self,
            import_path,
            *,
            max_doc_chars,
            compact_signature,
        ):
            self.import_paths.append(import_path)
            return FunctionDetail(
                schema_version=SCHEMA_VERSION,
                entry=FunctionCatalogEntry(
                    function_id=(
                        "openhcs:analysis_cell_counting_cpu_count_cells_single_channel"
                    ),
                    import_path=import_path,
                    name="count_cells_single_channel",
                    module=("openhcs.processing.backends.analysis.cell_counting_cpu"),
                    library="openhcs",
                    signature=(
                        "count_cells_single_channel("
                        "detection_method, min_sigma, max_sigma, ...)"
                    ),
                    summary="Count cells in single-channel image stack.",
                ),
                parameters=(
                    FunctionParameterSpec(
                        name="image_stack",
                        annotation="ndarray",
                        default_repr=None,
                        required=False,
                        supplied_by=FunctionParameterSource.PRIMARY_INPUT,
                    ),
                    FunctionParameterSpec(
                        name="min_cell_area",
                        annotation="int",
                        default_repr="10",
                        required=False,
                    ),
                ),
                runtime_contract=FunctionRuntimeContractSummary(
                    callable_kind="openhcs_function",
                    artifact_outputs=(
                        FunctionArtifactSpec(
                            name="cell_counts",
                            kind="special",
                        ),
                        FunctionArtifactSpec(
                            name="segmentation_masks",
                            kind="special",
                        ),
                    ),
                ),
                doc=(
                    "Count cells in single-channel image stack using watershed.\n"
                    "\n"
                    "Args:\n"
                    "    min_cell_area: Minimum area for valid cells."
                ),
            )

    class _UiBridgeService:
        def list_object_state_scopes(self, request, connection):
            assert connection == "ui-connection"
            assert request.include_field_values is True
            assert request.include_field_descriptions is True
            return UiObjectStateScopeCatalog(
                schema_version=SCHEMA_VERSION,
                object_state_token=1,
                current_branch="main",
                current_snapshot_index=0,
                scopes=(
                    UiObjectStateScopeSummary(
                        schema_version=SCHEMA_VERSION,
                        identity=UiObjectStateScopeIdentity(
                            object_state_scope_id="plate::step_0",
                        ),
                        object_type="FunctionStep",
                        parameter_count=1,
                        dirty_field_count=0,
                        signature_diff_field_count=0,
                        fields=(
                            UiObjectStateFieldSummary(
                                schema_version=SCHEMA_VERSION,
                                address=UiSemanticAddress(
                                    object_state_scope_id="plate::step_0",
                                    field_path="func",
                                ),
                                field_name="func",
                                container_path="",
                                object_state_path_type=(
                                    "openhcs.core.steps.function_step.FunctionStep"
                                ),
                                raw_value_type="tuple",
                                resolved_value_type="tuple",
                                dirty=False,
                                signature_diff=True,
                                last_changed=False,
                                raw_value=[
                                    {
                                        "kind": "callable",
                                        "name": "count_cells_single_channel",
                                        "module": (
                                            "openhcs.processing.backends.analysis."
                                            "cell_counting_cpu"
                                        ),
                                        "qualname": "count_cells_single_channel",
                                        "import_path": (
                                            "openhcs.processing.backends.analysis."
                                            "cell_counting_cpu."
                                            "count_cells_single_channel"
                                        ),
                                    },
                                    {
                                        "min_cell_area": 40,
                                    },
                                ],
                                resolved_value=[
                                    {
                                        "kind": "callable",
                                        "name": "count_cells_single_channel",
                                        "module": (
                                            "openhcs.processing.backends.analysis."
                                            "cell_counting_cpu"
                                        ),
                                        "qualname": "count_cells_single_channel",
                                        "import_path": (
                                            "openhcs.processing.backends.analysis."
                                            "cell_counting_cpu."
                                            "count_cells_single_channel"
                                        ),
                                    },
                                    {
                                        "min_cell_area": 40,
                                    },
                                ],
                            ),
                        ),
                    ),
                ),
            )

    function_catalog_service = _FunctionCatalogService()
    result = ObjectStateFieldHelpService(
        _UiBridgeService(),
        function_catalog_service,
    ).describe(
        UiObjectStateFieldHelpRequest(
            object_state_scope_id="plate::step_0",
            field_path="func",
            max_description_chars=1_000,
        ),
        "ui-connection",
    )

    assert result.errors == ()
    assert "Callable value:" in (result.description or "")
    assert "count_cells_single_channel" in (result.description or "")
    assert "min_cell_area=40" in (result.description or "")
    assert function_catalog_service.import_paths == [
        (
            "openhcs.processing.backends.analysis.cell_counting_cpu."
            "count_cells_single_channel"
        )
    ]
    assert (
        "function_id: openhcs:analysis_cell_counting_cpu_count_cells_single_channel"
        in (result.description or "")
    )
    assert "active kwargs:" in (result.description or "")
    assert "- min_cell_area, type=int, default=10" in (result.description or "")
    assert "runtime supplied: image_stack (runtime_primary_input)" in (
        result.description or ""
    )
    assert "artifact outputs: cell_counts:special, segmentation_masks:special" in (
        result.description or ""
    )
    assert "doc excerpt:" in (result.description or "")
    assert "Minimum area for valid cells." in (result.description or "")


def test_object_state_field_help_describes_runtime_supplied_function_parameter():
    target_name = (
        f"{runtime_artifact_parameter_example.__module__}."
        f"{runtime_artifact_parameter_example.__qualname__}"
    )

    class _FunctionCatalogService:
        def __init__(self):
            self.import_paths = []

        def get_by_import_path(
            self,
            import_path,
            *,
            max_doc_chars,
            compact_signature,
        ):
            self.import_paths.append(import_path)
            return FunctionDetail(
                schema_version=SCHEMA_VERSION,
                entry=FunctionCatalogEntry(
                    function_id="test:runtime_artifact_parameter_example",
                    import_path=import_path,
                    name="runtime_artifact_parameter_example",
                    module=runtime_artifact_parameter_example.__module__,
                    library="test",
                    signature=(
                        "runtime_artifact_parameter_example(overlap_ratio, ...)"
                    ),
                    summary="Compute tile positions.",
                ),
                parameters=(
                    FunctionParameterSpec(
                        name="image_stack",
                        annotation="ndarray",
                        default_repr=None,
                        required=False,
                        supplied_by=FunctionParameterSource.PRIMARY_INPUT,
                    ),
                    FunctionParameterSpec(
                        name="grid_dimensions",
                        annotation="tuple[int, int]",
                        default_repr=None,
                        required=False,
                        supplied_by=FunctionParameterSource.ARTIFACT_INPUT,
                    ),
                    FunctionParameterSpec(
                        name="overlap_ratio",
                        annotation="float",
                        default_repr="0.1",
                        required=False,
                    ),
                ),
                runtime_contract=FunctionRuntimeContractSummary(
                    callable_kind="openhcs_function",
                    artifact_inputs=(
                        FunctionArtifactSpec(
                            name="grid_dimensions",
                            kind="metadata",
                        ),
                    ),
                ),
                doc="Compute tile positions.",
            )

    class _UiBridgeService:
        def list_object_state_scopes(self, request, connection):
            assert connection == "ui-connection"
            assert request.include_field_values is True
            assert request.include_field_descriptions is True
            assert request.field_paths == ("grid_dimensions",)
            return UiObjectStateScopeCatalog(
                schema_version=SCHEMA_VERSION,
                object_state_token=1,
                current_branch="main",
                current_snapshot_index=0,
                scopes=(
                    UiObjectStateScopeSummary(
                        schema_version=SCHEMA_VERSION,
                        identity=UiObjectStateScopeIdentity(
                            object_state_scope_id="plate::step::function_0",
                        ),
                        object_type=target_name,
                        parameter_count=3,
                        dirty_field_count=0,
                        signature_diff_field_count=0,
                        fields=(
                            UiObjectStateFieldSummary(
                                schema_version=SCHEMA_VERSION,
                                address=UiSemanticAddress(
                                    object_state_scope_id=("plate::step::function_0"),
                                    field_path="grid_dimensions",
                                ),
                                field_name="grid_dimensions",
                                container_path="",
                                object_state_path_type=target_name,
                                raw_value_type="NoneType",
                                resolved_value_type="NoneType",
                                dirty=False,
                                signature_diff=False,
                                last_changed=False,
                                parameter_description="Logical tile grid.",
                                raw_value_is_none=True,
                                resolved_value_is_none=True,
                            ),
                        ),
                    ),
                ),
            )

    function_catalog_service = _FunctionCatalogService()
    result = ObjectStateFieldHelpService(
        _UiBridgeService(),
        function_catalog_service,
    ).describe(
        UiObjectStateFieldHelpRequest(
            object_state_scope_id="plate::step::function_0",
            field_path="grid_dimensions",
            max_description_chars=1_000,
        ),
        "ui-connection",
    )

    assert result.errors == ()
    assert function_catalog_service.import_paths == [target_name]
    assert result.summary == "• grid_dimensions (tuple[int, int])"
    assert "Logical tile grid." in (result.description or "")
    assert (
        "Function parameter contract: grid_dimensions is supplied by OpenHCS "
        "(runtime_artifact_input); do not pass it as a FunctionStep kwarg."
    ) in (result.description or "")
    assert "- type: tuple[int, int]" in (result.description or "")
    assert "- artifact: grid_dimensions:metadata required=True" in (
        result.description or ""
    )
