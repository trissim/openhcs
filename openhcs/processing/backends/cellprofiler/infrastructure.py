"""CellProfiler infrastructure module declarations."""

from __future__ import annotations
from openhcs.interop.cellprofiler.module_declarations import (
    InfrastructureCellProfilerModule,
)


class LoadDataModule(InfrastructureCellProfilerModule):
    """Declare OpenHCS source metadata handling for CellProfiler LoadData."""

    module_name = "LoadData"
    function_name = "load_data"
    validated = True
    contract = None
    confidence = 1.0
    infrastructure_import_note = (
        "LoadData -> handled by plate_path + openhcs_metadata.json"
    )


class ExportToSpreadsheetModule(InfrastructureCellProfilerModule):
    """Declare OpenHCS table materialization handling for ExportToSpreadsheet."""

    module_name = "ExportToSpreadsheet"
    function_name = "export_to_spreadsheet"
    validated = True
    contract = None
    confidence = 1.0
    infrastructure_import_note = (
        "ExportToSpreadsheet -> handled by @special_outputs(csv_materializer(...))"
    )
    infrastructure_exports_tables = True
