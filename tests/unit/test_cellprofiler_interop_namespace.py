"""CellProfiler interop namespace ownership tests."""

from benchmark.cellprofiler_compat import measurement_dialect as benchmark_dialect
from benchmark.cellprofiler_compat import measurement_scope as benchmark_scope
from benchmark.converter import parser as benchmark_parser
from benchmark.converter import settings_binder as benchmark_settings_binder
from benchmark.converter import source_schema as benchmark_source_schema
from openhcs.interop import cellprofiler
from openhcs.interop.cellprofiler import import_records
from openhcs.interop.cellprofiler import measurement_dialect
from openhcs.interop.cellprofiler import measurement_scope
from openhcs.interop.cellprofiler import module_roles
from openhcs.interop.cellprofiler import module_semantics
from openhcs.interop.cellprofiler import parser as product_parser
from openhcs.interop.cellprofiler import runtime as product_runtime
from openhcs.interop.cellprofiler import settings_binder as product_settings_binder
from openhcs.interop.cellprofiler import source_schema as product_source_schema
from openhcs.core.runtime_invocation import RuntimeImageExecutionContext


def test_cellprofiler_measurement_dialect_has_product_namespace_owner():
    """The product namespace owns CellProfiler measurement equivalence semantics."""
    assert (
        cellprofiler.CELLPROFILER_MEASUREMENT_DIALECT
        is measurement_dialect.CELLPROFILER_MEASUREMENT_DIALECT
    )
    assert (
        benchmark_dialect.CELLPROFILER_MEASUREMENT_DIALECT
        is measurement_dialect.CELLPROFILER_MEASUREMENT_DIALECT
    )
    assert (
        benchmark_dialect.cellprofiler_runtime_equivalence_policy
        is measurement_dialect.cellprofiler_runtime_equivalence_policy
    )


def test_cellprofiler_measurement_scope_has_product_namespace_owner():
    """Benchmark scope imports remain compatibility aliases."""
    assert (
        cellprofiler.CellProfilerMeasurementTargetScope
        is measurement_scope.CellProfilerMeasurementTargetScope
    )
    assert (
        benchmark_scope.CellProfilerMeasurementTargetScope
        is measurement_scope.CellProfilerMeasurementTargetScope
    )
    assert (
        benchmark_scope.CELLPROFILER_MEASUREMENT_TARGET_SCOPE_KWARG
        == measurement_scope.CELLPROFILER_MEASUREMENT_TARGET_SCOPE_KWARG
    )


def test_cellprofiler_import_records_have_product_namespace_owner():
    """Product namespace owns `.cppipe` import provenance records."""
    assert cellprofiler.CellProfilerPipelineProvenance is (
        import_records.CellProfilerPipelineProvenance
    )
    assert cellprofiler.CellProfilerPipelineImportResult is (
        import_records.CellProfilerPipelineImportResult
    )
    assert cellprofiler.CellProfilerModuleRole is import_records.CellProfilerModuleRole
    assert hasattr(cellprofiler, "CellProfilerPipelineImportRequest")
    assert hasattr(cellprofiler, "CellProfilerPipelineImporter")
    assert hasattr(cellprofiler, "CellProfilerDialectCompiler")


def test_cellprofiler_module_roles_have_product_namespace_owner():
    """Product namespace owns `.cppipe` module role semantics."""
    assert cellprofiler.CellProfilerModuleRole is module_roles.CellProfilerModuleRole
    assert cellprofiler.cellprofiler_module_role is (
        module_roles.cellprofiler_module_role
    )
    assert (
        cellprofiler.INFRASTRUCTURE_MODULE_NAMES
        is module_roles.INFRASTRUCTURE_MODULE_NAMES
    )


def test_cellprofiler_module_semantics_have_product_namespace_owner():
    """Product namespace owns CellProfiler manual module semantics."""
    assert (
        cellprofiler.CellProfilerModuleSemantics
        is module_semantics.CellProfilerModuleSemantics
    )
    assert (
        cellprofiler.CellProfilerModuleDimensionality
        is module_semantics.CellProfilerModuleDimensionality
    )
    assert (
        cellprofiler.cellprofiler_module_semantics
        is module_semantics.cellprofiler_module_semantics
    )


def test_cellprofiler_parser_has_product_namespace_owner():
    """Product namespace owns `.cppipe` syntax parsing semantics."""
    assert cellprofiler.CPPipeParser is product_parser.CPPipeParser
    assert benchmark_parser.CPPipeParser is product_parser.CPPipeParser
    assert benchmark_parser.ModuleBlock is product_parser.ModuleBlock


def test_cellprofiler_source_schema_has_product_namespace_owner():
    """Product namespace owns `.cppipe` source schema lowering."""
    assert cellprofiler.compile_image_schema is product_source_schema.compile_image_schema
    assert benchmark_source_schema.compile_image_schema is (
        product_source_schema.compile_image_schema
    )
    assert cellprofiler.SetupModuleCompiler is product_source_schema.SetupModuleCompiler


def test_cellprofiler_settings_binder_has_product_namespace_owner():
    """Product namespace owns `.cppipe` setting binding semantics."""
    assert cellprofiler.SettingsBinder is product_settings_binder.SettingsBinder
    assert benchmark_settings_binder.SettingsBinder is (
        product_settings_binder.SettingsBinder
    )
    assert cellprofiler.normalize_cellprofiler_setting_name is (
        product_settings_binder.normalize_cellprofiler_setting_name
    )


def test_cellprofiler_runtime_records_extend_core_invocation_semantics():
    """CellProfiler runtime records should compose generic OpenHCS invocation types."""
    assert (
        cellprofiler.CellProfilerInvocationRequest
        is product_runtime.CellProfilerInvocationRequest
    )
    assert issubclass(
        product_runtime.CellProfilerImageExecutionContext,
        RuntimeImageExecutionContext,
    )
    assert cellprofiler.CellProfilerMeasurementImageDomain is (
        product_runtime.CellProfilerMeasurementImageDomain
    )
