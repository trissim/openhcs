"""Move measurement-feature ownership off the CellProfiler registry root."""

from __future__ import annotations

from pathlib import Path

import libcst as cst


ROOT_PATH = Path("openhcs/interop/cellprofiler/module_declarations.py")
OWNER_PATH = Path("openhcs/interop/cellprofiler/module_measurement_features.py")

ATTRIBUTE_NAMES = frozenset(
    {
        "measurement_feature_part_aliases",
        "measurement_feature_part_rewrites",
        "measurement_category_prefixes",
        "measurement_source_feature_prefixes",
        "calculated_measurement_feature_prefixes",
        "numbered_measurement_feature_prefix_aliases",
        "directional_pair_feature_aliases",
        "scale_qualified_measurement_feature_prefixes",
        "pair_correlation_feature_name",
        "pair_regression_slope_feature_name",
        "undirected_pair_feature_names",
        "threshold_sensitive_pair_feature_names",
    }
)

METHOD_NAMES = frozenset(
    {
        "declared_authority_types",
        "derived_measurement_feature_relation_declarations",
        "measurement_feature_relation_declarations",
        "declared_measurement_feature_family_parts",
        "source_qualified_measurement_feature_family_parts",
        "declared_source_qualified_measurement_feature_family_parts",
        "source_qualified_measurement_feature_types",
        "measurement_feature_marker_types_for_key",
        "_measurement_feature_marker_types_for_key_payload",
        "measurement_feature_types",
        "owns_measurement_feature_name",
        "owns_measurement_feature_parts",
        "owns_primary_measurement_feature_name",
        "experiment_measurement_tables",
        "derive_experiment_measurement_tables",
        "alternative_measurement_feature_part_aliases",
        "measurement_feature_part_rewrite_declarations",
        "measurement_category_prefix_declarations",
        "primary_measurement_category_prefix_declarations",
        "measurement_source_feature_prefix_declarations",
        "calculated_measurement_feature_prefix_declarations",
        "numbered_measurement_feature_prefix_alias_declarations",
        "scale_qualified_measurement_feature_prefix_declarations",
        "directional_pair_feature_alias_declarations",
        "pair_correlation_feature_name_declaration",
        "pair_regression_slope_feature_name_declaration",
        "undirected_pair_feature_name_declarations",
        "threshold_sensitive_pair_feature_name_declarations",
    }
)


def _assigned_name(statement: cst.BaseStatement) -> str | None:
    if not isinstance(statement, cst.SimpleStatementLine) or len(statement.body) != 1:
        return None
    assignment = statement.body[0]
    if isinstance(assignment, cst.Assign) and len(assignment.targets) == 1:
        target = assignment.targets[0].target
    elif isinstance(assignment, cst.AnnAssign):
        target = assignment.target
    else:
        return None
    return target.value if isinstance(target, cst.Name) else None


def _declaration_name(statement: cst.BaseStatement) -> str | None:
    if isinstance(statement, (cst.FunctionDef, cst.ClassDef)):
        return statement.name.value
    return _assigned_name(statement)


def main() -> None:
    root_module = cst.parse_module(ROOT_PATH.read_text())
    moved: list[cst.BaseStatement] = []
    root_body: list[cst.BaseStatement] = []
    for statement in root_module.body:
        if not isinstance(statement, cst.ClassDef) or statement.name.value != "CellProfilerModule":
            root_body.append(statement)
            continue
        retained: list[cst.BaseStatement] = []
        for member in statement.body.body:
            name = _declaration_name(member)
            if name in ATTRIBUTE_NAMES or name in METHOD_NAMES:
                moved.append(member)
            elif name == "measurement_record_excluded_fields":
                continue
            else:
                retained.append(member)
        bases = tuple(
            cst.Arg(cst.Name("CellProfilerMeasurementFeatureOwner"))
            if isinstance(base.value, cst.Name)
            and base.value.value == "RuntimeMeasurementFeatureOwner"
            else base
            for base in statement.bases
        )
        root_body.append(
            statement.with_changes(
                bases=bases,
                body=statement.body.with_changes(body=tuple(retained)),
            )
        )

    expected = len(ATTRIBUTE_NAMES) + len(METHOD_NAMES)
    if len(moved) != expected:
        raise RuntimeError(f"Expected {expected} declarations, found {len(moved)}.")

    owner_module = cst.parse_module(OWNER_PATH.read_text())
    owner_class = cst.ClassDef(
        name=cst.Name("CellProfilerMeasurementFeatureOwner"),
        bases=(cst.Arg(cst.Name("RuntimeMeasurementFeatureOwner")),),
        body=cst.IndentedBlock(body=tuple(moved)),
        leading_lines=(cst.EmptyLine(), cst.EmptyLine()),
    )
    ROOT_PATH.write_text(root_module.with_changes(body=tuple(root_body)).code)
    OWNER_PATH.write_text(
        owner_module.with_changes(body=(*owner_module.body, owner_class)).code
    )


if __name__ == "__main__":
    main()
