"""Static ownership gates for the CellProfiler relationship hook caller."""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RELATIONSHIP_BACKEND_PATH = (
    PROJECT_ROOT / "openhcs/processing/backends/cellprofiler/relationships.py"
)


def test_relationship_hook_caller_uses_callable_artifact_authorities() -> None:
    """Keep wrapper-owned identity and relation lookups out of the backend caller."""

    tree = ast.parse(RELATIONSHIP_BACKEND_PATH.read_text(encoding="utf-8"))
    imported_modules = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    dotted_calls = {
        f"{node.func.value.id}.{node.func.attr}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
    }

    assert "openhcs.core.module_artifact_contract" not in imported_modules
    assert "ModuleArtifactContract" not in imported_names
    assert {
        "relationship_declaration",
        "relationship_endpoint_specs",
        "declared_inputs",
    }.isdisjoint(attributes)
    assert "CellProfilerModule.require_module" not in dotted_calls
    assert "CellProfilerModule.for_function_name" in dotted_calls

    hook = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "relationship_distance_measurements_apply"
    )
    assert [argument.arg for argument in hook.args.args] == [
        "cls",
        "callable_contract",
        "relationship_spec",
    ]
    assert isinstance(hook.args.args[1].annotation, ast.Constant)
    assert hook.args.args[1].annotation.value == "CallableContract"
    assert isinstance(hook.args.args[2].annotation, ast.Name)
    assert hook.args.args[2].annotation.id == "ArtifactSpec"
    assert "artifact_outputs" in {
        node.attr for node in ast.walk(hook) if isinstance(node, ast.Attribute)
    }
    assert "relations" in {
        node.attr for node in ast.walk(hook) if isinstance(node, ast.Attribute)
    }


def test_relate_objects_hook_uses_the_exact_declared_relationship_spec() -> None:
    """Select only the exact parent-to-child output declared by the callable."""

    from openhcs.core.artifacts import (
        ArtifactSpec,
        ObjectLabelsArtifactType,
        RelationshipsArtifactType,
    )
    from openhcs.core.callable_contract import CallableContract
    from openhcs.core.runtime_relationships import ObjectRelationshipDeclaration
    from openhcs.processing.backends.cellprofiler.relationships import (
        RelateObjectsModule,
    )

    parent = ArtifactSpec.input("Parents", ObjectLabelsArtifactType)
    child = ArtifactSpec.input("Children", ObjectLabelsArtifactType)
    parent_relationship = ArtifactSpec.output(
        "parent-relationship",
        RelationshipsArtifactType,
        relations=(
            ObjectRelationshipDeclaration.parent_child(
                source=parent.ref(),
                target=child.ref(),
                producer_module_number=1,
            ),
        ),
    )
    child_relationship = ArtifactSpec.output(
        "child-relationship",
        RelationshipsArtifactType,
        relations=(
            ObjectRelationshipDeclaration(
                source=child.ref(),
                target=parent.ref(),
                relationship_type="Child",
                source_role="child",
                target_role="parent",
                source_id_field="child_id",
                target_id_field="parent_id",
                producer_module_number=1,
            ),
        ),
    )
    callable_contract = CallableContract(
        func=lambda image: image,
        function_name="relate_objects",
        module_name="RelateObjects",
    )
    callable_contract = replace(
        callable_contract,
        metadata=replace(
            callable_contract.metadata,
            artifact_inputs=(parent, child),
            artifact_outputs=(parent_relationship, child_relationship),
        ),
    )

    assert RelateObjectsModule.relationship_distance_measurements_apply(
        callable_contract,
        parent_relationship,
    )
    assert not RelateObjectsModule.relationship_distance_measurements_apply(
        callable_contract,
        child_relationship,
    )
    with pytest.raises(ValueError, match="does not declare exact relationship output"):
        RelateObjectsModule.relationship_distance_measurements_apply(
            callable_contract,
            replace(parent_relationship, required=False),
        )


def test_relationship_rows_pass_callable_contract_and_exact_output_entry() -> None:
    """Exercise the migrated caller through endpoint resolution and row dispatch."""

    from openhcs.core.artifacts import (
        ArtifactSpec,
        ObjectLabelsArtifactType,
        RelationshipsArtifactType,
    )
    from openhcs.core.callable_contract import CallableContract
    from openhcs.core.measurement_row_materialization import (
        MeasurementSparseColumnarRows,
    )
    from openhcs.core.runtime_relationships import (
        DirectedObjectRelationshipPayload,
        ObjectRelationship,
        ObjectRelationshipDeclaration,
    )
    from openhcs.processing.backends.cellprofiler.relationships import (
        RelateObjectsRelationshipMeasurementRows,
    )

    parent = ArtifactSpec.input("Parents", ObjectLabelsArtifactType)
    child = ArtifactSpec.input("Children", ObjectLabelsArtifactType)
    declaration = ObjectRelationshipDeclaration.parent_child(
        source=parent.ref(),
        target=child.ref(),
        producer_module_number=1,
    )
    relationship_spec = ArtifactSpec.output(
        "parent-relationship",
        RelationshipsArtifactType,
        relations=(declaration,),
    )
    callable_contract = CallableContract(
        func=lambda image: image,
        function_name="relate_objects",
        module_name="RelateObjects",
    )
    callable_contract = replace(
        callable_contract,
        metadata=replace(
            callable_contract.metadata,
            artifact_inputs=(parent, child),
            artifact_outputs=(relationship_spec,),
        ),
    )
    relationship = ObjectRelationship.from_payload(
        name=relationship_spec.name,
        declaration=declaration,
        payload=DirectedObjectRelationshipPayload(
            source_ids=(1,),
            target_ids=(1,),
        ),
    )
    distance_calls: list[tuple[ArtifactSpec, ArtifactSpec, ObjectRelationship]] = []

    class FocusedRows(RelateObjectsRelationshipMeasurementRows):
        def output_entries(self):
            return ((relationship_spec, declaration, relationship),)

        def child_count_rows(self, **_kwargs):
            return MeasurementSparseColumnarRows.from_rows((), fields=())

        def parent_rows(self, **_kwargs):
            return MeasurementSparseColumnarRows.from_rows((), fields=())

        def per_parent_means_enabled(self) -> bool:
            return False

        def distance_rows(self, *, parent_spec, child_spec, payload, **_kwargs):
            distance_calls.append((parent_spec, child_spec, payload))
            return MeasurementSparseColumnarRows.from_rows((), fields=())

    rows = FocusedRows(SimpleNamespace(callable_contract=callable_contract)).rows()

    assert rows.row_count() == 0
    assert distance_calls == [(parent, child, relationship)]
