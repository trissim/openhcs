# Artifact Output Group Lineage Plan

Date: 2026-07-01
Branch: `benchmark-platform`
Worktree: `/home/ts/code/projects/openhcs`

## Current Implementation Status

Implemented in this worktree:

- `ArtifactPlan`, `ArtifactType`, `ArtifactSpec`, `ArtifactSpecRef`, and
  `ArtifactSpecRelation` are the core product terms.
- relation validation and planner application are full-ref based; relation
  sources are not looked up by bare input/output names.
- `ArtifactPlanKeySelector` selects by registered `ArtifactPlan` role through
  `ArtifactSpecCollection`.
- `InvocationArtifactDeclarations` stores one canonical artifact tuple and
  projects `inputs`/`outputs` by `ArtifactPlan` role.
- CellProfiler artifact-setting symbols carry `ArtifactSpec`; the old closed
  `ArtifactSettingRole` / `ArtifactSettingDirection` table is gone.
- CellProfiler artifact capabilities are registered product terms over
  `ArtifactPlan` role and `ArtifactType` payload. Module mixins declare those
  capabilities through MRO-collected `artifact_capabilities`, keeping the
  capability registry out of the `CellProfilerModule` `AutoRegisterMeta` MRO.
- `Align`, `ColorToGray`, `GrayToColor`, `UnmixColors`, `DefineGrid`, and
  `IdentifyObjectsInGrid` use the shared capability binder instead of local
  `ArtifactSpec.input(...)` / `ArtifactSpec.output(...)` construction.
- `FilterObjects` output declarations use `ArtifactSpecRef` plus
  `GroupLineageSourceRelation`; its optional parent/child relationship reads now
  use `RelationshipArtifactInputCapability`, and the module inherits both
  relationship input and relationship output capabilities.
- `FilterObjects.Plan.output_artifact_specs(...)` emits through
  `FilterObjectsModule.artifact_spec(...)`, not directly through
  `capability_type.spec(...)`, so the module's MRO-collected capability set is
  load-bearing for the generated relation-bearing outputs.
- `Threshold` keeps the default CellProfiler measurement artifact but declares
  image output before threshold measurements, matching the callable ABI
  `output_image, ThresholdResult(...)` without changing the default
  measurement-first object-module ABI.
- `ModuleArtifactContract` now has a registered owner-partition axis:
  `ModuleArtifactContractPartition` with
  `SourceArtifactInputPartition`, `RuntimeArtifactInputPartition`,
  `RecordedArtifactOutputPartition`, and `DeclaredArtifactOutputPartition`.
  It stores one canonical `items` tuple of `ModuleArtifactContractItem` values.
  AST evidence for dataclass fields is now:
  `['module_name', 'items', 'required_variable_components']`.
- The transitional `ModuleArtifactContract` constructor shim is gone. An AST
  audit across `openhcs/` and `tests/` reports `old_keyword_violations 0` for
  `inputs`, `runtime_artifact_inputs`, `outputs`, and `declared_outputs`.
- Production `ModuleArtifactContract(...)` construction under `openhcs/` now
  uses `items=` only. The AST audit shows only the CellProfiler symbol table,
  pipeline generator, and generated sidecar reader construct contracts, and all
  three use the partitioned item form.
- Generated CellProfiler contract sidecars are now schema version `3` and
  serialize `items` as `{"partition": ..., "spec": ...}` records. The old
  sidecar `inputs`, `runtime_artifact_inputs`, `outputs`, and
  `declared_outputs` keys are not emitted. Sidecar schema decoding now treats
  writer-emitted fields as required keys; there are no `payload.get(...)`
  fallbacks in `generated_pipeline.py` sidecar decoding.
- The generated artifact-contract comment sections iterate registered section
  declarations directly through `ArtifactContractCommentSection.__registry__.values()`.
  The previous numeric `order = 10/20/30/40` field is gone from that family.
- The unused `ArtifactPlan.registered_plan_types()` wrapper is gone; artifact
  plan code uses the registered family directly through `ArtifactPlan.__registry__`.
- Direct CellProfiler module-boundary `ArtifactSpec(...)` construction has been
  collapsed to the generic capability factory:
  `rg -n "ArtifactSpec\.(input|output)\(|ArtifactSpec\(" openhcs/processing/backends/cellprofiler`
  returns only `module_classes.py`, where `CellProfilerArtifactCapability.spec`
  owns the product construction.
- Cleanup grep for the removed lineage/planner surfaces is clean for
  `ArtifactKind`, `ArtifactOutputGroupLineage`, `output_group_lineage`,
  `OutputRole`, `select_input_plan_keys`, `select_output_plan_keys`,
  `.input_names`, `.output_names`, `artifact_specs_by_plan_type`,
  `group_lineage_input_name`, `validate_group_lineage_inputs`,
  `source_plan_type`, `by_name_and_kind`, `of_kind`,
  `object_only_input_group_scope`, `output_groups_from_input_identity`, and
  `RelationshipArtifactOutputCapability.spec(` in the planned core,
  CellProfiler, and FilterObjects surfaces.
- Focused verification currently passes:
  `/home/ts/.pyenv/versions/3.12.0/bin/python -m pytest tests/unit/test_artifact_contract_preview.py tests/unit/test_pycodify_formatters.py tests/unit/test_function_patterns.py tests/unit/test_path_planner_materialization.py tests/unit/test_cellprofiler_symbol_table.py tests/unit/test_cellprofiler_generated_pipeline_execution.py -q`
  completed with `168 passed`. This includes a regression assertion that
  `FilterObjectsModule.Plan.output_artifact_specs(...)` rejects a module type
  that does not declare the requested artifact capability.
- Additional migrated-fixture verification passes:
  `/home/ts/.pyenv/versions/3.12.0/bin/python -m pytest tests/unit/test_cellprofiler_runtime_callable_introspection.py tests/unit/test_function_step_transport.py tests/unit/test_function_invocation_callable_resolver.py tests/unit/test_compilation_session.py tests/unit/pyqt_gui/test_plate_manager_widget.py -q`
  completed with `60 passed`, and
  `/home/ts/.pyenv/versions/3.12.0/bin/python -m pytest tests/unit/test_cellprofiler_module_execution.py tests/unit/test_cellprofiler_runtime_adapter.py -q`
  completed with `527 passed`.
- Parity smoke for
  `ExampleImagingFlowCytometryObjectsInGrid` completed without runtime errors
  after the partition/sidecar tightening:
  `summary.csv` in
  `/tmp/openhcs_cp30_partition_contract_20260702_053356` reports
  `median_openhcs_execution_seconds=30.369418947999293`,
  `median_total_phase_speedup=4.200920928913542`,
  `meets_speedup_target=True`, and `min_parity_accuracy=0.0`.

Post-audit conclusion on module-contract projections:

- Runtime/generator reads of `contract.inputs`,
  `contract.runtime_artifact_inputs`, `contract.outputs`, and
  `contract.declared_outputs` are partition projections over the canonical
  `ModuleArtifactContract.items` tuple, not independent stored fields or
  compatibility constructor paths. The sites using them are asking for those
  semantic partitions: source-bound inputs, runtime artifact inputs, recorded
  outputs, or declared-before-pruning outputs.
- Exact persisted identity now uses `contract.items` directly: generated
  sidecars write/read partitioned `ModuleArtifactContractItem` records, and
  pycodify import collection reads `contract.artifact_specs`. No current
  runtime/generator call site reconstructs a local
  `{ArtifactInputPlan: ..., ArtifactOutputPlan: ...}` projection map.

Authoritative reconciliation of draft snippets below:

- The relation base is implemented as `ArtifactSpecRelation`, not the earlier
  draft name `ArtifactSpecRelationRef`. The relation value still carries a full
  `ArtifactSpecRef` source and is selected through the registered relation
  family.
- `ArtifactType` uses the existing `AutoRegisterMeta` value key named `value`
  rather than the draft spelling `type_key`. Sidecars serialize
  `artifact_type.value`, and decoders resolve through `ArtifactType.coerce(...)`
  / `ArtifactType.__registry__`.
- Generated contract sidecars are schema version `3`, not the draft schema
  version `2`, because the final `ModuleArtifactContract` storage is the
  partitioned `items` product. The version bump intentionally rejects older
  split-field sidecars instead of carrying compatibility decode branches.
- `ModuleArtifactContract` stores one canonical partitioned item tuple:
  `ModuleArtifactContract.items`. Its `artifact_specs` property is the unified
  `ArtifactSpecCollection` projection over those items. Source inputs, runtime
  inputs, recorded outputs, and declared outputs are registered
  `ModuleArtifactContractPartition` projections, not parallel constructor
  fields and not owner-local `{ArtifactInputPlan: ..., ArtifactOutputPlan: ...}`
  maps.
- Draft snippets below are retained as design history where they explain the
  algebra; this reconciliation and the implementation evidence above are the
  current target where they differ.

## Target

Fix `FilterObjects` output grouping without planner heuristics, without parallel
lineage declarations, and without input/output hardcoding in the abstraction.

The design rule is SSOT:

- `ArtifactPlan` is the registered role family.
- `ArtifactInputPlan` and `ArtifactOutputPlan` are role implementations, not a
  second enum and not keys mirrored by owners.
- `ArtifactType` is the registered payload/category family.
- `ArtifactSpec` is the declared product of one `ArtifactPlan` role and one
  `ArtifactType` category.
- Owners hold one `ArtifactSpecCollection`.
- Relations are nominal tags on `ArtifactSpec` and are validated/applied by
  iterating their registered behavior classes.
- Adding a new artifact role or a new relation behavior requires adding the new
  registered class, not touching every owner/callsite.

## Non-Negotiables

- no artifact-category-based output-group inference in `path_planner.py`;
- no `OutputRole` to artifact-category switch;
- no `ArtifactOutputGroupLineage` record;
- no lineage-specific collection class;
- no `ArtifactPlanDeclaration`;
- no owner-local `{ArtifactInputPlan: ..., ArtifactOutputPlan: ...}` maps;
- no `artifact_specs_by_plan_type` projection properties;
- no fallback `.get(...)` or missing-role defaulting in declaration code;
- no public compatibility shims for old selector names;
- CellProfiler modules instantiate core relation tags at the artifact boundary;
  they do not own relation validation, serialization, or planner behavior.

## 1. Make ArtifactSpec The Core Product Term

Edit `openhcs/core/artifacts.py`.

Replace the `ArtifactKind` enum with a registered nominal artifact category
family. This is core OpenHCS semantics, not CellProfiler-specific adapter
state:

```python
class ArtifactType(ABC, metaclass=AutoRegisterMeta):
    """Core payload/category type for an OpenHCS artifact."""

    __registry_key__ = "value"
    __skip_if_no_key__ = True

    value: ClassVar[str | None] = None
    payload_shape: ClassVar[ArtifactPayloadShape]
    uses_label_representation_payload_shape: ClassVar[bool] = False
    materialization_uses_source_identity_filename: ClassVar[bool] = False
    participates_in_measurement_source_names: ClassVar[bool] = False
    participates_in_main_flow_output: ClassVar[bool] = False
    participates_in_axis_plane_identity: ClassVar[bool] = False
    participates_in_object_domain_scope: ClassVar[bool] = False
    participates_in_pairwise_object_domain_input: ClassVar[bool] = False
    payload_description: ClassVar[str | None] = None
```

Concrete artifact types replace `ArtifactKind` members directly:

```python
class SpecialArtifactType(ArtifactType):
    value = "special"
    payload_shape = ArtifactPayloadShape.ANY


class ImageArtifactType(ArtifactType):
    value = "image"
    payload_shape = ArtifactPayloadShape.ARRAY
    materialization_uses_source_identity_filename = True
    participates_in_measurement_source_names = True
    participates_in_main_flow_output = True


class ObjectLabelArtifactType(ArtifactType):
    value = "object_labels"
    payload_shape = ArtifactPayloadShape.ARRAY
    materialization_uses_source_identity_filename = True
    participates_in_main_flow_output = True
    participates_in_object_domain_scope = True
    participates_in_pairwise_object_domain_input = True
    uses_label_representation_payload_shape = True
    payload_description = "object_labels payload"


class MeasurementArtifactType(ArtifactType):
    value = "measurements"
    payload_shape = ArtifactPayloadShape.TABLE
    materialization_uses_source_identity_filename = True
    participates_in_axis_plane_identity = True


class RelationshipArtifactType(ArtifactType):
    value = "relationships"
    payload_shape = ArtifactPayloadShape.TABLE
    materialization_uses_source_identity_filename = True
    participates_in_axis_plane_identity = True
    participates_in_object_domain_scope = True
    participates_in_pairwise_object_domain_input = True


class TableArtifactType(ArtifactType):
    value = "table"
    payload_shape = ArtifactPayloadShape.TABLE


class SpatialGridArtifactType(ArtifactType):
    value = "spatial_grid"
    payload_shape = ArtifactPayloadShape.MAPPING
    participates_in_pairwise_object_domain_input = True
    payload_description = "spatial grid mapping"


class MetadataArtifactType(ArtifactType):
    value = "metadata"
    payload_shape = ArtifactPayloadShape.MAPPING
    payload_description = "metadata mapping"
```

Move runtime identity fields to the same nominal type axis:

- `ArtifactKey.kind` becomes `ArtifactKey.artifact_type`;
- `ArtifactPlan.kind` becomes `ArtifactPlan.artifact_type`;
- JSON payloads write `artifact_type.value`;
- lookups deserialize through `ArtifactType.__registry__`;
- materialization policies key by `type[ArtifactType]`, not enum members.

Do this as an AST-backed migration, not by leaving public `kind` compatibility
properties in the core model.

`ArtifactSpec` becomes the declared product of the two registered core axes:

```python
@dataclass(frozen=True)
class ArtifactSpec:
    """Declared artifact contract for one plan role and one artifact type."""

    name: str
    plan_type: type["ArtifactPlan"]
    artifact_type: type["ArtifactType"]
    materialization: ArtifactMaterializationPayload | None = None
    required: bool = True
    sidecar_role: ArtifactSidecarRole | None = None
    relations: tuple["ArtifactSpecRelation", ...] = ()

    @classmethod
    def input(
        cls,
        name: str,
        artifact_type: type["ArtifactType"],
        **kwargs,
    ) -> "ArtifactSpec":
        return cls(
            name=name,
            plan_type=ArtifactInputPlan,
            artifact_type=artifact_type,
            **kwargs,
        )

    @classmethod
    def output(
        cls,
        name: str,
        artifact_type: type["ArtifactType"],
        **kwargs,
    ) -> "ArtifactSpec":
        return cls(
            name=name,
            plan_type=ArtifactOutputPlan,
            artifact_type=artifact_type,
            **kwargs,
        )
```

`ArtifactSpec.__post_init__` validates:

```python
def __post_init__(self) -> None:
    if not isinstance(self.plan_type, type) or not issubclass(
        self.plan_type,
        ArtifactPlan,
    ):
        raise TypeError("ArtifactSpec.plan_type must be an ArtifactPlan type.")
    if self.plan_type not in ArtifactPlan.__registry__.values():
        raise ValueError(
            f"{self.plan_type.__name__} is not a registered ArtifactPlan type."
        )
    if not isinstance(self.artifact_type, type) or not issubclass(
        self.artifact_type,
        ArtifactType,
    ):
        raise TypeError("ArtifactSpec.artifact_type must be an ArtifactType type.")
    if self.artifact_type not in ArtifactType.__registry__.values():
        raise ValueError(
            f"{self.artifact_type.__name__} is not a registered ArtifactType."
        )

    object.__setattr__(self, "relations", tuple(self.relations))
    for relation in self.relations:
        if not isinstance(relation, ArtifactSpecRelation):
            raise TypeError("ArtifactSpec.relations must contain relation tags.")
        relation.require_target_spec(self)
```

Add a full spec reference helper for relation targets/sources:

```python
@dataclass(frozen=True)
class ArtifactSpecRef:
    """Scope-free reference to a declared artifact spec."""

    plan_type: type[ArtifactPlan]
    artifact_type: type[ArtifactType]
    name: str
```

`ArtifactSpec.ref()` returns `ArtifactSpecRef(self.plan_type,
self.artifact_type, self.name)`.

Include `plan_type`, `artifact_type`, and `relations` in `__hash__`. Identity is
now the whole declared artifact term: role, artifact type, name,
materialization, required flag, sidecar role, and relation tags.

Do not give `plan_type` or `artifact_type` a default. The AST migration must
convert every declaration site to `ArtifactSpec.input(...)`,
`ArtifactSpec.output(...)`, or an explicit `plan_type=...,
artifact_type=...` product.

## 2. Generic Relation Tags

Add a generic relation tag family in `openhcs/core/artifacts.py`.

The base relation knows only declared artifact references and optional target
constraints. It does not know input/output. Concrete relation subclasses declare
only the constraints that are actually part of that relation.

This is a core OpenHCS callable abstraction. Every declaration owner that stores
`ArtifactSpecCollection` gets the same relation validation, regardless of
whether the specs came from decorators, generated CellProfiler modules,
hand-written `FunctionStep`s, sidecars, or future callable families.

```python
@dataclass(frozen=True)
class ArtifactSpecRelation(ABC, metaclass=AutoRegisterMeta):
    """Nominal relation tag attached to one target ArtifactSpec."""

    __registry_key__ = "relation_key"
    __skip_if_no_key__ = True

    relation_key: ClassVar[str | None] = None
    target_plan_type: ClassVar[type["ArtifactPlan"] | None] = None
    target_artifact_type: ClassVar[type["ArtifactType"] | None] = None

    source: ArtifactSpecRef

    def __post_init__(self) -> None:
        if not isinstance(self.source, ArtifactSpecRef):
            raise TypeError("Artifact relation source must be an ArtifactSpecRef.")
        self.require_registered_plan_type(self.source.plan_type, "source.plan_type")
        self.require_registered_artifact_type(
            self.source.artifact_type,
            "source.artifact_type",
        )
        if self.target_plan_type is not None:
            self.require_registered_plan_type(self.target_plan_type, "target_plan_type")
        if self.target_artifact_type is not None:
            self.require_registered_artifact_type(
                self.target_artifact_type,
                "target_artifact_type",
            )

    @classmethod
    def require_registered_plan_type(
        cls,
        plan_type: type["ArtifactPlan"],
        field_name: str,
    ) -> None:
        if not isinstance(plan_type, type) or not issubclass(plan_type, ArtifactPlan):
            raise TypeError(f"{cls.__name__}.{field_name} must be an ArtifactPlan type.")
        if plan_type not in ArtifactPlan.__registry__.values():
            raise ValueError(
                f"{cls.__name__}.{field_name} is not registered: "
                f"{plan_type.__name__}."
            )

    @classmethod
    def require_registered_artifact_type(
        cls,
        artifact_type: type["ArtifactType"],
        field_name: str,
    ) -> None:
        if not isinstance(artifact_type, type) or not issubclass(
            artifact_type,
            ArtifactType,
        ):
            raise TypeError(f"{cls.__name__}.{field_name} must be an ArtifactType.")
        if artifact_type not in ArtifactType.__registry__.values():
            raise ValueError(
                f"{cls.__name__}.{field_name} is not registered: "
                f"{artifact_type.__name__}."
            )

    def require_target_spec(self, spec: ArtifactSpec) -> None:
        if self.target_plan_type is not None and spec.plan_type is not self.target_plan_type:
            raise ValueError(
                f"{type(self).__name__} requires target plan "
                f"{self.target_plan_type.__name__}, got {spec.plan_type.__name__}."
            )
        if (
            self.target_artifact_type is not None
            and spec.artifact_type is not self.target_artifact_type
        ):
            raise ValueError(
                f"{type(self).__name__} requires target artifact type "
                f"{self.target_artifact_type.__name__}, "
                f"got {spec.artifact_type.__name__}."
            )

    def payload(self) -> dict[str, object]:
        return {
            "relation_key": type(self).relation_key,
            "source": self.source.payload(),
        }
```

Add a nominal behavior marker for relations that affect group scope:

```python
class ArtifactGroupScopeSourceRelation(ArtifactSpecRelation, ABC):
    """Target artifact inherits group scope from the named source artifact."""
```

Define the concrete group-lineage tag after `ArtifactOutputPlan` is defined:

```python
@dataclass(frozen=True)
class GroupLineageSourceRelation(ArtifactGroupScopeSourceRelation):
    """Target output inherits grouping from a declared source artifact ref."""

    relation_key: ClassVar[str] = "group_lineage_source"
    target_plan_type: ClassVar[type[ArtifactPlan]] = ArtifactOutputPlan
```

This relation only constrains the target role because that is its real semantic
fact: group lineage is declared on produced artifacts. The source is a full
`ArtifactSpecRef` supplied by the declaration site, so it can point to any
registered plan role and artifact type without changing relation code. Owners,
validators, sidecars, graph extraction, and the planner iterate registered
relation/plan/type declarations.

Adding another relation behavior means adding another subclass. Callers keep
iterating `ArtifactSpecRelation.__registry__.values()` or checking nominal
behavior markers such as `ArtifactGroupScopeSourceRelation`.

## 3. ArtifactSpecCollection Is The Query Surface

Extend `ArtifactSpecCollection`; do not create relation-specific collections.

```python
def for_plan_type(
    self,
    plan_type: type[ArtifactPlan],
) -> "ArtifactSpecCollection":
    if plan_type not in ArtifactPlan.__registry__.values():
        raise ValueError(f"{plan_type.__name__} is not a registered ArtifactPlan type.")
    return ArtifactSpecCollection(
        spec for spec in self.specs if spec.plan_type is plan_type
    )


def names_for_plan_type(self, plan_type: type[ArtifactPlan]) -> tuple[str, ...]:
    return self.for_plan_type(plan_type).names()


def for_artifact_type(
    self,
    artifact_type: type[ArtifactType],
) -> "ArtifactSpecCollection":
    if artifact_type not in ArtifactType.__registry__.values():
        raise ValueError(f"{artifact_type.__name__} is not a registered ArtifactType.")
    return ArtifactSpecCollection(
        spec for spec in self.specs if spec.artifact_type is artifact_type
    )


def ref_set(self) -> frozenset[ArtifactSpecRef]:
    return frozenset(spec.ref() for spec in self.specs)


def by_ref(self, ref: ArtifactSpecRef) -> ArtifactSpec | None:
    matches = tuple(spec for spec in self.specs if spec.ref() == ref)
    if len(matches) > 1:
        raise ValueError(f"Duplicate artifact spec ref {ref!r}.")
    if not matches:
        return None
    return matches[0]


def relation_refs(
    self,
    relation_type: type[ArtifactSpecRelation],
) -> tuple[tuple[ArtifactSpec, ArtifactSpecRelation], ...]:
    return tuple(
        (spec, relation)
        for spec in self.specs
        for relation in spec.relations
        if isinstance(relation, relation_type)
    )


def validate_registered_relation_refs(self, *, owner_name: str) -> None:
    refs = self.ref_set()
    for relation_type in ArtifactSpecRelation.__registry__.values():
        unknown = tuple(
            relation.source
            for spec, relation in self.relation_refs(relation_type)
            if relation.source not in refs
        )
        if unknown:
            raise ValueError(
                f"{owner_name} declares {relation_type.__name__} references to "
                "unknown artifact specs: "
                f"{unknown!r}."
            )
```

`unique(...)` and accumulator conflict checks must use `ArtifactSpec.ref()`.
The identity key is `(spec.plan_type, spec.artifact_type, spec.name)`, not
`spec.name` alone, because the same artifact name can appear in different plan
roles or artifact categories.

## 4. Selector ABC

Edit `openhcs/core/artifact_key_selection.py`.

Replace the input/output selector surface with one role-parametric selector:

```python
"""Nominal artifact-plan key selection shared by compiler declarations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TypeVar

from openhcs.core.artifacts import ArtifactPlan, ArtifactSpecCollection

ArtifactPlanT = TypeVar("ArtifactPlanT", bound=ArtifactPlan)


class ArtifactPlanKeySelector(ABC):
    """Nominal interface for declarations that select compiled artifact plans."""

    @property
    @abstractmethod
    def artifact_specs(self) -> ArtifactSpecCollection:
        """All artifact specs declared by this owner."""

    def artifact_names_for(
        self,
        plan_type: type[ArtifactPlanT],
    ) -> tuple[str, ...]:
        return self.artifact_specs.names_for_plan_type(plan_type)

    def select_plan_keys(
        self,
        plan_type: type[ArtifactPlanT],
        plans: Mapping[str, ArtifactPlanT],
    ) -> tuple[str, ...]:
        declared = set(self.artifact_names_for(plan_type))
        return tuple(key for key in plans if key in declared)

    def validate_artifact_relation_refs(self, *, owner_name: str) -> None:
        self.artifact_specs.validate_registered_relation_refs(owner_name=owner_name)
```

Do not add:

- `input_names`;
- `output_names`;
- `select_input_plan_keys`;
- `select_output_plan_keys`;
- `artifact_inputs`;
- `artifact_outputs`;
- `runtime_artifact_inputs`;
- `outputs`;
- `artifact_specs_by_plan_type`.

Those are not the abstraction. They are either old storage details or old
selectors that split one declaration algebra into two hand-maintained halves.

Use AST or an equivalent mechanical rewrite:

- `declarations.input_names` becomes
  `declarations.artifact_names_for(ArtifactInputPlan)`.
- `declarations.output_names` becomes
  `declarations.artifact_names_for(ArtifactOutputPlan)`.
- `declarations.select_input_plan_keys(input_plans)` becomes
  `declarations.select_plan_keys(ArtifactInputPlan, input_plans)`.
- `declarations.select_output_plan_keys(output_plans)` becomes
  `declarations.select_plan_keys(ArtifactOutputPlan, output_plans)`.

The only reason callsites mention `ArtifactInputPlan` or `ArtifactOutputPlan`
after this migration is because that site is actually asking for that registered
role. Generic validation and relation application do not spell the pair.

## 5. Owner Storage Migration

Owner classes store one `ArtifactSpecCollection` and validate registered
relations once in `__post_init__`.

Examples of the target shape:

```python
@dataclass(frozen=True)
class InvocationArtifactDeclarations(ArtifactPlanKeySelector):
    artifact_specs: ArtifactSpecCollection

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "artifact_specs",
            ArtifactSpecCollection(self.artifact_specs).unique(
                conflict_context="InvocationArtifactDeclarations artifact spec"
            ),
        )
        self.validate_artifact_relation_refs(
            owner_name="InvocationArtifactDeclarations"
        )
```

```python
@dataclass(frozen=True)
class ModuleArtifactContract(ArtifactPlanKeySelector):
    module_name: str
    artifact_specs: ArtifactSpecCollection

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "artifact_specs",
            ArtifactSpecCollection(self.artifact_specs).unique(
                conflict_context=f"ModuleArtifactContract({self.module_name})"
            ),
        )
        self.validate_artifact_relation_refs(
            owner_name=f"ModuleArtifactContract({self.module_name})",
        )
```

`CallableContract`, `InvocationArtifactDeclarations`, `ModuleArtifactContract`,
and `ModuleArtifactContracts` should all use the same field shape. If a class
needs constructor-local migration from an existing parser value, build
`ArtifactSpec.input(...)` / `ArtifactSpec.output(...)` there and immediately
store the unified `artifact_specs` collection. Do not keep `inputs` and
`outputs` as parallel public storage.

`CallableContract` can keep convenience properties only if they are genuinely
external API:

```python
@property
def artifact_input_names(self) -> tuple[str, ...]:
    return self.artifact_names_for(ArtifactInputPlan)


@property
def artifact_output_names(self) -> tuple[str, ...]:
    return self.artifact_names_for(ArtifactOutputPlan)
```

Those properties are role-specific API names, not a validation or declaration
surface. Internal generic code uses `artifact_specs` and `artifact_names_for`.

## 6. Sidecar Serialization

Edit `openhcs/interop/cellprofiler/runtime/generated_pipeline.py`.

Set:

```python
_CONTRACT_SIDECAR_VERSION = 3
```

Serialize partitioned `ModuleArtifactContractItem` records and the declared
artifact term, including role and relation tags:

```python
def payload(self, module_num: int, contract: ModuleArtifactContract) -> dict[str, Any]:
    return {
        "module_num": module_num,
        "module_name": contract.module_name,
        "items": [self.item_payload(item) for item in contract.items],
        "required_variable_components": [
            component.value for component in contract.required_variable_components
        ],
    }


def item_payload(self, item: ModuleArtifactContractItem) -> dict[str, Any]:
    return {
        "partition": item.partition_type.partition_key,
        "spec": self.spec_codec.payload(item.spec),
    }


def payload(self, spec: ArtifactSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "plan_role": spec.plan_type.plan_role,
        "artifact_type": spec.artifact_type.value,
        "required": spec.required,
        "sidecar_role": (
            None if spec.sidecar_role is None else spec.sidecar_role.value
        ),
        "relations": [relation.payload() for relation in spec.relations],
        "materialization": self.materialization_codec.payload(spec.materialization),
    }
```

Decode relation tags through the relation registry:

```python
def relation_from_payload(self, payload: Mapping[str, object]) -> ArtifactSpecRelation:
    relation_key = str(payload["relation_key"])
    relation_type = ArtifactSpecRelation.__registry__[relation_key]
    return relation_type.from_payload(payload)
```

Decode plan roles and artifact types through the existing core registries:

```python
def plan_type_from_payload(self, payload: Mapping[str, object]) -> type[ArtifactPlan]:
    plan_role = str(payload["plan_role"])
    return ArtifactPlan.__registry__[plan_role]


def artifact_type_from_payload(self, payload: Mapping[str, object]) -> type[ArtifactType]:
    artifact_type_key = str(payload["artifact_type"])
    return ArtifactType.coerce(artifact_type_key)


def spec_ref_from_payload(self, payload: Mapping[str, object]) -> ArtifactSpecRef:
    return ArtifactSpecRef(
        plan_type=ArtifactPlan.__registry__[str(payload["plan_role"])],
        artifact_type=ArtifactType.coerce(str(payload["artifact_type"])),
        name=str(payload["name"]),
    )
```

Do not add contract-level lineage JSON. Relations belong to the `ArtifactSpec`
that declares the target artifact.

## 7. Artifact Graph And Planner

Edit `openhcs/core/pipeline/artifact_planning.py`.

Graph extraction consumes the one declaration collection:

```python
output_specs = declarations.artifact_specs.for_plan_type(ArtifactOutputPlan)
input_specs = declarations.artifact_specs.for_plan_type(ArtifactInputPlan)

for spec in output_specs:
    producer_specs.add(spec)
    producer_groups[spec.name].append(normalized_key)
    producer_invocations[spec.name].append(invocation.key)

for spec in input_specs:
    consumer_specs.add(spec)
    consumer_invocations[spec.name].append(invocation.key)
```

This site names input/output because artifact graph construction is explicitly
building producer and consumer graph partitions. It is not relation validation
and it is not an owner projection map.

`ArtifactProducer` keeps the full output spec:

```python
ArtifactProducer(
    name=name,
    spec=spec,
    groups=ArtifactGraph.unique_preserving_order(producer_groups[name]),
    invocation_keys=tuple(producer_invocations[name]),
)
```

Edit `openhcs/core/pipeline/path_planner.py`.

Remove:

- `output_groups_from_input_identity(...)`;
- `object_only_input_group_scope(...)`;
- any artifact-category import used only for grouping heuristics.

Replace output-group resolution with relation behavior dispatch:

```python
def output_groups_from_declared_relations(
    self,
    declarations: ArtifactGraph,
    plans_by_ref: Mapping[ArtifactSpecRef, ArtifactPlan],
    *,
    step_index: int,
    step_name: str | None,
) -> Mapping[str, Iterable[Hashable | None]]:
    output_groups: dict[str, Iterable[Hashable | None]] = {
        name: groups
        for name, groups in declarations.output_groups.items()
    }

    for producer in declarations.producers:
        for relation in producer.spec.relations:
            if not isinstance(relation, ArtifactGroupScopeSourceRelation):
                continue
            source_plan = plans_by_ref.get(relation.source)
            if source_plan is None:
                raise MissingArtifactInputError(
                    step_id=step_index,
                    artifact_key=relation.source.name,
                    step_name=step_name,
                )
            output_groups[producer.name] = PathPlannerGroupScope.from_plan(
                source_plan
            ).keys

    return output_groups
```

The planner knows the nominal behavior
`ArtifactGroupScopeSourceRelation`. It does not know
`GroupLineageSourceRelation`, does not inspect artifact category, and does not
rediscover CellProfiler semantics.

If `PathPlannerGroupScope.from_input_plan(...)` exists today, generalize it to
`from_plan(...)` over `ArtifactPlan`. The group source relation owns the role
type; the group-scope helper only needs a plan with `group_keys`.

## 8. CellProfiler Boundary

Keep `FilterObjects` row and plan objects semantic. They expose source and
target artifact names. They should not own OpenHCS enum transport semantics and
they should not mirror a closed output-role family.

Edit `openhcs/processing/backends/cellprofiler/object_filtering.py`.

Remove:

- `FilterObjectsModule.OutputRole`;
- `FilterObjectsModule.Output`;
- `ObjectPair.filtered_object_output`;
- `ObjectPair.relationship_output`;
- `Plan.outputs`;
- the artifact-category switch in `artifact_contract(...)`.

Do not replace those with `FilterObjects*Output` subclasses. That is the same
manual family split in another form.

First replace the current single `artifact_kind` mixin attribute with a small
nominal capability algebra in
`openhcs/processing/backends/cellprofiler/module_classes.py`.
The current `artifact_kind` attribute is not MI-safe: a module that consumes
objects, emits objects, emits measurements, and emits relationships cannot have
one correct `artifact_kind`. The existing code already works around this by
naming base classes directly. Remove that pressure by separating the two axes:

- artifact category: the existing core `ArtifactType`;
- artifact plan role: registered `ArtifactPlan` implementations.

The product is a reusable capability mixin. Modules compose capabilities with
MI, and the generic contract helpers verify that composition before they emit a
spec.

Do not introduce a `CellProfilerArtifactCategory` registry that just wraps
`ArtifactType`; that would mirror the core category SSOT. If CellProfiler ever
needs category semantics that are not already in `ArtifactType`, add a separate
CP semantic declaration for that new fact. Do not duplicate the artifact
category itself.

Plan-role binding is also nominal and registered. This avoids hardcoding
`require_artifact`/`declare_artifact` branches in module code:

```python
class CellProfilerArtifactPlanBinding(ABC, metaclass=AutoRegisterMeta):
    __registry_key__ = "plan_type"
    __skip_if_no_key__ = True

    plan_type: ClassVar[type[ArtifactPlan]]

    @classmethod
    @abstractmethod
    def bind(
        cls,
        builder: "_SymbolTableBuilder",
        module: "ModuleBlock",
        spec: ArtifactSpec,
    ) -> object:
        """Bind one spec through the contract builder for this plan role."""


class ArtifactInputPlanBinding(CellProfilerArtifactPlanBinding):
    plan_type = ArtifactInputPlan

    @classmethod
    def bind(cls, builder, module, spec) -> object:
        return builder.require_artifact(spec, module)


class ArtifactOutputPlanBinding(CellProfilerArtifactPlanBinding):
    plan_type = ArtifactOutputPlan

    @classmethod
    def bind(cls, builder, module, spec) -> object:
        return builder.declare_artifact(spec, module)
```

The reusable capability family is the category/plan product, but the product
must be load-bearing. The product class identity is used by module inheritance,
setting declarations, and contract construction. Do not add dead marker classes
or per-capability forwarding methods.

Do not use `AutoRegisterMeta` directly for this capability base. Concrete
modules inherit these capability classes, and `AutoRegisterMeta` resolves keys
with `getattr`, so an inherited `capability_key` can accidentally register a
real module class as a capability. Use `__init_subclass__` and register only
classes that declare `capability_key` in their own class body.

```python
class CellProfilerArtifactCapability(ABC):
    """Reusable artifact capability declared by category and plan role."""

    __registry__: ClassVar[dict[str, type["CellProfilerArtifactCapability"]]] = {}
    capability_key: ClassVar[str | None] = None
    artifact_type: ClassVar[type[ArtifactType] | None] = None
    artifact_plan_type: ClassVar[type[ArtifactPlan] | None] = None

    def __init_subclass__(
        cls,
        **kwargs,
    ) -> None:
        super().__init_subclass__(**kwargs)
        declared_key = cls.__dict__.get("capability_key")
        if declared_key is None:
            return
        if cls.artifact_type is None or cls.artifact_plan_type is None:
            raise TypeError(
                f"{cls.__name__} must inherit one artifact-type axis and "
                "one artifact plan-role axis."
            )
        if cls.artifact_type not in ArtifactType.__registry__.values():
            raise ValueError(
                f"{cls.__name__} declares an unregistered artifact type."
            )
        if cls.artifact_plan_type not in ArtifactPlan.__registry__.values():
            raise ValueError(
                f"{cls.__name__} declares an unregistered artifact plan role."
            )
        existing = CellProfilerArtifactCapability.__registry__.get(declared_key)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Duplicate artifact capability {declared_key!r}: "
                f"{existing.__name__} vs {cls.__name__}."
            )
        CellProfilerArtifactCapability.__registry__[declared_key] = cls

    @classmethod
    def spec(
        cls,
        name: str,
        *,
        relations: Iterable[ArtifactSpecRelation] = (),
        **kwargs,
    ) -> ArtifactSpec:
        if cls.artifact_type is None or cls.artifact_plan_type is None:
            raise TypeError(f"{cls.__name__} is not a concrete artifact capability.")
        return ArtifactSpec(
            name,
            plan_type=cls.artifact_plan_type,
            artifact_type=cls.artifact_type,
            relations=tuple(relations),
            **kwargs,
        )

    @classmethod
    def artifact(
        cls,
        builder: "_SymbolTableBuilder",
        module: "ModuleBlock",
        name: str,
        **kwargs,
    ) -> object:
        if cls.artifact_plan_type is None:
            raise TypeError(f"{cls.__name__} is not a concrete artifact capability.")
        binding_type = CellProfilerArtifactPlanBinding.__registry__[cls.artifact_plan_type]
        return binding_type.bind(builder, module, cls.spec(name, **kwargs))
```

Axis mixins carry the reusable facts. They deliberately have no registry key:
they are not concrete capabilities by themselves.

```python
class ArtifactInputCapability(CellProfilerArtifactCapability):
    artifact_plan_type = ArtifactInputPlan


class ArtifactOutputCapability(CellProfilerArtifactCapability):
    artifact_plan_type = ArtifactOutputPlan


class ImageArtifactCapability(CellProfilerArtifactCapability):
    artifact_type = ImageArtifactType


class ObjectLabelArtifactCapability(CellProfilerArtifactCapability):
    artifact_type = ObjectLabelArtifactType


class MeasurementArtifactCapability(CellProfilerArtifactCapability):
    artifact_type = MeasurementArtifactType


class RelationshipArtifactCapability(CellProfilerArtifactCapability):
    artifact_type = RelationshipArtifactType
```

Concrete capability mixins are one-line products of the reusable axes. Each
declaration is load-bearing:

- modules inherit it to declare they are allowed to consume/produce that term;
- setting declarations reference it;
- `ArtifactContractModule` checks it before producing an `ArtifactSpec`;
- the registry can audit every declared product.

```python
class ImageArtifactInputModule(ImageArtifactCapability, ArtifactInputCapability):
    capability_key = "image_input"


class ImageArtifactOutputModule(ImageArtifactCapability, ArtifactOutputCapability):
    capability_key = "image_output"


class ObjectArtifactInputModule(ObjectLabelArtifactCapability, ArtifactInputCapability):
    capability_key = "object_label_input"


class ObjectArtifactOutputModule(ObjectLabelArtifactCapability, ArtifactOutputCapability):
    capability_key = "object_label_output"


class MeasurementArtifactInputModule(
    MeasurementArtifactCapability,
    ArtifactInputCapability,
):
    capability_key = "measurement_input"


class MeasurementArtifactOutputModule(
    MeasurementArtifactCapability,
    ArtifactOutputCapability,
):
    capability_key = "measurement_output"


class RelationshipArtifactInputModule(
    RelationshipArtifactCapability,
    ArtifactInputCapability,
):
    capability_key = "relationship_input"


class RelationshipArtifactOutputModule(
    RelationshipArtifactCapability,
    ArtifactOutputCapability,
):
    capability_key = "relationship_output"
```

Add the generic load-bearing check on `ArtifactContractModule`:

```python
class ArtifactContractModule(ABC, metaclass=AutoRegisterMeta):
    ...

    @classmethod
    def require_artifact_capability(
        cls,
        capability_type: type[CellProfilerArtifactCapability],
    ) -> type[CellProfilerArtifactCapability]:
        if not issubclass(capability_type, CellProfilerArtifactCapability):
            raise TypeError("artifact capability must be a CellProfilerArtifactCapability.")
        if not issubclass(cls, capability_type):
            raise TypeError(
                f"{cls.__name__} cannot declare {capability_type.__name__}; "
                "the module does not inherit that artifact capability."
            )
        return capability_type

    @classmethod
    def artifact_spec(
        cls,
        capability_type: type[CellProfilerArtifactCapability],
        name: str,
        **kwargs,
    ) -> ArtifactSpec:
        return cls.require_artifact_capability(capability_type).spec(name, **kwargs)

    @classmethod
    def artifact_ref(
        cls,
        capability_type: type[CellProfilerArtifactCapability],
        name: str,
    ) -> ArtifactSpecRef:
        return cls.require_artifact_capability(capability_type).spec(name).ref()

    @classmethod
    def artifact(
        cls,
        builder: "_SymbolTableBuilder",
        module: "ModuleBlock",
        capability_type: type[CellProfilerArtifactCapability],
        name: str,
        **kwargs,
    ) -> object:
        return cls.require_artifact_capability(capability_type).artifact(
            builder,
            module,
            name,
            **kwargs,
        )
```

`ArtifactContractModule.declared_artifact_input_settings()` and
`declared_artifact_output_settings()` should return setting declarations keyed
by capability types instead of raw artifact types:

```python
ArtifactSettingDeclaration = tuple[
    str | SettingNameFamily,
    type[CellProfilerArtifactCapability],
]
```

`artifact_inputs_from_setting(...)` and
`declared_output_artifacts_from_settings(...)` then call
`cls.artifact(builder, module, capability_type, name)`. This removes the last
ordinary contract-building reason to write `ArtifactSpec(name, artifact_type=...)`
inside module implementations, and it makes each declaration reusable.

`ObjectLineageTransformContractModule` becomes a composition of capabilities,
not a class that owns a conflicting relationship artifact type:

```python
class ObjectLineageTransformContractModule(
    PlaneRuntimeArtifactModule,
    MeasurementArtifactOutputModule,
    ObjectArtifactInputModule,
    ObjectArtifactOutputModule,
    RelationshipArtifactOutputModule,
):
    ...
```

Then make the existing typed `FilterObjects` plan/row objects emit artifact
specs through the composed module type and the load-bearing capability
declarations:

```python
@dataclass(frozen=True, slots=True)
class ObjectPair:
    input_object_name: str
    output_object_name: str

    def group_relation_for(
        self,
        module_type: type["FilterObjectsModule"],
    ) -> GroupLineageSourceRelation:
        return GroupLineageSourceRelation(
            source=module_type.artifact_ref(
                ObjectArtifactInputModule,
                self.input_object_name,
            )
        )

    def output_artifact_specs(
        self,
        module_type: type["FilterObjectsModule"],
    ) -> tuple[ArtifactSpec, ...]:
        group_relation = self.group_relation_for(module_type)
        return (
            module_type.artifact_spec(
                ObjectArtifactOutputModule,
                self.output_object_name,
                relations=(group_relation,),
            ),
            module_type.artifact_spec(
                RelationshipArtifactOutputModule,
                parent_child_relationship_artifact_name(
                    self.input_object_name,
                    self.output_object_name,
                ),
                relations=(group_relation,),
            ),
        )
```

```python
@dataclass(frozen=True, slots=True)
class Plan(ObjectPair):
    ...

    def outline_source_pairs(self) -> tuple[tuple[str, str], ...]:
        ...

    def output_artifact_specs(
        self,
        module_type: type["FilterObjectsModule"],
        module: "ModuleBlock",
    ) -> tuple[ArtifactSpec, ...]:
        group_relation = self.group_relation_for(module_type)
        return (
            module_type.artifact_spec(
                MeasurementArtifactOutputModule,
                module_type.measurement_artifact_name(module),
                relations=(group_relation,),
            ),
            *(
                spec
                for pair in self.object_pairs
                for spec in pair.output_artifact_specs(module_type)
            ),
            *(
                module_type.artifact_spec(
                    ImageArtifactOutputModule,
                    outline_name,
                    relations=(
                        GroupLineageSourceRelation(
                            source=module_type.artifact_ref(
                                ObjectArtifactInputModule,
                                input_name,
                            )
                        ),
                    ),
                )
                for outline_name, input_name in self.outline_source_pairs()
            ),
        )
```

`FilterObjectsModule` must inherit the capabilities it uses:

```python
class FilterObjectsModule(
    PlaneRuntimeArtifactModule,
    FilterObjectsInputPolicy,
    ModuleSettingsSourceModule,
    ObjectArtifactInputModule,
    ObjectArtifactOutputModule,
    ImageArtifactOutputModule,
    MeasurementArtifactOutputModule,
    RelationshipArtifactOutputModule,
):
    ...
```

The exact MRO can be adjusted for the existing `CellProfilerModule` /
`ArtifactContractModule` base constraints, but the capability set is not
optional. The generic `artifact_spec(...)` helper enforces it.

This is the reuse rule: every future module uses the same capability classes in
its inheritance list and the same generic helper in its contract code. No module
gets a local output taxonomy.

```python
for capability_type in CellProfilerArtifactCapability.__registry__.values():
    ...
```

Registry iteration now audits real declarations: every registered value is a
concrete category/plan product that can appear in module MI or setting
declarations.

Module boundary:

```python
@classmethod
def output_specs(cls, module, plan) -> tuple[ArtifactSpec, ...]:
    return plan.output_artifact_specs(cls, module)
```

In `artifact_contract(...)`:

```python
outputs = [
    builder.declare_artifact(spec, module)
    for spec in cls.output_specs(module, plan)
]
return assembler.assemble_contract(module, builder, artifact_specs=inputs + outputs)
```

If the assembler still accepts separate `inputs=` and `outputs=` lists, migrate
it in the same pass. Do not preserve split declarations as compatibility
storage.

After this migration, `object_filtering.py` should not spell concrete
`ArtifactType` classes in the contract-building path except through inherited
capability declarations. Runtime filtering code can still query artifact types
when it is actually matching runtime payload categories.

## 9. Symbol Table

`CellProfilerSymbol` should preserve the artifact term instead of mirroring
role/kind/relation fields separately.

Target shape:

```python
@dataclass(frozen=True)
class CellProfilerSymbol:
    artifact_spec: ArtifactSpec
    ...
```

If non-artifact symbols share this class today, split the nominal family rather
than adding nullable mirrors. Artifact symbols can expose convenience
properties by delegating to `artifact_spec`, but conflict detection must compare
the full artifact spec term.

`DeclaredArtifactSymbolCollector.declare_artifact(...)` and
`CellProfilerSymbolTableBuilder.declare_artifact(...)` accept and store the
`ArtifactSpec`. Duplicate artifact declarations conflict on
`ArtifactSpec.ref()` and then compare the full `ArtifactSpec`.

## 10. Verification

Focused tests:

- `tests/unit/test_function_patterns.py`
  - `ArtifactSpec` rejects unregistered `plan_type`;
  - `ArtifactSpec` rejects unregistered `artifact_type`;
  - `ArtifactSpec` rejects a relation whose target role does not match the spec;
  - relation sources are full `ArtifactSpecRef` values, not bare names;
  - `ArtifactSpecCollection.validate_registered_relation_refs(...)` rejects
    unknown relation sources;
  - `ArtifactSpecCollection.unique(...)` keys conflicts by `ArtifactSpec.ref()`.

- `tests/unit/test_path_planner_materialization.py`
  - a spec tagged with `GroupLineageSourceRelation` inherits groups from its
    source plan;
  - an untagged output keeps execution/output declaration scope;
  - a relation naming a missing source plan raises `MissingArtifactInputError`;
  - no object-artifact-type heuristic remains.

- `tests/unit/test_cellprofiler_symbol_table.py`
  - `FilterObjectsModule` inherits the artifact capabilities used by
    `Plan.output_artifact_specs(...)`;
  - `CellProfilerArtifactCapability.__registry__` contains only classes that
    declare `capability_key` directly, not concrete modules that inherit them;
  - `ArtifactContractModule.artifact_spec(...)` rejects a capability that the
    module type does not inherit;
  - `Plan.output_artifact_specs(...)` emits through load-bearing capability
    declarations, not an output-role enum, output-family registry, or category
    switch;
  - filtered objects, relationships, outlines, and measurements each carry the
    expected `GroupLineageSourceRelation` with full `ArtifactSpecRef` sources;
  - generated semantic contracts preserve `ArtifactSpec.relations`.

- `tests/unit/test_cellprofiler_generated_pipeline_execution.py`
  - JSON runtime sidecar version is `3`;
  - JSON sidecar round-trips `ArtifactSpec.plan_type`;
  - JSON sidecar round-trips `ArtifactSpec.artifact_type`;
  - JSON sidecar round-trips relation tags through
    `ArtifactSpecRelation.__registry__`;
  - Python semantic sidecar round-trips the same fields.

Focused command:

```bash
/home/ts/.pyenv/versions/3.12.0/bin/python -m pytest \
  tests/unit/test_function_patterns.py \
  tests/unit/test_path_planner_materialization.py \
  tests/unit/test_cellprofiler_symbol_table.py \
  tests/unit/test_cellprofiler_generated_pipeline_execution.py -q
```

Cleanup checks:

```bash
rg -n "ArtifactKind|ArtifactOutputGroupLineage|output_group_lineage|OutputRole|select_input_plan_keys|select_output_plan_keys|\\.input_names|\\.output_names|artifact_specs_by_plan_type|group_lineage_input_name|validate_group_lineage_inputs|source_plan_type|by_name_and_kind|of_kind" \
  openhcs/core \
  openhcs/interop/cellprofiler \
  openhcs/processing/backends/cellprofiler/object_filtering.py
```

The command should return no hits for the removed enum, lineage, selector, and
bare-name relation surfaces.

`source_name` is deliberately not part of this cleanup grep. It is a valid
measurement/equivalence source-image concept in OpenHCS, not the removed
artifact relation source-name surface. Relation cleanup is covered by the
`group_lineage_input_name`, `source_plan_type`, and full-ref relation checks
above.

Parity command:

```bash
OUT=/tmp/openhcs_cp30_single_grid_lineage_$(date +%Y%m%d_%H%M%S)
CELLPROFILER_EXECUTABLE=/home/ts/code/projects/openhcs/.venv-cellprofiler39/bin/cellprofiler \
/home/ts/.pyenv/versions/3.12.0/bin/python scripts/benchmark_cellprofiler_vs_openhcs.py run \
  --manifest benchmark/manifests/official30_portable_axis1.json \
  --output-dir "$OUT" \
  --case ExampleImagingFlowCytometryObjectsInGrid \
  --native-reference-root benchmark/native_refs/official30_scoped_rows \
  --require-native-reference \
  --no-memory-metric \
  --speedup-target 4 \
  --force-openhcs-run \
  --discard-openhcs-outputs \
  --continue-on-error \
  --source-schema-max-image-set-count 1 \
  --log-level INFO \
  --no-figures
```

## Dry Run Notes

- There is one artifact declaration collection per owner, not one per role.
- The role split is nominal and local to each `ArtifactSpec.plan_type`.
- The payload/category split is nominal and local to each
  `ArtifactSpec.artifact_type`; `ArtifactKind` is not part of the target model.
- Relation sources are full `ArtifactSpecRef` values, not bare names.
- The relation abstraction is core and callable-wide. CellProfiler only
  instantiates core relation tags at its artifact boundary.
- The concrete group-lineage relation is one registered subclass. It constrains
  its target to produced artifacts and takes a full source ref from the
  declaration site.
- The planner dispatches on the nominal behavior marker
  `ArtifactGroupScopeSourceRelation`, so additional group-scope relations compose
  without planner edits.
- FilterObjects composes load-bearing artifact capabilities through MI and emits
  specs through the generic capability-checked helper. There is no role enum,
  local output taxonomy, output-family registry, or category switch.
