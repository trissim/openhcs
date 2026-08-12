"""CellProfiler artifact-contract declaration authority."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import TYPE_CHECKING

from openhcs.constants.constants import AllComponents
from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactOutputPlan,
    ArtifactSidecarSourceRelation,
    ArtifactSpec,
    ArtifactSpecCollection,
    ArtifactSpecRef,
    ArtifactSpecRelation,
    ArtifactType,
    ImageArtifactType,
    MeasurementsArtifactType,
    SourceStackLineageSourceRelation,
)
from openhcs.core.callable_contract import CallableContract
from openhcs.core.function_patterns import DEFAULT_GROUP_KEY
from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext
from openhcs.interop.cellprofiler.settings_binder import SettingToKeywordBinding
from openhcs.interop.cellprofiler_setting_normalization import (
    normalize_cellprofiler_setting_name,
)
from openhcs.interop.cellprofiler.setting_names import (
    setting_name_matches,
    setting_names,
    setting_values,
)

if TYPE_CHECKING:
    from openhcs.core.function_patterns import (
        FunctionInvocationKey,
        NormalizedFunctionItem,
    )
    from openhcs.interop.cellprofiler.parser import ModuleBlock, ModuleSetting
    from openhcs.core.pipeline.artifact_planning import ArtifactProducer


class CellProfilerModuleArtifactContracts:
    @classmethod
    def finalize_module_blocks_for_invocation(
        cls,
        blocks: tuple["ModuleBlock", ...],
        *,
        invocation: "NormalizedFunctionItem",
        step_context: "ArtifactDeclarationStepContext",
    ) -> tuple["ModuleBlock", ...]:
        """Apply module-owned transformations after generic reconstruction."""

        del cls, invocation, step_context
        return blocks

    @classmethod
    def artifact_bindings_for(
        cls,
        module: "ModuleBlock | None",
        *,
        invocation_key: "FunctionInvocationKey | None" = None,
        plan_type: type[ArtifactInputPlan] | type[ArtifactOutputPlan] | None = None,
        artifact_type: type[ArtifactType] | None = None,
    ) -> tuple[SettingToKeywordBinding, ...]:
        """Return active artifact bindings from the canonical setting declarations."""

        bindings = cls.active_artifact_bindings(
            module,
            invocation_key=invocation_key,
        )
        return tuple(
            binding
            for binding in bindings
            if plan_type is None or binding.require_artifact_plan_type() is plan_type
            if artifact_type is None or binding.require_artifact_type() is artifact_type
        )

    @classmethod
    def declared_artifact_bindings(
        cls,
        *,
        plan_type: type[ArtifactInputPlan] | type[ArtifactOutputPlan] | None = None,
        artifact_type: type[ArtifactType] | None = None,
    ) -> tuple[SettingToKeywordBinding, ...]:
        """Return the class-declared artifact bindings before value selection."""

        return tuple(
            binding
            for binding in cls.declared_setting_bindings()
            if binding.declares_artifact
            if plan_type is None or binding.require_artifact_plan_type() is plan_type
            if artifact_type is None or binding.require_artifact_type() is artifact_type
        )

    @classmethod
    def active_artifact_bindings(
        cls,
        module: "ModuleBlock | None",
        *,
        invocation_key: "FunctionInvocationKey | None" = None,
    ) -> tuple[SettingToKeywordBinding, ...]:
        """Return artifact bindings active for one exact module invocation."""

        del module, invocation_key
        return cls.declared_artifact_bindings()

    @classmethod
    def artifact_names_for_binding(
        cls,
        module: "ModuleBlock",
        binding: SettingToKeywordBinding,
    ) -> tuple[str, ...]:
        """Return artifact identities selected by one exact binding."""

        from openhcs.interop.cellprofiler.setting_names import split_symbol_names

        return tuple(
            name
            for value in setting_values(module, binding.setting_name)
            for name in split_symbol_names(value)
        )

    @classmethod
    def split_invocation_blocks_for_binding(
        cls,
        modules: tuple["ModuleBlock", ...],
        binding: SettingToKeywordBinding,
    ) -> tuple["ModuleBlock", ...]:
        """Split repeated artifact selections into exact scalar invocations."""

        split_blocks: list[ModuleBlock] = []
        for module in modules:
            names = cls.artifact_names_for_binding(module, binding)
            if len(names) <= 1:
                split_blocks.append(module)
                continue
            retained_records = tuple(
                record
                for record in module.iter_settings()
                if not setting_name_matches(record.name, binding.setting_name)
            )
            split_blocks.extend(
                replace(
                    module,
                    setting_records=[
                        *retained_records,
                        *binding.records_from_kwargs(
                            {binding.require_parameter_name(): name}
                        ),
                    ],
                )
                for name in names
            )
        return tuple(split_blocks)

    @classmethod
    def main_flow_output_specs(
        cls,
        main_flow_candidates: tuple[ArtifactSpec, ...],
    ) -> tuple[ArtifactSpec, ...]:
        """Publish every eligible image output through canonical main flow."""

        del cls
        return tuple(
            ArtifactSpecCollection(main_flow_candidates).of_artifact_type(
                ImageArtifactType
            )
        )

    @classmethod
    def advance_artifact_context(
        cls,
        step_context: ArtifactDeclarationStepContext,
        *,
        contract: CallableContract,
        invocation_key: "FunctionInvocationKey",
    ) -> ArtifactDeclarationStepContext:
        """Advance declaration state using this module's main-flow semantics."""

        from openhcs.core.pipeline.artifact_planning import (
            ArtifactGraph,
            artifact_producers_for_outputs,
        )

        main_flow_artifacts = step_context.main_flow_artifacts
        published_main_flow_outputs = cls.main_flow_output_specs(
            tuple(
                spec
                for spec in contract.artifact_outputs
                if spec.participates_in_main_flow
            )
        )
        if published_main_flow_outputs:
            main_flow_artifacts = ArtifactSpecCollection(
                spec.for_plan_type(ArtifactInputPlan)
                for spec in published_main_flow_outputs
            )
        if invocation_key.group_key != DEFAULT_GROUP_KEY:
            producer_groups: tuple[str | None, ...] = (invocation_key.group_key,)
        elif (
            step_context.source_bindings.binding_declarations
            and step_context.group_by.value is not None
        ):
            scoped_outputs = tuple(
                spec for spec in contract.artifact_outputs if spec.group_scope_sources()
            )
            producer_groups = (
                step_context.source_bindings.component_group_keys_for_artifact_specs(
                    AllComponents.from_value(step_context.group_by.value),
                    scoped_outputs or contract.artifact_inputs,
                    step_context.available_artifacts,
                )
                or (None,)
            )
        else:
            producer_groups = (None,)
        return step_context.advance_artifact_graph(
            ArtifactGraph(
                producers=artifact_producers_for_outputs(
                    contract.artifact_outputs,
                    groups=producer_groups,
                    invocation_keys=(invocation_key,),
                )
            ),
            main_flow_artifacts=main_flow_artifacts,
        )

    @classmethod
    def _artifact_context_for_group(
        cls,
        step_context: "ArtifactDeclarationStepContext",
        *,
        group_key: str,
    ) -> "ArtifactDeclarationStepContext":
        """Return the exact artifact scope represented by one invocation group."""

        del cls
        source_bindings = step_context.source_bindings
        main_flow_artifacts = step_context.main_flow_artifacts
        if group_key != DEFAULT_GROUP_KEY and source_bindings.binding_declarations:
            grouped_component = AllComponents.from_value(step_context.group_by.value)
            scoped_bindings = tuple(
                binding
                for binding in source_bindings.binding_declarations
                if binding.is_compatible_with_component_group(
                    grouped_component,
                    group_key,
                )
            )
            if source_bindings.primary_plane_bindings and not any(
                binding in source_bindings.primary_plane_bindings
                for binding in scoped_bindings
            ):
                raise ValueError(
                    f"No source binding declares {grouped_component.value} "
                    f"group {group_key!r}."
                )
            source_bindings = replace(
                source_bindings,
                bindings=scoped_bindings,
            )
        if group_key != DEFAULT_GROUP_KEY:
            excluded_producer_refs = frozenset(
                producer.spec.ref().for_plan_type(ArtifactInputPlan)
                for producer in step_context.available_artifact_producers
                if None not in producer.groups and group_key not in producer.groups
            )
            main_flow_artifacts = ArtifactSpecCollection(
                spec
                for spec in main_flow_artifacts.specs
                if spec.ref() not in excluded_producer_refs
            )
        scoped_context = replace(
            step_context,
            source_bindings=source_bindings,
            main_flow_artifacts=main_flow_artifacts,
        )
        if not source_bindings.binding_declarations:
            return scoped_context
        return scoped_context.with_source_declarations(
            binding.input_spec() for binding in source_bindings.primary_plane_bindings
        )

    @classmethod
    def _artifact_input_record_groups(
        cls,
        *,
        module: "ModuleBlock",
        invocation_key: "FunctionInvocationKey",
        step_context: "ArtifactDeclarationStepContext",
    ) -> tuple[tuple["ModuleSetting", ...], ...]:
        """Reconstruct missing input identities from the scoped artifact context."""

        bindings = cls.artifact_input_bindings_for_reconstruction(
            module,
            invocation_key=invocation_key,
            step_context=step_context,
        )
        groups: tuple[tuple["ModuleSetting", ...], ...] = ((),)
        for domain_key in dict.fromkeys(
            binding.artifact_input_domain_key() for binding in bindings
        ):
            groups = cls._combine_artifact_input_record_groups(
                groups,
                cls._artifact_input_record_groups_for_bindings(
                    module=module,
                    invocation_key=invocation_key,
                    bindings=tuple(
                        binding
                        for binding in bindings
                        if binding.artifact_input_domain_key() == domain_key
                    ),
                    step_context=step_context,
                ),
            )
        return groups

    @classmethod
    def artifact_input_bindings_for_reconstruction(
        cls,
        module: "ModuleBlock",
        *,
        invocation_key: "FunctionInvocationKey",
        step_context: "ArtifactDeclarationStepContext",
    ) -> tuple[SettingToKeywordBinding, ...]:
        """Return active input bindings before generic ordered assignment."""

        del step_context
        return cls.artifact_bindings_for(
            module,
            invocation_key=invocation_key,
            plan_type=ArtifactInputPlan,
        )

    @classmethod
    def _artifact_input_record_groups_for_bindings(
        cls,
        *,
        module: "ModuleBlock",
        invocation_key: "FunctionInvocationKey",
        bindings: tuple[SettingToKeywordBinding, ...],
        step_context: "ArtifactDeclarationStepContext",
    ) -> tuple[tuple["ModuleSetting", ...], ...]:
        """Derive one artifact kind's missing setting rows from current flow."""

        from openhcs.interop.cellprofiler.parser import ModuleSetting

        existing_names = cls._normalized_record_setting_names(module.iter_settings())
        missing_bindings: list[SettingToKeywordBinding] = []
        for binding in bindings:
            if any(
                normalize_cellprofiler_setting_name(name) in existing_names
                for name in setting_names(binding.setting_name)
            ):
                continue
            missing_bindings.append(binding)
        if not missing_bindings:
            return ((),)

        def unassigned_specs(
            binding: SettingToKeywordBinding,
            specs: tuple[ArtifactSpec, ...],
        ) -> tuple[ArtifactSpec, ...]:
            remaining = list(specs)
            consumed_names = (
                name
                for existing_binding in bindings
                if existing_binding not in missing_bindings
                if existing_binding.runtime_parameter_name
                == binding.runtime_parameter_name
                for name in cls.artifact_names_for_binding(module, existing_binding)
            )
            for consumed_name in consumed_names:
                matching_position = next(
                    (
                        position
                        for position, spec in enumerate(remaining)
                        if spec.name == consumed_name
                    ),
                    None,
                )
                if matching_position is not None:
                    remaining.pop(matching_position)
            return tuple(remaining)

        candidate_specs = tuple(
            unassigned_specs(
                binding,
                cls._available_artifact_input_specs(
                    binding=binding,
                    invocation_key=invocation_key,
                    step_context=step_context,
                )
            )
            for binding in missing_bindings
        )
        if len(missing_bindings) == 1 and missing_bindings[0].repeated:
            return (
                tuple(
                    ModuleSetting(
                        setting_names(missing_bindings[0].setting_name)[0],
                        spec.name,
                    )
                    for spec in candidate_specs[0]
                ),
            )
        if len(missing_bindings) == 1:
            if not candidate_specs[0]:
                return ()
            return tuple(
                (
                    ModuleSetting(
                        setting_names(missing_bindings[0].setting_name)[0],
                        spec.name,
                    ),
                )
                for spec in candidate_specs[0]
            )
        candidate_refs = tuple(
            tuple(spec.ref().for_plan_type(ArtifactInputPlan) for spec in specs)
            for specs in candidate_specs
        )
        if all(refs == candidate_refs[0] for refs in candidate_refs[1:]):
            specs = candidate_specs[0]
            repeated_positions = tuple(
                position
                for position, binding in enumerate(missing_bindings)
                if binding.repeated
            )
            if len(repeated_positions) > 1 or (
                repeated_positions
                and repeated_positions[0] != len(missing_bindings) - 1
            ):
                return ()
            scalar_count = len(missing_bindings) - len(repeated_positions)
            if len(specs) < scalar_count:
                return ()
            records: list[ModuleSetting] = []
            candidate_position = 0
            for binding in missing_bindings:
                assigned_specs = (
                    specs[candidate_position:]
                    if binding.repeated
                    else specs[candidate_position : candidate_position + 1]
                )
                records.extend(
                    ModuleSetting(setting_names(binding.setting_name)[0], spec.name)
                    for spec in assigned_specs
                )
                candidate_position += len(assigned_specs)
            return (
                tuple(records),
            )
        if any(not specs for specs in candidate_specs):
            return ()
        candidates_by_lineage: list[dict[ArtifactSpecRef, ArtifactSpec]] = []
        for specs in candidate_specs:
            by_lineage: dict[ArtifactSpecRef, ArtifactSpec] = {}
            for spec in specs:
                lineage = spec.source_stack_scope_identity()
                if lineage in by_lineage:
                    return ()
                by_lineage[lineage] = spec
            candidates_by_lineage.append(by_lineage)
        lineage_order = tuple(candidates_by_lineage[0])
        if any(
            tuple(candidates) != lineage_order for candidates in candidates_by_lineage
        ):
            return ()
        return tuple(
            tuple(
                ModuleSetting(
                    setting_names(binding.setting_name)[0],
                    candidates[lineage].name,
                )
                for binding, candidates in zip(
                    missing_bindings,
                    candidates_by_lineage,
                    strict=True,
                )
            )
            for lineage in lineage_order
        )

    @staticmethod
    def _combine_artifact_input_record_groups(
        earlier: tuple[tuple["ModuleSetting", ...], ...],
        later: tuple[tuple["ModuleSetting", ...], ...],
    ) -> tuple[tuple["ModuleSetting", ...], ...]:
        """Compose independent input-row alternatives in declaration order."""

        return tuple((*left, *right) for left in earlier for right in later)

    @classmethod
    def _available_artifact_input_specs(
        cls,
        *,
        binding: SettingToKeywordBinding,
        invocation_key: "FunctionInvocationKey",
        step_context: "ArtifactDeclarationStepContext",
    ) -> tuple[ArtifactSpec, ...]:
        """Return exact scoped input specs in their declared flow order."""

        artifact_type = binding.require_artifact_type()
        main_flow_specs = step_context.main_flow_artifacts.of_artifact_type(
            artifact_type
        )
        producer_specs = tuple(
            producer.spec
            for producer in step_context.available_artifact_producers
            if producer.spec.artifact_type is artifact_type
            if invocation_key.group_key == DEFAULT_GROUP_KEY
            or None in producer.groups
            or invocation_key.group_key in producer.groups
        )
        if binding.sidecar_role is None:
            candidates = (
                producer_specs
                if binding.runtime_parameter_name is not None
                else main_flow_specs
            )
            return tuple(spec for spec in candidates if spec.sidecar_role is None)

        sidecar_specs = ArtifactSpecCollection(
            producer.spec
            for producer in step_context.available_artifact_producers
            if producer.spec.artifact_type is artifact_type
            if producer.spec.sidecar_role is binding.sidecar_role
        ).unique(conflict_context="scoped artifact sidecar input")
        return tuple(
            ArtifactSpec.input(
                relation.source.name,
                relation.source.artifact_type,
            )
            for spec in sidecar_specs
            for relation in spec.relations
            if isinstance(relation, ArtifactSidecarSourceRelation)
        )

    @classmethod
    def _available_artifact_producers_for_input(
        cls,
        spec: ArtifactSpec,
        *,
        step_context: "ArtifactDeclarationStepContext",
    ) -> tuple["ArtifactProducer", ...]:
        """Return producers owning this exact artifact identity."""

        del cls
        input_ref = spec.ref().for_plan_type(ArtifactInputPlan)
        return tuple(
            producer
            for producer in step_context.available_artifact_producers
            if producer.spec.ref().for_plan_type(ArtifactInputPlan) == input_ref
        )

    @classmethod
    def _derived_identity_setting_records(
        cls,
        *,
        invocation: "NormalizedFunctionItem",
        block_position: int,
        existing_records: tuple["ModuleSetting", ...],
        step_context: "ArtifactDeclarationStepContext",
    ) -> tuple["ModuleSetting", ...]:
        """Derive missing output identities from canonical output bindings."""

        module = cls._module_block_from_setting_records(existing_records)
        bindings = cls.artifact_bindings_for(
            module,
            invocation_key=invocation.key,
            plan_type=ArtifactOutputPlan,
        )
        derived: tuple["ModuleSetting", ...] = ()
        for artifact_type in dict.fromkeys(
            binding.require_artifact_type() for binding in bindings
        ):
            own = cls._derived_identity_setting_records_for_output_bindings(
                bindings=tuple(
                    binding
                    for binding in bindings
                    if binding.require_artifact_type() is artifact_type
                ),
                artifact_type=artifact_type,
                block_position=block_position,
                existing_records=(*existing_records, *derived),
                step_context=step_context,
            )
            derived = (*derived, *own)
        return derived

    @classmethod
    def _derived_identity_setting_records_for_output_bindings(
        cls,
        *,
        bindings: tuple[SettingToKeywordBinding, ...],
        artifact_type: type[ArtifactType],
        block_position: int,
        existing_records: tuple["ModuleSetting", ...],
        step_context: "ArtifactDeclarationStepContext",
    ) -> tuple["ModuleSetting", ...]:
        """Derive defaults for one MRO owner's declared output settings."""

        from openhcs.interop.cellprofiler.parser import ModuleSetting

        existing_names = cls._normalized_record_setting_names(existing_records)
        return tuple(
            ModuleSetting(
                setting_names(binding.setting_name)[0],
                cls.canonical_output_artifact_name(
                    artifact_type=artifact_type,
                    output_position=output_position,
                    block_position=block_position,
                    step_context=step_context,
                ),
            )
            for output_position, binding in enumerate(bindings)
            if cls.derives_missing_output_identity(binding)
            if not any(
                normalize_cellprofiler_setting_name(name) in existing_names
                for name in setting_names(binding.setting_name)
            )
        )

    @classmethod
    def derives_missing_output_identity(
        cls,
        binding: SettingToKeywordBinding,
    ) -> bool:
        """Return whether reconstruction should synthesize this output setting."""

        del cls, binding
        return True

    @classmethod
    def _module_block_from_setting_records(
        cls,
        records: tuple["ModuleSetting", ...],
    ) -> "ModuleBlock":
        """Build the transient module value shared by declaration algorithms."""

        from openhcs.interop.cellprofiler.parser import ModuleBlock

        return ModuleBlock(
            name=cls.require_module_name(),
            module_num=0,
            enabled=True,
            setting_records=list(records),
        )

    @classmethod
    def canonical_output_artifact_name(
        cls,
        *,
        artifact_type: type[ArtifactType],
        output_position: int,
        block_position: int,
        step_context: "ArtifactDeclarationStepContext",
    ) -> str:
        """Return a deterministic default identity for one module output."""

        step_index = step_context.step_index
        if not isinstance(step_index, int):
            raise TypeError(
                f"{cls.__name__} requires an integer step index to derive output "
                "artifact identity."
            )
        suffix = artifact_type.require_value()
        ordinal = output_position + block_position + 1
        return f"{cls.require_module_name()}_{step_index + 1}_{suffix}_{ordinal}"

    @classmethod
    def canonical_numbered_module_output_artifact_name(
        cls,
        module: "ModuleBlock",
        *,
        artifact_type: type[ArtifactType],
        output_position: int,
    ) -> str:
        """Return the output identity owned by one exact numbered module block."""

        if module.module_num < 1:
            raise ValueError(
                f"{cls.__name__} requires a numbered module block to derive an "
                "occurrence-owned output artifact identity."
            )
        suffix = artifact_type.require_value()
        return (
            f"{cls.require_module_name()}_{module.module_num}_{suffix}_"
            f"{output_position + 1}"
        )

    @staticmethod
    def _normalized_record_setting_names(
        records: Iterable["ModuleSetting"],
    ) -> frozenset[str]:
        return frozenset(
            normalize_cellprofiler_setting_name(record.name) for record in records
        )

    @classmethod
    def measurement_artifact_name(
        cls,
        module: "ModuleBlock",
        *,
        invocation_key: "FunctionInvocationKey",
        step_context: "ArtifactDeclarationStepContext",
    ) -> str:
        """Return a deterministic measurement identity for one public invocation."""

        del invocation_key
        if module.module_num < 1:
            raise ValueError(
                f"{cls.__name__} requires a numbered module block to derive its "
                "measurement artifact identity."
            )
        step_name = (
            cls.require_module_name()
            if step_context.step_name is None
            else step_context.step_name
        )
        return f"{step_name}_{module.module_num}_measurements"

    @classmethod
    def artifact_inputs_from_bindings(
        cls,
        module: "ModuleBlock",
        bindings: tuple[SettingToKeywordBinding, ...],
        *,
        invocation_key: "FunctionInvocationKey",
        step_context: "ArtifactDeclarationStepContext",
    ) -> tuple[ArtifactSpec, ...]:
        """Resolve exact input bindings through the scoped artifact context."""

        inputs: list[ArtifactSpec] = []
        for binding in bindings:
            inputs.extend(
                cls.artifact_inputs_for_binding(
                    module,
                    binding=binding,
                    invocation_key=invocation_key,
                    step_context=step_context,
                )
            )
        return tuple(inputs)

    @classmethod
    def artifact_inputs_for_binding(
        cls,
        module: "ModuleBlock",
        *,
        binding: SettingToKeywordBinding,
        invocation_key: "FunctionInvocationKey",
        step_context: "ArtifactDeclarationStepContext",
    ) -> tuple[ArtifactSpec, ...]:
        """Resolve one active binding through its nominal declaration owner."""

        declared_names = cls.artifact_names_for_binding(module, binding)
        if binding.repeated:
            names = declared_names
        elif not declared_names:
            names = ()
        else:
            name = declared_names[0]
            if any(declared_name != name for declared_name in declared_names[1:]):
                raise ValueError(
                    f"Module {module.name} declares one artifact role for "
                    f"{setting_names(binding.setting_name)[0]!r}, but its setting "
                    f"rows select conflicting artifacts {declared_names!r}."
                )
            names = (name,)
        if binding.sidecar_role is not None:
            resolved_sidecars = []
            for name in names:
                source = (
                    step_context.available_artifacts.require_by_name_and_artifact_type(
                        name,
                        binding.require_artifact_type(),
                    )
                )
                matching_sidecars = ArtifactSpecCollection(
                    spec
                    for spec, relation in step_context.available_artifacts.relation_refs(
                        ArtifactSidecarSourceRelation
                    )
                    if spec.sidecar_role is binding.sidecar_role
                    if relation.source == source.ref()
                ).unique(conflict_context="artifact sidecar input")
                if len(matching_sidecars) != 1:
                    raise ValueError(
                        f"Module {module.name} requires exactly one "
                        f"{binding.sidecar_role.value} sidecar for {source.ref()!r}, "
                        f"got {tuple(spec.ref() for spec in matching_sidecars)!r}."
                    )
                resolved_sidecars.append(
                    cls.bind_artifact_input(binding, matching_sidecars[0])
                )
            return tuple(resolved_sidecars)

        return tuple(
            cls.require_available_artifact_input(
                module,
                binding=binding,
                name=name,
                invocation_key=invocation_key,
                step_context=step_context,
            )
            for name in names
        )

    @classmethod
    def artifact_input_occurrences_for_binding(
        cls,
        modules: Iterable["ModuleBlock"],
        *,
        binding: SettingToKeywordBinding,
        invocation_key: "FunctionInvocationKey",
        step_context: "ArtifactDeclarationStepContext",
    ) -> tuple[tuple[ArtifactSpec, ...], ...]:
        """Resolve one binding without erasing reconstructed block occurrences."""

        return tuple(
            cls.artifact_inputs_for_binding(
                module,
                binding=binding,
                invocation_key=invocation_key,
                step_context=step_context,
            )
            for module in modules
            if binding
            in cls.active_artifact_bindings(
                module,
                invocation_key=invocation_key,
            )
        )

    @classmethod
    def artifact_input_ref_occurrences_equivalent(
        cls,
        *,
        binding: SettingToKeywordBinding,
        target: tuple[tuple[ArtifactSpecRef, ...], ...],
        candidate: tuple[tuple[ArtifactSpecRef, ...], ...],
    ) -> bool:
        """Compare exact binding occurrences using declaration-owned cardinality."""

        del cls
        if binding.preserves_artifact_input_occurrence_partitions():
            return target == candidate
        unmatched_candidates = list(candidate)
        for target_occurrence in target:
            try:
                matching_position = unmatched_candidates.index(target_occurrence)
            except ValueError:
                return False
            unmatched_candidates.pop(matching_position)
        return not unmatched_candidates

    @classmethod
    def bind_artifact_input(
        cls,
        binding: SettingToKeywordBinding,
        spec: ArtifactSpec,
    ) -> ArtifactSpec:
        """Bind one resolved artifact occurrence to its exact runtime target."""

        del cls
        return replace(
            spec.for_plan_type(ArtifactInputPlan),
            parameter_name=binding.runtime_parameter_name,
        )

    @classmethod
    def require_available_artifact_input(
        cls,
        module: "ModuleBlock",
        *,
        binding: SettingToKeywordBinding,
        name: str,
        invocation_key: "FunctionInvocationKey",
        step_context: "ArtifactDeclarationStepContext",
    ) -> ArtifactSpec:
        """Return one exact context-owned input artifact or fail."""

        artifact_type = binding.require_artifact_type()
        input_ref = ArtifactSpec.input(name, artifact_type).ref()
        source_binding = step_context.source_bindings.binding_for_artifact_ref(
            input_ref
        )
        main_flow_spec = step_context.main_flow_artifacts.by_name_and_artifact_type(
            name,
            artifact_type,
        )
        producer_specs = tuple(
            producer.spec.for_plan_type(ArtifactInputPlan)
            for producer in cls._available_artifact_producers_for_input(
                ArtifactSpec.input(name, artifact_type),
                step_context=step_context,
            )
        )
        candidates = ArtifactSpecCollection(
            (
                *((source_binding.input_spec(),) if source_binding is not None else ()),
                *((main_flow_spec,) if main_flow_spec is not None else ()),
                *producer_specs,
            )
        ).unique(conflict_context="available artifact input")
        if len(candidates) == 1:
            return cls.bind_artifact_input(binding, candidates[0])
        if len(candidates) > 1:
            raise ValueError(
                f"Module {module.name} has conflicting context declarations for "
                f"{artifact_type.require_value()} artifact {name!r}: {candidates!r}."
            )

        matching_available_spec = (
            step_context.available_artifacts.by_name_and_artifact_type(
                name,
                artifact_type,
            )
        )
        if matching_available_spec is not None:
            producers = cls._available_artifact_producers_for_input(
                ArtifactSpec.input(name, artifact_type),
                step_context=step_context,
            )
            raise ValueError(
                f"Module {module.name} references available "
                f"{artifact_type.require_value()} artifact {name!r}, but no exact "
                "source binding, scoped main-flow declaration, or ArtifactProducer "
                f"owns it; candidates={producers!r}."
            )

        conflicting_types = tuple(
            dict.fromkeys(
                spec.artifact_type.require_value()
                for spec in (
                    *step_context.available_artifacts.specs,
                    *step_context.main_flow_artifacts.specs,
                )
                if spec.name == name
            )
        )
        if conflicting_types:
            raise ValueError(
                f"Module {module.name} references {name!r} as "
                f"{artifact_type.require_value()}, but current artifacts "
                f"declare types {conflicting_types!r}."
            )
        raise ValueError(
            f"Module {module.name} references unknown "
            f"{artifact_type.require_value()} artifact {name!r}."
        )

    @classmethod
    def artifact_contract_inputs(
        cls,
        module: "ModuleBlock",
        *,
        invocation_key: "FunctionInvocationKey",
        step_context: "ArtifactDeclarationStepContext",
    ) -> tuple[ArtifactSpec, ...]:
        """Construct input specs from active canonical artifact bindings."""

        return cls.artifact_inputs_from_bindings(
            module,
            cls.artifact_bindings_for(
                module,
                invocation_key=invocation_key,
                plan_type=ArtifactInputPlan,
            ),
            invocation_key=invocation_key,
            step_context=step_context,
        )

    @classmethod
    def finalize_artifact_contract_inputs(
        cls,
        module: "ModuleBlock",
        *,
        invocation_key: "FunctionInvocationKey",
        step_context: "ArtifactDeclarationStepContext",
        artifact_inputs: ArtifactSpecCollection,
    ) -> tuple[ArtifactSpec, ...]:
        """Resolve declarations that depend on the complete input collection."""

        del (
            cls,
            module,
            invocation_key,
            step_context,
        )
        return artifact_inputs.specs

    @classmethod
    def finalize_artifact_contract_outputs(
        cls,
        module: "ModuleBlock",
        *,
        invocation_key: "FunctionInvocationKey",
        step_context: "ArtifactDeclarationStepContext",
        artifact_inputs: ArtifactSpecCollection,
        artifact_outputs: ArtifactSpecCollection,
    ) -> tuple[ArtifactSpec, ...]:
        """Resolve declarations that depend on the complete output collection."""

        del (
            cls,
            module,
            invocation_key,
            step_context,
            artifact_inputs,
        )
        return artifact_outputs.specs

    @classmethod
    def artifact_contract_outputs(
        cls,
        module: "ModuleBlock",
        *,
        invocation_key: "FunctionInvocationKey",
        step_context: "ArtifactDeclarationStepContext",
        artifact_inputs: ArtifactSpecCollection,
    ) -> tuple[ArtifactSpec, ...]:
        """Construct output specs from active canonical artifact bindings."""

        bindings = cls.artifact_bindings_for(
            module,
            invocation_key=invocation_key,
            plan_type=ArtifactOutputPlan,
        )
        output_positions: dict[type[ArtifactType], int] = {}
        outputs: list[ArtifactSpec] = []
        for binding in bindings:
            artifact_type = binding.require_artifact_type()
            output_position = output_positions.get(artifact_type, 0)
            for name in cls.artifact_names_for_binding(module, binding):
                outputs.append(
                    cls.artifact_output_spec(
                        module,
                        binding=binding,
                        name=name,
                        invocation_key=invocation_key,
                        step_context=step_context,
                        artifact_inputs=artifact_inputs,
                        output_position=output_position,
                    )
                )
                output_position += 1
            output_positions[artifact_type] = output_position
        return tuple(outputs)

    @classmethod
    def artifact_output_relations(
        cls,
        module: "ModuleBlock",
        *,
        binding: SettingToKeywordBinding,
        name: str,
        invocation_key: "FunctionInvocationKey",
        step_context: "ArtifactDeclarationStepContext",
        artifact_inputs: ArtifactSpecCollection,
        output_position: int,
    ) -> tuple[ArtifactSpecRelation, ...]:
        """Return ordinary source-stack lineage for one declared output."""

        del module, binding, name, step_context, output_position
        primary_images = cls.primary_image_inputs(
            cls.require_callable(invocation_key.function_name),
            artifact_inputs.specs,
        )
        lineage_inputs = ArtifactSpecCollection(
            primary_images
            or tuple(
                spec
                for spec in artifact_inputs
                if spec.artifact_type.carries_source_image_context
            )
        )
        source = cls.single_artifact_lineage_input(lineage_inputs)
        return (
            ()
            if source is None
            else (SourceStackLineageSourceRelation(source=source.ref()),)
        )

    @classmethod
    def artifact_output_spec(
        cls,
        module: "ModuleBlock",
        *,
        binding: SettingToKeywordBinding,
        name: str,
        invocation_key: "FunctionInvocationKey",
        step_context: "ArtifactDeclarationStepContext",
        artifact_inputs: ArtifactSpecCollection,
        output_position: int,
    ) -> ArtifactSpec:
        """Build one output spec from its exact setting binding."""

        return ArtifactSpec.output(
            name,
            binding.require_artifact_type(),
            relations=cls.artifact_output_relations(
                module,
                binding=binding,
                name=name,
                invocation_key=invocation_key,
                step_context=step_context,
                artifact_inputs=artifact_inputs,
                output_position=output_position,
            ),
        )

    @classmethod
    def measurement_output_relations(
        cls,
        module: "ModuleBlock",
        *,
        invocation_key: "FunctionInvocationKey",
        step_context: "ArtifactDeclarationStepContext",
        artifact_inputs: ArtifactSpecCollection,
    ) -> tuple[ArtifactSpecRelation, ...]:
        """Terminate cooperative measurement-output relation declaration."""

        del cls, module, invocation_key, step_context, artifact_inputs
        return ()

    @classmethod
    def single_artifact_lineage_input(
        cls,
        artifact_inputs: ArtifactSpecCollection,
    ) -> ArtifactSpec | None:
        """Return the sole declared lineage input when it is unambiguous."""

        del cls
        refs = artifact_inputs.ref_set()
        if len(refs) != 1:
            return None
        return artifact_inputs.by_ref(next(iter(refs)))

    @classmethod
    def default_artifact_output_relations(
        cls,
        artifact_inputs: ArtifactSpecCollection,
    ) -> tuple[ArtifactSpecRelation, ...]:
        """Derive ordinary output lineage from one declared module input."""

        lineage_source = cls.single_artifact_lineage_input(artifact_inputs)
        if lineage_source is None:
            return ()
        return (SourceStackLineageSourceRelation(source=lineage_source.ref()),)

    @classmethod
    def measurement_output_artifact(
        cls,
        module: "ModuleBlock",
        *,
        invocation_key: "FunctionInvocationKey",
        step_context: "ArtifactDeclarationStepContext",
        artifact_inputs: ArtifactSpecCollection,
    ) -> ArtifactSpec:
        """Declare the standard CellProfiler measurement output artifact."""
        return ArtifactSpec.output(
            cls.measurement_artifact_name(
                module,
                invocation_key=invocation_key,
                step_context=step_context,
            ),
            MeasurementsArtifactType,
            measurement_feature_owner=cls,
            relations=cls.measurement_output_relations(
                module,
                invocation_key=invocation_key,
                step_context=step_context,
                artifact_inputs=artifact_inputs,
            ),
        )

    @classmethod
    def callable_contract(
        cls,
        *,
        module: "ModuleBlock",
        invocation_key: "FunctionInvocationKey",
        step_context: "ArtifactDeclarationStepContext",
    ) -> CallableContract:
        """Resolve dynamic artifact names onto the canonical callable contract."""
        step_context = cls._artifact_context_for_group(
            step_context,
            group_key=invocation_key.group_key,
        )
        func = cls.require_callable(invocation_key.function_name)
        callable_contract = CallableContract.from_callable(func)
        inputs = cls.artifact_contract_inputs(
            module,
            invocation_key=invocation_key,
            step_context=step_context,
        )
        inputs = cls.finalize_artifact_contract_inputs(
            module,
            invocation_key=invocation_key,
            step_context=step_context,
            artifact_inputs=ArtifactSpecCollection(inputs),
        )
        remaining_inputs = list(inputs)
        primary_image_inputs = cls.primary_image_inputs(func, inputs)
        for primary_image_input in primary_image_inputs:
            remaining_inputs.remove(primary_image_input)
        artifact_inputs = ArtifactSpecCollection(
            (*primary_image_inputs, *remaining_inputs)
        )
        outputs = cls.artifact_contract_outputs(
            module,
            invocation_key=invocation_key,
            step_context=step_context,
            artifact_inputs=artifact_inputs,
        )
        outputs = cls.finalize_artifact_contract_outputs(
            module,
            invocation_key=invocation_key,
            step_context=step_context,
            artifact_inputs=artifact_inputs,
            artifact_outputs=ArtifactSpecCollection(outputs),
        )
        artifact_outputs = ArtifactSpecCollection(outputs)
        artifact_outputs.unique(conflict_context="module artifact output")
        resolved = replace(
            callable_contract,
            metadata=replace(
                callable_contract.metadata,
                artifact_inputs=artifact_inputs.specs,
                artifact_outputs=artifact_outputs.specs,
            ),
        )
        resolved.validate_artifact_relation_refs(owner_name=cls.__name__)
        return resolved
