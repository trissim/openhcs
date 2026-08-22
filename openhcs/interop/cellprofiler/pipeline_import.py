"""Pure CellProfiler module translation into public OpenHCS declarations."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from inspect import Parameter, signature
from pathlib import Path

from objectstate import config_context

from objectstate.lazy_factory import (
    resolve_lazy_configurations_for_serialization,
)
from openhcs.constants import AllComponents, Backend, GroupBy, VariableComponents
from openhcs.constants.input_source import InputSource
from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactOutputPlan,
    ArtifactSpec,
    ArtifactSpecCollection,
    ArtifactSpecRef,
)
from openhcs.core.callable_contract import CallableContract
from openhcs.core.component_set import ComponentSet
from openhcs.core.config import (
    LazyProcessingConfig,
    LazyStepSourceBindingsConfig,
    PipelineConfig,
    ProcessingConfig,
)
from openhcs.core.function_patterns import (
    DEFAULT_GROUP_KEY,
    FunctionInvocationKey,
    FunctionPatternSyntax,
    normalize_function_pattern,
)
from openhcs.core.invocation_artifacts import (
    ArtifactDeclarationStepContext,
    InvocationContractPlan,
)
from openhcs.core.pipeline.artifact_planning import (
    ArtifactProducer,
    extract_artifact_declarations,
)
from openhcs.core.source_bindings import (
    SourceBindingsConfig,
    StepSourceBindingsConfig,
)
from openhcs.core.source_bindings import source_binding_group_keys_for_group_by
from openhcs.core.steps.function_step import FunctionStep
from openhcs.core.vfs_protocol import FileManagerLike
from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule
from openhcs.interop.cellprofiler.parser import CPPipeParser, ModuleBlock
from openhcs.interop.cellprofiler.settings_binder import (
    SettingToKeywordBinding,
    SettingsBinder,
)


@dataclass(frozen=True, slots=True)
class _SelectedInputBindingOccurrence:
    """One binding-owned explicit input selection on a parsed target unit."""

    binding: SettingToKeywordBinding
    refs: tuple[ArtifactSpecRef, ...]


@dataclass(frozen=True, slots=True)
class _ParsedTargetUnit:
    """One exact parsed module invocation before public pattern lowering."""

    module: ModuleBlock
    invocation_key: FunctionInvocationKey
    contract: CallableContract
    raw_callable: Callable[..., object]
    behavior_kwargs: dict[str, object]
    compile_kwargs: dict[str, object]
    identity_kwargs: dict[str, object]
    processing_config: ProcessingConfig
    context: ArtifactDeclarationStepContext
    step_source_bindings: StepSourceBindingsConfig
    output_producers: tuple[ArtifactProducer, ...]
    selected_input_bindings: tuple[_SelectedInputBindingOccurrence, ...]
    target_position: int


@dataclass(frozen=True, slots=True)
class _PublicKwargProjection:
    """Public kwargs plus immutable per-unit selection analysis."""

    kwargs: dict[str, object]
    units: tuple[_ParsedTargetUnit, ...]


@dataclass(frozen=True, slots=True)
class _LoweredModuleBatch:
    """One module batch lowered by the sole pattern-shape owner."""

    function_pattern: FunctionPatternSyntax
    processing_config: ProcessingConfig
    step_source_bindings: StepSourceBindingsConfig
    accepted_context: ArtifactDeclarationStepContext
    next_module_num: int
    units: tuple[_ParsedTargetUnit, ...]


def import_cellprofiler_pipeline(
    cppipe_path: str | Path,
    *,
    filemanager: FileManagerLike | None = None,
    backend: Backend = Backend.DISK,
    source_root: str | Path | None = None,
) -> tuple[list[FunctionStep], PipelineConfig]:
    """Translate one `.cppipe` into ordinary public OpenHCS declarations."""

    path = Path(cppipe_path)
    parser = CPPipeParser()
    modules = tuple(parser.parse(path, filemanager=filemanager, backend=backend))
    if not modules:
        raise ValueError(f"CellProfiler pipeline {path} contains no modules.")
    source_bindings = CellProfilerModule.source_bindings_for_modules(
        modules,
        SourceBindingsConfig(image_plane_sources=parser.image_plane_sources),
    )
    return _public_pipeline(
        modules,
        source_bindings,
        binder=SettingsBinder(
            source_root=path.parent if source_root is None else source_root,
        ),
    )


def _public_pipeline(
    modules: tuple[ModuleBlock, ...],
    source_bindings: SourceBindingsConfig,
    *,
    binder: SettingsBinder,
) -> tuple[list[FunctionStep], PipelineConfig]:
    executable_modules: list[tuple[type[CellProfilerModule], ModuleBlock]] = []
    for module in modules:
        if not module.enabled:
            continue
        module_type = CellProfilerModule.require_module(module.name)
        if not module_type.emits_function_step():
            continue
        executable_modules.extend(
            (module_type, block)
            for block in module_type.invocation_module_blocks(module)
        )

    pipeline_processing = _pipeline_processing_config(
        executable_modules,
        source_bindings,
    )
    source_fragments = () if source_bindings.is_empty else (source_bindings,)
    pipeline_config = PipelineConfig.from_config(
        *source_fragments,
        pipeline_processing,
    )
    with config_context(pipeline_config):
        step_source_bindings = resolve_lazy_configurations_for_serialization(
            LazyStepSourceBindingsConfig()
        )
    if not isinstance(step_source_bindings, StepSourceBindingsConfig):
        raise TypeError(
            "CellProfiler source declarations must resolve to "
            f"StepSourceBindingsConfig, got {type(step_source_bindings).__name__}."
        )

    forward_context = ArtifactDeclarationStepContext().with_source_binding_scope(
        source_bindings=step_source_bindings,
        group_by=GroupBy.NONE,
        input_source=InputSource.PIPELINE_START,
    )
    target_units = _parsed_target_units(
        executable_modules,
        binder=binder,
        step_context=forward_context,
        inherited_processing_config=pipeline_processing,
    )
    emissions: list[
        tuple[
            FunctionPatternSyntax,
            str,
            ProcessingConfig,
            StepSourceBindingsConfig,
        ]
    ] = []
    next_module_num = 1
    module_position = 0
    while module_position < len(executable_modules):
        module_type = executable_modules[module_position][0]
        run_end = module_position + 1
        while (
            run_end < len(executable_modules)
            and executable_modules[run_end][0] is module_type
        ):
            run_end += 1

        lowered = None
        consumed_end = run_end
        while consumed_end > module_position:
            lowered = _lower_module_batch(
                module_type,
                target_units[module_position:consumed_end],
                step_context=replace(
                    forward_context,
                    step_name=executable_modules[module_position][1].name,
                    step_index=len(emissions),
                ),
                first_module_num=next_module_num,
            )
            if lowered is not None:
                break
            consumed_end -= 1
        if lowered is None:
            raise ValueError(
                f"CellProfiler module {executable_modules[module_position][1].name} "
                "cannot be represented by public FunctionStep declarations."
            )

        forward_context = lowered.accepted_context
        next_module_num = lowered.next_module_num
        emissions.append(
            (
                lowered.function_pattern,
                executable_modules[module_position][1].name,
                lowered.processing_config,
                lowered.step_source_bindings,
            )
        )
        module_position = consumed_end

    steps = [
        FunctionStep(
            func=function_pattern,
            name=module_name,
            processing_config=LazyProcessingConfig.from_config(
                processing_config,
                inherited=pipeline_processing,
            ),
            source_bindings=LazyStepSourceBindingsConfig.from_config(
                public_step_source_bindings,
                inherited=step_source_bindings,
            ),
        )
        for (
            function_pattern,
            module_name,
            processing_config,
            public_step_source_bindings,
        ) in emissions
    ]
    return steps, pipeline_config


def _pipeline_processing_config(
    executable_modules: list[tuple[type[CellProfilerModule], ModuleBlock]],
    source_bindings: SourceBindingsConfig,
) -> ProcessingConfig:
    """Resolve pipeline-wide axes from exact source and callable declarations."""

    variable_components = tuple(
        VariableComponents(component.value)
        for component in source_bindings.source_stack_components
    )
    if source_bindings.grouping_metadata_fields:
        grouped_components = tuple(
            dict.fromkeys(
                component
                for module_type, _module in executable_modules
                for function_name in module_type.declared_function_names()
                for component in CallableContract.from_callable(
                    module_type.require_callable(function_name)
                ).required_variable_components
            )
        )
        if len(grouped_components) > 1:
            raise ValueError(
                "CellProfiler source grouping contains callables with incompatible "
                f"required variable components {grouped_components!r}."
            )
        variable_components = tuple(
            dict.fromkeys((*variable_components, *grouped_components))
        )
    if not variable_components:
        return ProcessingConfig()
    inherited = ProcessingConfig()
    group_by = inherited.group_by
    if (
        group_by is not None
        and group_by.value is not None
        and AllComponents.from_value(group_by.value)
        in ComponentSet.from_enum_values(variable_components)
    ):
        group_by = GroupBy.NONE
    return replace(
        inherited,
        variable_components=list(variable_components),
        group_by=group_by,
    )


def _public_step_source_bindings(
    source_bindings: StepSourceBindingsConfig,
    artifact_inputs: Iterable[ArtifactSpec],
    input_source: InputSource,
) -> StepSourceBindingsConfig:
    """Project only direct source inputs loaded beside the resolved main flow."""

    if input_source is not InputSource.PREVIOUS_STEP:
        return source_bindings
    direct_source_refs = tuple(
        spec.ref()
        for spec in artifact_inputs
        if source_bindings.binding_for_artifact_ref(spec.ref()) is not None
    )
    if not direct_source_refs:
        return source_bindings
    return replace(
        source_bindings.for_artifact_refs(direct_source_refs),
        enabled=True,
    )


def _parsed_target_units(
    executable_modules: list[tuple[type[CellProfilerModule], ModuleBlock]],
    *,
    binder: SettingsBinder,
    step_context: ArtifactDeclarationStepContext,
    inherited_processing_config: ProcessingConfig,
) -> tuple[_ParsedTargetUnit, ...]:
    """Build the parsed target graph once in exact module order."""

    numbered_invocations, _next_module_num = (
        CellProfilerModule.number_step_invocation_blocks(
            tuple((module,) for _module_type, module in executable_modules),
            first_module_num=1,
        )
    )
    source_bindings = step_context.source_bindings
    probe_context = step_context
    units: list[_ParsedTargetUnit] = []
    for target_position, (
        (module_type, _module),
        numbered_blocks,
    ) in enumerate(zip(executable_modules, numbered_invocations, strict=True)):
        module = numbered_blocks[0]
        primary_function_name = module_type.require_callable().__name__
        invocation_key = FunctionInvocationKey(
            function_name=primary_function_name,
            group_key=DEFAULT_GROUP_KEY,
            position=0,
        )
        declaration_context = replace(
            probe_context,
            step_name=module.name,
            step_index=target_position,
            source_bindings=source_bindings,
        )
        contract = module_type.callable_contract(
            module=module,
            invocation_key=invocation_key,
            step_context=declaration_context,
        )
        raw_callable = module_type.resolve_function(
            module,
            contract=contract,
            source_bindings=source_bindings,
        )
        selected_invocation_key = replace(
            invocation_key,
            function_name=raw_callable.__name__,
        )
        if selected_invocation_key != invocation_key:
            invocation_key = selected_invocation_key
            contract = module_type.callable_contract(
                module=module,
                invocation_key=invocation_key,
                step_context=declaration_context,
            )
            resolved_callable = module_type.resolve_function(
                module,
                contract=contract,
                source_bindings=source_bindings,
            )
            if resolved_callable is not raw_callable:
                raise ValueError(
                    f"CellProfiler module {module.name}({module.module_num}) callable "
                    "selection did not converge after rebuilding its exact artifact "
                    "contract."
                )
        if raw_callable is not module_type.require_callable(raw_callable.__name__):
            raise ValueError(
                f"CellProfiler module {module.name} resolved a noncanonical callable."
            )
        follows_produced_main_flow = any(
            declaration_context.main_flow_artifacts.by_ref(spec.ref()) is not None
            and any(
                producer.spec.ref().for_plan_type(ArtifactInputPlan) == spec.ref()
                for producer in declaration_context.available_artifact_producers
            )
            for spec in contract.artifact_inputs
        )
        input_source = (
            InputSource.PREVIOUS_STEP
            if follows_produced_main_flow
            else (
                InputSource.PIPELINE_START
                if any(
                    source_bindings.binding_for_artifact_ref(spec.ref()) is not None
                    for spec in contract.artifact_inputs
                )
                else InputSource.PREVIOUS_STEP
            )
        )
        public_source_bindings = _public_step_source_bindings(
            source_bindings,
            contract.artifact_inputs,
            input_source,
        )
        processing_context = declaration_context.with_source_binding_scope(
            source_bindings=public_source_bindings.for_input_source(input_source),
            group_by=declaration_context.group_by,
            input_source=input_source,
        )
        processing_config = module_type.processing_config(
            callable_contract=contract,
            inherited=replace(
                inherited_processing_config,
                input_source=input_source,
            ),
            step_context=processing_context,
        )
        public_source_bindings = _public_step_source_bindings(
            source_bindings,
            contract.artifact_inputs,
            processing_config.input_source,
        )
        resolved_context = processing_context.with_source_binding_scope(
            source_bindings=public_source_bindings.for_input_source(
                processing_config.input_source
            ),
            group_by=processing_config.group_by,
            input_source=processing_config.input_source,
        )
        resolved_contract = module_type.callable_contract(
            module=module,
            invocation_key=invocation_key,
            step_context=resolved_context,
        )
        resolved_callable = module_type.resolve_function(
            module,
            contract=resolved_contract,
            source_bindings=source_bindings,
        )
        if resolved_callable is not raw_callable:
            raw_callable = resolved_callable
            invocation_key = replace(
                invocation_key,
                function_name=raw_callable.__name__,
            )
            if raw_callable is not module_type.require_callable(raw_callable.__name__):
                raise ValueError(
                    f"CellProfiler module {module.name} resolved a noncanonical "
                    "callable after input-source resolution."
                )
            resolved_contract = module_type.callable_contract(
                module=module,
                invocation_key=invocation_key,
                step_context=resolved_context,
            )
            converged_callable = module_type.resolve_function(
                module,
                contract=resolved_contract,
                source_bindings=source_bindings,
            )
            if converged_callable is not raw_callable:
                raise ValueError(
                    f"CellProfiler module {module.name}({module.module_num}) callable "
                    "selection did not converge after resolving its exact "
                    "input-source context."
                )
            processing_config = module_type.processing_config(
                callable_contract=resolved_contract,
                inherited=replace(
                    inherited_processing_config,
                    input_source=input_source,
                ),
                step_context=resolved_context,
            )
            public_source_bindings = _public_step_source_bindings(
                source_bindings,
                resolved_contract.artifact_inputs,
                processing_config.input_source,
            )
            resolved_context = resolved_context.with_source_binding_scope(
                source_bindings=public_source_bindings.for_input_source(
                    processing_config.input_source
                ),
                group_by=processing_config.group_by,
                input_source=processing_config.input_source,
            )
            resolved_contract = module_type.callable_contract(
                module=module,
                invocation_key=invocation_key,
                step_context=resolved_context,
            )
            final_callable = module_type.resolve_function(
                module,
                contract=resolved_contract,
                source_bindings=source_bindings,
            )
            if final_callable is not raw_callable:
                raise ValueError(
                    f"CellProfiler module {module.name}({module.module_num}) callable "
                    "selection did not converge after resolving its exact "
                    "input-source context."
                )
        bound_settings = module_type.bind_settings(module, binder=binder)
        parameters = signature(raw_callable).parameters
        artifact_parameters = frozenset(
            binding.require_parameter_name()
            for binding in module_type.declared_setting_bindings()
            if binding.declares_artifact
        )
        behavior_kwargs = {
            name: value
            for name, value in bound_settings.kwargs.items()
            if name in parameters
            and name not in artifact_parameters
            and not (
                parameters[name].default is not Parameter.empty
                and value == parameters[name].default
            )
        }
        compile_kwargs = {
            name: value
            for name, value in bound_settings.kwargs.items()
            if name not in parameters and name not in artifact_parameters
        }
        identity_kwargs = {
            name: value
            for name, value in bound_settings.kwargs.items()
            if name in artifact_parameters
        }
        next_probe_context = module_type.advance_artifact_context(
            resolved_context,
            contract=resolved_contract,
            invocation_key=invocation_key,
        )
        output_count = len(resolved_contract.artifact_outputs)
        output_producers = (
            next_probe_context.available_artifact_producers[-output_count:]
            if output_count
            else ()
        )
        if tuple(producer.spec for producer in output_producers) != tuple(
            resolved_contract.artifact_outputs
        ):
            raise ValueError(
                f"CellProfiler module {module.name}({module.module_num}) did not "
                "publish its parsed output producers in contract order."
            )
        probe_context = next_probe_context
        units.append(
            _ParsedTargetUnit(
                module=module,
                invocation_key=invocation_key,
                contract=resolved_contract,
                raw_callable=raw_callable,
                behavior_kwargs=behavior_kwargs,
                compile_kwargs=compile_kwargs,
                identity_kwargs=identity_kwargs,
                processing_config=processing_config,
                context=resolved_context,
                step_source_bindings=public_source_bindings,
                output_producers=output_producers,
                selected_input_bindings=(),
                target_position=target_position,
            )
        )
    return tuple(units)


def _public_kwargs_for_target(
    module_type: type[CellProfilerModule],
    units: tuple[_ParsedTargetUnit, ...],
    *,
    candidate_group_keys: tuple[str, ...],
    step_context: ArtifactDeclarationStepContext,
) -> _PublicKwargProjection | None:
    """Analyze or project exact identities through their owning bindings."""

    def merged_identity_value(
        binding: SettingToKeywordBinding,
        selected_units: tuple[_ParsedTargetUnit, ...],
    ) -> object | None:
        parameter_name = binding.require_parameter_name()
        if not selected_units or any(
            parameter_name not in unit.identity_kwargs for unit in selected_units
        ):
            return None
        values = tuple(unit.identity_kwargs[parameter_name] for unit in selected_units)
        if not binding.repeated or (
            binding.require_artifact_plan_type() is ArtifactInputPlan
            and binding.preserves_artifact_input_occurrence_partitions()
        ):
            return (
                values[0] if all(value == values[0] for value in values[1:]) else None
            )
        identities: list[object] = []
        for value in values:
            for identity in value if isinstance(value, (tuple, list)) else (value,):
                if identity not in identities:
                    identities.append(identity)
        if not identities:
            return None
        return identities[0] if len(identities) == 1 else tuple(identities)

    base_kwargs = {
        **units[0].behavior_kwargs,
        **units[0].compile_kwargs,
    }
    if any(
        {**unit.behavior_kwargs, **unit.compile_kwargs} != base_kwargs for unit in units
    ):
        return None

    retained_identity: dict[str, object] = {}
    analyzed_units = units

    for binding in module_type.declared_artifact_bindings(plan_type=ArtifactInputPlan):
        parameter_name = binding.require_parameter_name()
        active_units = tuple(
            unit
            for unit in analyzed_units
            if binding
            in module_type.active_artifact_bindings(
                unit.module,
                invocation_key=unit.invocation_key,
            )
        )
        selected_units = tuple(
            unit
            for unit in active_units
            if any(
                selection.binding is binding
                for selection in unit.selected_input_bindings
            )
        )
        if not selected_units:
            continue
        retained_value = merged_identity_value(binding, active_units)
        if retained_value is None:
            return None
        retained_identity[parameter_name] = retained_value

    declared_input_bindings = module_type.declared_artifact_bindings(
        plan_type=ArtifactInputPlan
    )
    while True:
        candidate_occurrences_by_binding: dict[
            SettingToKeywordBinding,
            list[tuple[ArtifactSpec, ...]],
        ] = {binding: [] for binding in declared_input_bindings}
        candidate_kwargs = {**base_kwargs, **retained_identity}
        for group_key in candidate_group_keys or (DEFAULT_GROUP_KEY,):
            leaf: FunctionPatternSyntax = (
                units[0].raw_callable
                if not candidate_kwargs
                else (units[0].raw_callable, candidate_kwargs)
            )
            pattern: FunctionPatternSyntax = (
                leaf if group_key == DEFAULT_GROUP_KEY else {group_key: leaf}
            )
            candidate_invocation = next(
                normalize_function_pattern(pattern).iter_items()
            )
            candidate_blocks, _consumed_names = (
                module_type.module_blocks_for_invocation(
                    invocation=candidate_invocation,
                    step_context=step_context,
                )
            )
            for binding in declared_input_bindings:
                candidate_occurrences_by_binding[binding].extend(
                    module_type.artifact_input_occurrences_for_binding(
                        candidate_blocks,
                        binding=binding,
                        invocation_key=candidate_invocation.key,
                        step_context=step_context,
                    )
                )

        mismatches: list[
            tuple[
                SettingToKeywordBinding,
                tuple[tuple[_ParsedTargetUnit, tuple[ArtifactSpec, ...]], ...],
            ]
        ] = []
        for binding, candidate_occurrences in candidate_occurrences_by_binding.items():
            if binding.require_parameter_name() in retained_identity:
                continue
            target_occurrences_by_unit = tuple(
                (unit, occurrences[0])
                for unit in analyzed_units
                for occurrences in (
                    module_type.artifact_input_occurrences_for_binding(
                        (unit.module,),
                        binding=binding,
                        invocation_key=unit.invocation_key,
                        step_context=unit.context,
                    ),
                )
                if occurrences
            )
            target_ref_occurrences = tuple(
                tuple(
                    spec.ref().for_plan_type(ArtifactInputPlan) for spec in occurrence
                )
                for unit, occurrence in target_occurrences_by_unit
            )
            candidate_ref_occurrences = tuple(
                tuple(
                    spec.ref().for_plan_type(ArtifactInputPlan) for spec in occurrence
                )
                for occurrence in candidate_occurrences
            )
            if not module_type.artifact_input_ref_occurrences_equivalent(
                binding=binding,
                target=target_ref_occurrences,
                candidate=candidate_ref_occurrences,
            ):
                mismatches.append((binding, target_occurrences_by_unit))
        if not mismatches:
            break

        selected_by_position: dict[
            int,
            list[_SelectedInputBindingOccurrence],
        ] = {}
        newly_selected_bindings: set[SettingToKeywordBinding] = set()
        for binding, target_occurrences_by_unit in mismatches:
            parameter_name = binding.require_parameter_name()
            if not target_occurrences_by_unit:
                return None
            selected_units = tuple(unit for unit, _specs in target_occurrences_by_unit)
            retained_value = merged_identity_value(binding, selected_units)
            if retained_value is None:
                return None
            retained_identity[parameter_name] = retained_value
            newly_selected_bindings.add(binding)
            for unit, target_specs in target_occurrences_by_unit:
                selected_by_position.setdefault(unit.target_position, []).append(
                    _SelectedInputBindingOccurrence(
                        binding=binding,
                        refs=tuple(spec.ref() for spec in target_specs),
                    )
                )
        analyzed_units = tuple(
            replace(
                unit,
                selected_input_bindings=(
                    *(
                        selection
                        for selection in unit.selected_input_bindings
                        if selection.binding not in newly_selected_bindings
                    ),
                    *selected_by_position.get(unit.target_position, ()),
                ),
            )
            for unit in analyzed_units
        )

    for binding in module_type.declared_artifact_bindings(plan_type=ArtifactOutputPlan):
        parameter_name = binding.require_parameter_name()
        active_units = tuple(
            unit
            for unit in analyzed_units
            if binding
            in module_type.active_artifact_bindings(
                unit.module,
                invocation_key=unit.invocation_key,
            )
        )
        if all(parameter_name not in unit.identity_kwargs for unit in active_units):
            continue
        retained_value = merged_identity_value(binding, active_units)
        if retained_value is None:
            return None
        retained_identity[parameter_name] = retained_value

    return _PublicKwargProjection(
        kwargs={**base_kwargs, **retained_identity},
        units=analyzed_units,
    )


def _lower_module_batch(
    module_type: type[CellProfilerModule],
    units: tuple[_ParsedTargetUnit, ...],
    *,
    step_context: ArtifactDeclarationStepContext,
    first_module_num: int,
) -> _LoweredModuleBatch | None:
    """Lower one adjacent same-module run when exact contracts prove it safe."""

    if not units:
        raise ValueError("CellProfiler import requires at least one target unit.")
    if step_context.step_index is None:
        raise ValueError("CellProfiler import lowering requires a step index.")
    step_index = step_context.step_index
    source_bindings = step_context.source_bindings
    raw_callable = units[0].raw_callable
    input_source = (
        InputSource.PIPELINE_START
        if any(
            unit.processing_config.input_source is InputSource.PIPELINE_START
            for unit in units
        )
        else InputSource.PREVIOUS_STEP
    )
    processing_config = replace(units[0].processing_config, input_source=input_source)
    if any(
        unit.raw_callable is not raw_callable
        or replace(unit.processing_config, input_source=input_source)
        != processing_config
        for unit in units
    ):
        return None
    prior_output_producers: set[ArtifactProducer] = set()
    for unit in units:
        input_refs = frozenset(
            spec.ref().for_plan_type(ArtifactInputPlan)
            for spec in unit.contract.artifact_inputs
        )
        if any(
            producer in prior_output_producers
            and producer.spec.ref().for_plan_type(ArtifactInputPlan) in input_refs
            for producer in unit.context.available_artifact_producers
        ):
            return None
        prior_output_producers.update(unit.output_producers)
    public_step_source_bindings = _public_step_source_bindings(
        source_bindings,
        (spec for unit in units for spec in unit.contract.artifact_inputs),
        input_source,
    )
    unit_target_contracts = tuple(unit.contract for unit in units)
    target_contract: CallableContract | None
    try:
        target_contract = module_type.combine_callable_contracts(unit_target_contracts)
    except ValueError:
        if len(unit_target_contracts) == 1:
            raise
        target_contract = None
    source_group_keys = source_binding_group_keys_for_group_by(
        source_bindings,
        processing_config.group_by,
    )
    reconstruction_context = step_context.with_source_binding_scope(
        source_bindings=public_step_source_bindings.for_input_source(input_source),
        group_by=processing_config.group_by,
        input_source=input_source,
    )
    lineage_keys: tuple[tuple[str, ...], ...] = ((),) * len(units)
    if source_group_keys:
        group_component = AllComponents.from_value(processing_config.group_by.value)
        lineage_keys = tuple(
            source_bindings.component_group_keys_for_artifact_specs(
                group_component,
                unit.contract.group_scope_inputs,
                unit.context.available_artifacts,
            )
            for unit in units
        )
    covered_source_group_keys = tuple(
        group_key for unit_lineage in lineage_keys for group_key in unit_lineage
    )
    covers_full_source_domain = bool(source_group_keys) and Counter(
        covered_source_group_keys
    ) == Counter(source_group_keys)
    has_unscoped_single_invocation = len(units) == 1 and not lineage_keys[0]
    follows_previous_main_flow = (
        len(units) == 1 and processing_config.input_source is InputSource.PREVIOUS_STEP
    )
    public_pattern: FunctionPatternSyntax | None = None
    lowered_units = units
    target_contracts_by_default_position: list[CallableContract] = []
    target_contracts_by_group_key: dict[str, list[CallableContract]] = {}
    if target_contract is not None and (
        not source_group_keys
        or has_unscoped_single_invocation
        or covers_full_source_domain
        or follows_previous_main_flow
    ):
        plain_kwargs = _public_kwargs_for_target(
            module_type,
            units,
            candidate_group_keys=(),
            step_context=reconstruction_context,
        )
        if plain_kwargs is not None:
            lowered_units = plain_kwargs.units
            public_pattern = (
                raw_callable
                if not plain_kwargs.kwargs
                else (raw_callable, plain_kwargs.kwargs)
            )

    if public_pattern is None:

        def projected_leaf(
            unit: _ParsedTargetUnit,
            candidate_group_keys: tuple[str, ...],
        ) -> tuple[FunctionPatternSyntax, tuple[_ParsedTargetUnit, ...]] | None:
            projection = _public_kwargs_for_target(
                module_type,
                (unit,),
                candidate_group_keys=candidate_group_keys,
                step_context=reconstruction_context,
            )
            if projection is None:
                return None
            return (
                (
                    unit.raw_callable
                    if not projection.kwargs
                    else (unit.raw_callable, projection.kwargs)
                ),
                projection.units,
            )

        uses_default_chain = (
            not source_group_keys
            and all(not declared_lineage for declared_lineage in lineage_keys)
        ) or (
            bool(source_group_keys)
            and all(
                Counter(declared_lineage) == Counter(source_group_keys)
                for declared_lineage in lineage_keys
            )
        )
        preserves_input_main_flow = all(
            not unit.contract.canonical_return_output_specs for unit in units
        )

        analyzed_units: list[_ParsedTargetUnit] = []
        if uses_default_chain:
            if len(units) != 1 and not preserves_input_main_flow:
                return None
            default_chain: list[FunctionPatternSyntax] = []
            for unit in units:
                projected = projected_leaf(unit, ())
                if projected is None:
                    return None
                leaf, projected_units = projected
                default_chain.append(leaf)
                analyzed_units.extend(projected_units)
                target_contracts_by_default_position.append(unit.contract)
            public_pattern = (
                default_chain[0] if len(default_chain) == 1 else default_chain
            )
        else:
            grouped_items: dict[str, list[FunctionPatternSyntax]] = {}
            for unit, declared_lineage in zip(units, lineage_keys, strict=True):
                if not declared_lineage or any(
                    group_key not in source_group_keys for group_key in declared_lineage
                ):
                    return None
                projected = projected_leaf(unit, declared_lineage)
                if projected is None:
                    return None
                leaf, projected_units = projected
                analyzed_units.extend(projected_units)
                for group_key in declared_lineage:
                    if group_key in grouped_items and not preserves_input_main_flow:
                        return None
                    grouped_items.setdefault(group_key, []).append(leaf)
                    target_contracts_by_group_key.setdefault(group_key, []).append(
                        unit.contract
                    )
            public_pattern = {
                group_key: items[0] if len(items) == 1 else items
                for group_key, items in grouped_items.items()
            }
        lowered_units = tuple(analyzed_units)

    normalized_public_pattern = normalize_function_pattern(public_pattern)
    public_contract_plans: dict[
        tuple[int, FunctionInvocationKey],
        InvocationContractPlan,
    ] = {}

    def target_contract_for_invocation(
        invocation_key: FunctionInvocationKey,
    ) -> CallableContract | None:
        if invocation_key.group_key == DEFAULT_GROUP_KEY:
            if target_contracts_by_default_position:
                if invocation_key.position >= len(target_contracts_by_default_position):
                    return None
                return target_contracts_by_default_position[invocation_key.position]
            return target_contract
        grouped_contracts = target_contracts_by_group_key.get(
            invocation_key.group_key,
            (),
        )
        if invocation_key.position >= len(grouped_contracts):
            return None
        return grouped_contracts[invocation_key.position]

    public_next_module_num = first_module_num
    group_contexts: list[ArtifactDeclarationStepContext] = []
    public_invocation_blocks = []
    for group in normalized_public_pattern.groups:
        group_context = reconstruction_context
        for invocation in group.items:
            invocation_blocks, consumed_names = (
                module_type.module_blocks_for_invocation(
                    invocation=invocation,
                    step_context=group_context,
                )
            )
            public_invocation_blocks.append(invocation_blocks)
            numbered_public_invocations, public_next_module_num = (
                CellProfilerModule.number_step_invocation_blocks(
                    tuple(public_invocation_blocks),
                    first_module_num=first_module_num,
                )
            )
            _public_contract, _consumed_names = (
                module_type.invocation_callable_contract(
                    invocation=invocation,
                    numbered_module_blocks=numbered_public_invocations[-1],
                    consumed_kwarg_names=consumed_names,
                    step_context=group_context,
                )
            )
            invocation_target_contract = target_contract_for_invocation(invocation.key)
            if invocation_target_contract is None:
                return None
            public_contract_plans[(step_index, invocation.key)] = (
                InvocationContractPlan(
                    contract=invocation_target_contract,
                    consumed_kwarg_names=_consumed_names,
                )
            )
            group_context = module_type.advance_artifact_context(
                group_context,
                contract=invocation_target_contract,
                invocation_key=invocation.key,
            )
        group_contexts.append(group_context)

    from openhcs.interop.cellprofiler.compile_time_contracts import (
        CellProfilerInvocationContractProvider,
    )

    public_graph = extract_artifact_declarations(
        normalized_public_pattern,
        invocation_contract_provider=CellProfilerInvocationContractProvider(
            public_contract_plans
        ),
        step_context=reconstruction_context,
    )
    next_main_flow = ArtifactSpecCollection(
        ArtifactSpecCollection(
            spec
            for group_context in group_contexts
            for spec in group_context.main_flow_artifacts.specs
        ).unique(conflict_context="function-pattern group main flow")
    )
    accepted_context = replace(
        reconstruction_context,
        source_bindings=source_bindings,
    ).advance_artifact_graph(
        public_graph,
        main_flow_artifacts=next_main_flow,
    )
    return _LoweredModuleBatch(
        function_pattern=public_pattern,
        processing_config=processing_config,
        step_source_bindings=public_step_source_bindings,
        accepted_context=accepted_context,
        next_module_num=public_next_module_num,
        units=lowered_units,
    )
