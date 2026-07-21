"""Dependency-free callable contract metadata keys."""

from __future__ import annotations

from typing import ClassVar


class FunctionContractAttribute:
    """Callable namespace keys owned by function-level contract decorators."""

    artifact_inputs: ClassVar[str] = "__artifact_inputs__"
    artifact_outputs: ClassVar[str] = "__artifact_outputs__"
    processing_contract: ClassVar[str] = "__processing_contract__"
    execution_scope: ClassVar[str] = "__openhcs_execution_scope__"
    declared_processing_contract: ClassVar[str] = (
        "__openhcs_declared_processing_contract__"
    )
    raw_processing_function: ClassVar[str] = "__openhcs_raw_processing_function__"
    processing_prepare: ClassVar[str] = "__openhcs_prepare__"
    runtime_adapter: ClassVar[str] = "__runtime_adapter__"
    runtime_image_execution_mode: ClassVar[str] = (
        "__openhcs_runtime_image_execution_mode__"
    )
    callable_request_binding: ClassVar[str] = "__openhcs_callable_request_binding__"
    special_inputs: ClassVar[str] = "__special_inputs__"
    artifact_input_parameter_names: ClassVar[str] = (
        "__openhcs_artifact_input_parameter_names__"
    )
    runtime_bound_parameters: ClassVar[str] = "__openhcs_runtime_bound_parameters__"
    runtime_context_parameter: ClassVar[str] = "__openhcs_runtime_context_parameter__"
    required_variable_components: ClassVar[str] = (
        "__openhcs_required_variable_components__"
    )
    variable_component_stack_requirement: ClassVar[str] = (
        "__openhcs_variable_component_stack_requirement__"
    )
    allowed_group_by: ClassVar[str] = "__openhcs_allowed_group_by__"
    object_label_input_execution_mode: ClassVar[str] = (
        "__object_label_input_execution_mode__"
    )
    image_payload_consumption: ClassVar[str] = "__openhcs_image_payload_consumption__"
