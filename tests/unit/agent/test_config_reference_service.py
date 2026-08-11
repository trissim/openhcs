from objectstate import ObjectState

from openhcs.agent.dto.knowledge import KnowledgeBaseDocumentRequest
from openhcs.agent.services.config_reference_service import (
    CONFIG_REFERENCE_DIRECTIVE,
    ConfigReferenceRstRenderer,
    expand_config_reference_directives,
)
from openhcs.agent.services.config_service import (
    agent_config_declaration_from_request,
)
from openhcs.agent.services.knowledge_base_service import KnowledgeBaseService
from openhcs.pyqt_gui.config import UIConfig


def test_every_visible_agent_config_field_has_declaration_help() -> None:
    for config_name in ("global", "pipeline", "step", "ui"):
        declaration = agent_config_declaration_from_request(config_name)

        assert tuple(
            field.path
            for field in declaration.reflected_fields()
            if not field.ui_hidden and not (field.description or "").strip()
        ) == ()


def test_ui_object_state_and_agent_schema_share_exact_help() -> None:
    state = ObjectState(UIConfig(), scope_id="UIConfig-help-authority")
    schema_descriptions = {
        field.path: field.description
        for field in agent_config_declaration_from_request("ui").reflected_fields()
    }

    assert state.parameter_descriptions
    assert schema_descriptions == state.parameter_descriptions


def test_reference_renderer_projects_every_visible_field_from_declarations() -> None:
    renderer = ConfigReferenceRstRenderer()

    for config_name in ("global", "pipeline", "step", "ui"):
        declaration = agent_config_declaration_from_request(config_name)
        rendered = "\n".join(renderer.render(config_name))

        assert f"Configuration type: ``{declaration.display_name()}``" in rendered
        assert f"Authoring path: ``{declaration.authoring_path}``" in rendered
        for field in declaration.reflected_fields():
            if field.ui_hidden:
                continue
            assert f"``{field.path}``" in rendered
            assert " ".join(field.description.split()) in rendered


def test_reference_directive_expands_for_non_sphinx_consumers() -> None:
    source_lines = (
        "Configuration reference",
        "=======================",
        "",
        f".. {CONFIG_REFERENCE_DIRECTIVE}:: step",
    )

    expanded = expand_config_reference_directives(source_lines)
    rendered = "\n".join(expanded)

    assert CONFIG_REFERENCE_DIRECTIVE not in rendered
    assert "Configuration type: ``FunctionStep``" in rendered
    assert "``processing_config.variable_components``" in rendered

    ui_rendered = "\n".join(ConfigReferenceRstRenderer().render("ui"))
    assert "``logging.level``" in ui_rendered
    assert "``agent_bridge.enabled``" in ui_rendered
    assert "``zmq.default_port``" in ui_rendered
    assert "Declared by: ``zmqruntime.config.ZMQConfig``" in ui_rendered


def test_knowledge_base_sections_include_declaration_derived_config_help() -> None:
    service = KnowledgeBaseService()
    catalog_document = service.get_document(
        KnowledgeBaseDocumentRequest.from_fields(
            document_id="openhcs_configuration_reference",
            max_chars=500,
        )
    )
    section_id = "globalpipelineconfig-source-bindings-config"

    assert any(
        section.section_id == section_id for section in catalog_document.sections
    )

    section = service.get_document(
        KnowledgeBaseDocumentRequest.from_fields(
            document_id="openhcs_configuration_reference",
            section_id=section_id,
            max_chars=50_000,
        )
    )

    assert section.errors == ()
    assert section.truncated is False
    assert section.selected_section_id == section_id
    assert "``source_bindings_config.metadata_rules[].pattern``" in section.content
    assert (
        "Regular expression containing one or more named metadata capture groups."
        in section.content
    )

    ui_section = service.get_document(
        KnowledgeBaseDocumentRequest.from_fields(
            document_id="openhcs_configuration_reference",
            section_id="uiconfig-zmq",
            max_chars=50_000,
        )
    )
    assert ui_section.errors == ()
    assert ui_section.truncated is False
    assert "``zmq.default_port``" in ui_section.content
    assert "First data port used when a caller" in ui_section.content
    assert "Declared by: ``zmqruntime.config.ZMQConfig``" in ui_section.content
