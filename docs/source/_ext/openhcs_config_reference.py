"""Sphinx directive for declaration-derived OpenHCS configuration reference."""

from __future__ import annotations

from docutils.parsers.rst import Directive

from openhcs.agent.services.config_reference_service import (
    CONFIG_REFERENCE_DIRECTIVE,
    ConfigReferenceRstRenderer,
)


class OpenHCSConfigReferenceDirective(Directive):
    """Insert owner-derived configuration reference into the RST input stream."""

    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = False
    has_content = False

    def run(self):
        config_name = self.arguments[0].strip()
        source = self.state.document.current_source or "<openhcs-config-reference>"
        self.state_machine.insert_input(
            list(ConfigReferenceRstRenderer().render(config_name)),
            source,
        )
        return []


def setup(app):
    app.add_directive(CONFIG_REFERENCE_DIRECTIVE, OpenHCSConfigReferenceDirective)
    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
