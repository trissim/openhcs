LLM-assisted code generation
============================

The desktop code editor can ask a configured LLM endpoint to draft pipeline or
custom-function source. The prompt catalog is built from the current function
catalog and documented declaration authorities so generated code can reference
available callables.

Generated text is untrusted source. Review imports, callable names, processing
contracts, parameters, and output configuration before applying it. OpenHCS
validates and parses the result into the same live declarations used by manual
code/UI editing; generation does not bypass compilation.

Configure the endpoint and model in the desktop LLM service settings. A local
endpoint keeps prompts on the local machine, while a remote endpoint receives
the prompt context according to that provider's policy. Never include secrets,
credentials, or private image data in a prompt.

See :doc:`code_ui_editing` and :doc:`custom_functions` for the authoritative
application boundaries.

