LLM-assisted code generation
============================

The desktop code editor can ask a configured LLM endpoint to draft pipeline or
custom-function source. The prompt catalog is built from the current function
catalog and documented declaration authorities so generated code can reference
available callables.

Generate a draft
----------------

1. Open the code view for the pipeline or custom function you want to edit.
2. Select **LLM Assist**. Use its settings control to choose the endpoint and
   model; the connection indicator must succeed before generation.
3. Describe the intended workflow or callable and request a draft.
4. Review the generated imports, declarations, parameters, and outputs in the
   code editor.
5. Validate the document, then apply it through the editor's normal
   revision-checked action. Compile a changed pipeline before execution.

Generated text is untrusted source. Review imports, callable names, processing
contracts, parameters, and output configuration before applying it. OpenHCS
validates and parses the result into the same live declarations used by manual
code/UI editing; generation does not bypass compilation.

The settings control updates the registered LLM service connection while
preserving the function-catalog-backed prompt authority. A local endpoint keeps
prompts on the local machine, while a remote endpoint receives the prompt
context according to that provider's policy. Never include secrets,
credentials, or private image data in a prompt.

See :doc:`code_ui_editing` and :doc:`custom_functions` for the authoritative
application boundaries.
