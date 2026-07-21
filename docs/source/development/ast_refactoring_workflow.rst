AST-assisted documentation refactoring
======================================

Use structured parsing for mechanical documentation migrations, then review the
semantic boundary manually. Text replacement alone cannot determine ownership.

Workflow
--------

1. Search source and docs for the domain terms plus the nominal registry roots.
2. Identify the owning declaration and current public import before editing.
3. Parse Python code blocks with ``ast.parse`` and classify imports, calls, and
   keyword arguments.
4. Generate an ``apply_patch`` batch for only the mechanically equivalent
   changes.
5. Review every change that alters configuration nesting, ownership, or
   runtime semantics.
6. run ``scripts/validate_docs.py`` and Sphinx with warnings as errors.

The AST pass is useful for import moves, renamed call sites, and detecting old
constructor keywords. It must not invent compatibility aliases, mirror
registries, or translate an obsolete semantic model without checking the
compiler/declaration authority.

For Python source refactors, preserve comments and formatting with a concrete
syntax tree tool when round-trip fidelity matters. Always validate behavioral
tests after a mechanical source rewrite.
