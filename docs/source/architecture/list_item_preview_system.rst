List-item previews
==================

This page moved when its generic implementation was extracted.
:external+pyqt-reactive:doc:`pyqt-reactive's list-item preview documentation
<architecture/list_item_preview_system>` owns the reusable mechanism. OpenHCS
documents only the domain declarations, adapters, and cross-package invariants
that compose it.

OpenHCS integration
-------------------

OpenHCS registers one formatter for the ``WellFilterConfig`` declaration
family. Config classes own their preview labels and whether an enabled member
remains visible without an explicit well selection. ``WellFilterMode`` members
own the compact include or exclude operator. The formatter resolves those
declarations through pyqt-reactive's preview API; it does not scan unrelated
registries or reproduce subtype tables.

See :doc:`ui_services_architecture` for the current OpenHCS integration boundary
and :doc:`external_foundations` for the package ownership policy.
