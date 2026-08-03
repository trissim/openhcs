# OpenHCS brand assets

`openhcs-logo-source.svg` is the one editable geometry authority for the OpenHCS
array-processing mark. The generated family contains:

- `openhcs-mark.svg` and `openhcs-mark-mono.svg` for symbol-only use.
- `openhcs-lockup-horizontal.svg` for wide headers.
- `openhcs-lockup-stacked.svg` for compact full-logo placement.
- `openhcs-icon-square.svg` for application and plugin icons.
- `openhcs-favicon.svg` for browser chrome.

Do not edit the derivative SVGs directly. `openhcs-mark.svg`, the lockups, icon,
and favicon are mechanically generated from the source geometry. The PNG, ICO,
and ICNS files are then rendered from `openhcs-icon-square.svg`. Regenerate the
complete family with:

```bash
scripts/render_brand_assets.sh
```
