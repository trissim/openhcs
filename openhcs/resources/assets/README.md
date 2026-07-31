# OpenHCS brand assets

The official source family contains:

- `openhcs-mark.svg` and `openhcs-mark-mono.svg` for symbol-only use.
- `openhcs-lockup-horizontal.svg` for wide headers.
- `openhcs-lockup-stacked.svg` for compact full-logo placement.
- `openhcs-icon-square.svg` for application and plugin icons.
- `openhcs-favicon.svg` for browser chrome.

`openhcs-icon-square.png`, `openhcs.ico`, and `openhcs.icns` are mechanically
rendered platform encodings of `openhcs-icon-square.svg`. Regenerate them with:

```bash
scripts/render_brand_assets.sh
```
