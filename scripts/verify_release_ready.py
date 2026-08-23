#!/usr/bin/env python3
"""Compatibility entry point for the authoritative release-readiness proof."""

from scripts.release_readiness import main

if __name__ == "__main__":
    raise SystemExit(main())
