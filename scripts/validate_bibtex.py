#!/usr/bin/env python3
"""
Validate BibTeX file using pybtex.
Checks for syntax errors, duplicate keys, and missing required fields.
"""

import sys
from pathlib import Path

try:
    from pybtex import PybtexEngine
except ImportError:
    print("Error: pybtex not installed")
    print("Install with: pip install pybtex")
    sys.exit(1)


def validate_bib_file(bib_file):
    """Validate a BibTeX file using pybtex."""
    print(f"Validating: {bib_file}")
    print("=" * 60)

    try:
        # Use PybtexEngine to read and parse the file
        bib_data = PybtexEngine().format_from_file(bib_file, None)
    except Exception as e:
        print(f"ERROR: Failed to parse {bib_file}")
        print(f"  {e}")
        return False

    # Get all entries
    entries = bib_data.entries
    print(f"Total entries: {len(entries)}")
    print()

    # Check for duplicate keys
    keys = [entry.key for entry in entries]
    duplicates = [k for i, k in enumerate(keys) if k in keys[:i]]
    if duplicates:
        print("⚠️  DUPLICATE KEYS FOUND:")
        for dup in duplicates:
            print(f"  - {dup}")
        print()
    else:
        print("✓ No duplicate keys found")
        print()

    # Check required fields based on entry type
    required_fields = {
        "article": ["author", "title", "journal", "year"],
        "inproceedings": ["author", "title", "booktitle", "year"],
        "book": ["author", "title", "publisher", "year"],
        "phdthesis": ["author", "title", "school", "year"],
        "mastersthesis": ["author", "title", "school", "year"],
    }

    missing_fields = []
    warnings = []

    for entry in entries:
        entry_type = entry.type
        entry_key = entry.key

        # Check if entry has required fields for its type
        if entry_type in required_fields:
            for field in required_fields[entry_type]:
                if field not in entry.fields:
                    missing_fields.append((entry_key, entry_type, field))
        else:
            warnings.append(
                (entry_key, entry_type, "Unknown entry type, skipping validation")
            )

    if missing_fields:
        print("⚠️  MISSING REQUIRED FIELDS:")
        for entry_key, entry_type, field in missing_fields:
            print(f"  {entry_key} ({entry_type}): missing '{field}'")
        print()

    if warnings:
        print("⚠️  WARNINGS:")
        for entry_key, entry_type, message in warnings:
            print(f"  {entry_key} ({entry_type}): {message}")
        print()

    # Check for common issues
    issues = []
    for entry in entries:
        entry_key = entry.key
        entry_type = entry.type

        # Check for empty year
        if "year" in entry.fields and not str(entry.fields["year"]).strip():
            issues.append((entry_key, "Year is empty"))

        # Check for empty title
        if "title" in entry.fields and not str(entry.fields["title"]).strip():
            issues.append((entry_key, "Title is empty"))

        # Check for empty author
        if "author" in entry.fields and not str(entry.fields["author"]).strip():
            issues.append((entry_key, "Author is empty"))

    if issues:
        print("⚠️  OTHER ISSUES:")
        for entry_key, message in issues:
            print(f"  {entry_key}: {message}")
        print()
    else:
        print("✓ No other issues found")
        print()

    # Summary
    print("=" * 60)
    if duplicates or missing_fields or issues:
        print("❌ VALIDATION FAILED")
        return False
    else:
        print("✅ VALIDATION PASSED")
        return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_bibtex.py <bib_file.bib>")
        sys.exit(1)

    bib_file = Path(sys.argv[1])
    if not bib_file.exists():
        print(f"Error: File not found: {bib_file}")
        sys.exit(1)

    success = validate_bib_file(bib_file)
    sys.exit(0 if success else 1)
