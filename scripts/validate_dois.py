#!/usr/bin/env python3
"""
Validate DOIs in BibTeX file.
Checks that all DOIs resolve correctly to valid papers.
"""

import sys
import requests
from pathlib import Path


def validate_doi(doi):
    """Validate a DOI by checking if it resolves via CrossRef."""
    if not doi:
        return None, "No DOI provided"

    # Format as URL
    doi_url = f"https://api.crossref.org/works/{doi}"

    try:
        response = requests.get(
            doi_url, params={"mailto": "bibtex-validator@example.com"}, timeout=10
        )
        if response.status_code == 200:
            data = response.json()

            # Check if we got a valid response
            if "status" in data and data["status"] == "ok":
                # Get the paper title from CrossRef
                if (
                    "message" in data
                    and "items" in data["message"]
                    and len(data["message"]["items"]) > 0
                ):
                    item = data["message"]["items"][0]
                    title = item.get("title", ["Unknown"])[0]
                    return True, title
                else:
                    return False, "DOI not found"
            else:
                return False, f"Invalid DOI: {data.get('status', 'unknown')}"
        elif response.status_code == 404:
            return False, "DOI not found (404)"
        else:
            return False, f"HTTP {response.status_code}"
    except Exception as e:
        return False, f"Request failed: {str(e)}"


def extract_dois_from_bib(bib_file):
    """Extract all DOI fields from BibTeX file."""
    import re

    with open(bib_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Find all DOI fields
    doi_pattern = r"doi\s*=\s*[{\']?([^}\']+)[{\']?"
    dois = re.findall(doi_pattern, content)

    return dois


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_dois.py <bib_file.bib>")
        sys.exit(1)

    bib_file = Path(sys.argv[1])
    if not bib_file.exists():
        print(f"Error: File not found: {bib_file}")
        sys.exit(1)

    print(f"Validating DOIs in: {bib_file}")
    print("=" * 60)

    # Extract DOIs from BibTeX file
    dois = extract_dois_from_bib(bib_file)
    print(f"Found {len(dois)} DOI(s)")
    print()

    if not dois:
        print("No DOIs found in bibliography")
        sys.exit(0)

    # Validate each DOI
    valid_count = 0
    invalid_count = 0
    error_count = 0

    results = []

    for i, doi in enumerate(dois, 1):
        print(f"[{i}/{len(dois)}] Checking: {doi}")
        is_valid, info = validate_doi(doi)

        if is_valid:
            print(f"  ✅ OK: {info[:80]}..." if len(info) > 80 else info)
            valid_count += 1
            results.append((doi, "valid", info))
        else:
            print(f"  ❌ ERROR: {info}")
            invalid_count += 1
            results.append((doi, "invalid", info))
        print()

    # Summary
    print("=" * 60)
    print(f"Valid DOIs: {valid_count}")
    print(f"Invalid DOIs: {invalid_count}")
    print(f"Errors: {error_count}")
    print()

    # Report invalid DOIs
    if invalid_count > 0:
        print("⚠️  INVALID DOIs FOUND:")
        print()
        for doi, status, info in results:
            if status == "invalid":
                print(f"  {doi}")
                print(f"    Reason: {info}")
        print()

    if error_count > 0:
        print("⚠️  VALIDATION ERRORS:")
        print()
        for doi, status, info in results:
            if status == "error":
                print(f"  {doi}")
                print(f"    Reason: {info}")
        print()

    # Exit with appropriate code
    sys.exit(0 if invalid_count == 0 else 1)
