#!/usr/bin/env python3
"""
Validate BibTeX citations against CrossRef database.
Checks that:
1. All citations in .tex files exist in .bib file
2. DOIs resolve to the correct papers
3. Paper titles match between .bib and CrossRef
"""

import sys
import re
import requests
from pathlib import Path

try:
    from pybtex import PybtexEngine
except ImportError:
    print("Error: pybtex not installed")
    print("Install with: pip install pybtex")
    sys.exit(1)


def validate_doi_crossref(doi, expected_title=None):
    """Validate a DOI against CrossRef API."""
    if not doi or not doi.strip():
        return False, None, None, "No DOI provided"

    # Format DOI for API (remove leading zeros if numeric-only)
    doi_clean = doi.strip().lstrip("0")

    # Try multiple URL formats
    urls_to_try = [
        f"https://api.crossref.org/works/{doi_clean}",
        f"https://api.crossref.org/works/{doi}",
    ]

    for url in urls_to_try:
        try:
            response = requests.get(
                url, params={"mailto": "bibtex-validator@example.com"}, timeout=10
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
                        if isinstance(title, list):
                            title = title[0] if len(title) > 0 else "Unknown"

                        # Get authors
                        authors = []
                        if "author" in item:
                            for author in item["author"]:
                                given = author.get("given", "")
                                family = author.get("family", "")
                                if given or family:
                                    authors.append(f"{given} {family}")
                                elif given:
                                    authors.append(given)
                                elif family:
                                    authors.append(family)

                        author_str = ", ".join(authors[:3]) + (
                            " et al." if len(authors) > 3 else ""
                        )

                        info_display = title[:60] + "..." if len(title) > 60 else title
                        info = info_display
                        if author_str:
                            info = f"{author_str} ({info})"

                        return True, info, url, None
                    else:
                        return False, None, url, "DOI found but no title"
                else:
                    return (
                        False,
                        None,
                        url,
                        f"Invalid DOI: {data.get('status', 'unknown')}",
                    )
            elif response.status_code == 404:
                return False, None, url, "DOI not found (404)"
            elif response.status_code == 429:
                return False, None, url, "Rate limited (429)"
        except requests.exceptions.Timeout:
            continue  # Try next URL format
        except Exception as e:
            continue  # Try next URL format

    return False, None, None, "DOI validation failed"


def extract_citations_from_tex(tex_files_dir):
    """Extract all citations from .tex files."""
    tex_dir = Path(tex_files_dir)
    citations = set()

    for tex_file in tex_dir.glob("*.tex"):
        with open(tex_file, "r", encoding="utf-8") as f:
            content = f.read()
            # Extract citations using regex
            matches = re.findall(r"\\cite{([^}]+)}", content)
            for match in matches:
                for key in match.split(","):
                    citations.add(key.strip())

    return citations


def load_bibtex_database(bib_file):
    """Load BibTeX file using pybtex."""
    bib_data = PybtexEngine().format_from_file(bib_file, None)

    # Build a dictionary of entries by key
    entries_by_key = {}
    for entry in bib_data.entries:
        entries_by_key[entry.key] = entry

    return bib_data, entries_by_key


def main():
    if len(sys.argv) < 3:
        print("Usage: python validate_citations.py <bib_file.bib> <tex_directory>")
        print("Example: python validate_citations.py references.bib content/")
        sys.exit(1)

    bib_file = Path(sys.argv[1])
    tex_dir = Path(sys.argv[2])

    if not bib_file.exists():
        print(f"Error: BibTeX file not found: {bib_file}")
        sys.exit(1)

    if not tex_dir.exists():
        print(f"Error: TeX directory not found: {tex_dir}")
        sys.exit(1)

    print("=" * 80)
    print("CITATION VALIDATION TOOL")
    print("=" * 80)
    print(f"BibTeX file: {bib_file}")
    print(f"TeX directory: {tex_dir}")
    print()

    # Load BibTeX database
    print("Loading BibTeX database...")
    bib_data, entries_by_key = load_bibtex_database(bib_file)
    print(f"Loaded {len(bib_data.entries)} entries from BibTeX file")
    print()

    # Extract citations from .tex files
    print("Extracting citations from .tex files...")
    cited_keys = extract_citations_from_tex(tex_dir)
    print(f"Found {len(cited_keys)} unique citations")
    print()

    # Validate each citation
    print("=" * 80)
    print("VALIDATING CITATIONS")
    print("=" * 80)
    print()

    not_in_bib = []
    valid_count = 0
    invalid_doi_count = 0
    title_mismatch_count = 0

    for i, cite_key in enumerate(sorted(cited_keys), 1):
        print(f"[{i}/{len(cited_keys)}] Checking: {cite_key}")

        if cite_key not in entries_by_key:
            print(f"  ❌ NOT IN BIBTEX FILE")
            not_in_bib.append(cite_key)
        else:
            entry = entries_by_key[cite_key]
            doi = entry.fields.get("doi", None)
            title = str(entry.fields.get("title", ""))

            title_display = title[:60] + "..." if len(title) > 60 else title
            print(f"  Title: {title_display}")

            if doi:
                is_valid, crossref_info, crossref_url, error_msg = (
                    validate_doi_crossref(doi, title)
                )

                if is_valid:
                    print(f"  ✅ DOI VALID: {crossref_info}")
                    if crossref_url:
                        print(f"     URL: {crossref_url}")
                    valid_count += 1
                else:
                    print(f"  ❌ DOI INVALID: {error_msg}")
                    if crossref_url:
                        print(f"     URL: {crossref_url}")
                    invalid_doi_count += 1

                    # Check if titles match
                    if crossref_info and title and crossref_info not in title:
                        print(f"  ⚠️  TITLE MISMATCH!")
                        bib_title_display = title[:80] if len(title) > 80 else title
                        crossref_title_display = (
                            crossref_info[:80]
                            if len(crossref_info) > 80
                            else crossref_info
                        )
                        print(f"     BibTeX: {bib_title_display}")
                        print(f"     CrossRef: {crossref_title_display}")
                        title_mismatch_count += 1
            else:
                print(f"  ℹ️  No DOI")
            print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total citations checked: {len(cited_keys)}")
    print(f"Citations in BibTeX: {len(cited_keys) - len(not_in_bib)}")
    print(f"Missing from BibTeX: {len(not_in_bib)}")
    print(f"Valid DOIs: {valid_count}")
    print(f"Invalid DOIs: {invalid_doi_count}")
    print(f"Title mismatches: {title_mismatch_count}")
    print()

    # Report issues
    if not_in_bib:
        print("⚠️  CITATIONS NOT IN BIBTEX:")
        for key in sorted(not_in_bib):
            print(f"  - {key}")
        print()

    if title_mismatch_count > 0:
        print("⚠️  TITLE MISMATCHES DETECTED (DOI resolves to different paper):")
        print()

    # Exit code
    if not_in_bib or title_mismatch_count > 0:
        sys.exit(1)
    else:
        print("✅ All citations validated successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
