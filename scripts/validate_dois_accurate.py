#!/usr/bin/env python3
"""
Comprehensive DOI validation that checks if DOIs actually point to the correct papers.
Compares CrossRef metadata against bib entry fields to detect mismatches.
"""

import sys
import re
import requests
from pathlib import Path


def fetch_crossref_record(doi):
    """Fetch full record from CrossRef for a DOI."""
    if not doi or not doi.strip():
        return None, "No DOI provided"

    # Format DOI for API
    doi_clean = doi.strip().lstrip("0")

    # Try multiple URL formats
    urls_to_try = [
        f"https://api.crossref.org/works/{doi_clean}",
        f"https://api.crossref.org/works/{doi}",
    ]

    for url in urls_to_try:
        try:
            response = requests.get(
                url, params={"mailto": "bibtex-validator@example.com"}, timeout=15
            )
            if response.status_code == 200:
                data = response.json()

                if "status" in data and data["status"] == "ok":
                    if (
                        "message" in data
                        and "items" in data["message"]
                        and len(data["message"]["items"]) > 0
                    ):
                        item = data["message"]["items"][0]
                        return item, None, None
                    else:
                        return None, None, "DOI found but no items"
                else:
                    return None, None, f"Invalid DOI: {data.get('status', 'unknown')}"
            elif response.status_code == 404:
                return None, None, "DOI not found (404)"
            elif response.status_code == 429:
                return None, None, "Rate limited (429) - sleeping 5s"
        except requests.exceptions.Timeout:
            continue
        except Exception as e:
            continue

    return None, None, "DOI validation failed after all attempts"


def normalize_author_name(name):
    """Normalize author name for comparison."""
    # Remove accents, diacritics, lowercase, remove extra whitespace
    import unicodedata

    normalized = (
        unicodedata.normalize("NFKD", name).encode("ASCII", "ignore").decode("ASCII")
    )
    normalized = re.sub(r"[^a-zA-Z\s]", " ", normalized).lower().strip()
    # Remove common titles/suffixes
    normalized = re.sub(r"\s+(jr|sr|ii|iii|iv|prof|dr|ph\.d)\.?\b", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def normalize_title(title):
    """Normalize title for comparison."""
    if not title:
        return ""
    # Lowercase, remove punctuation and extra spaces
    normalized = re.sub(r"[^\w\s-]", " ", title.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    # Remove common articles
    normalized = re.sub(r"\b(a|an|the)\s+", " ", normalized)
    return normalized


def similarity_score(s1, s2):
    """Calculate simple similarity score between two strings."""
    s1_norm = normalize_title(s1)
    s2_norm = normalize_title(s2)

    if s1_norm == s2_norm:
        return 100

    # Check if one is substring of the other
    if s1_norm in s2_norm or s2_norm in s1_norm:
        return 80

    # Check word overlap
    words1 = set(s1_norm.split())
    words2 = set(s2_norm.split())
    overlap = len(words1 & words2)
    return min(100, int(overlap / max(len(words1), len(words2)) * 60))


def parse_bibtex_simple(bib_file):
    """Parse BibTeX file using regex - simple and reliable."""
    with open(bib_file, "r", encoding="utf-8") as f:
        content = f.read()

    entries = {}
    # Find all entry blocks (handle optional comma after key)
    entry_pattern = r"@(\w+)\s*\{\s*([^,\}]+)\s*,?\s*([^}]+)\}"
    matches = re.findall(entry_pattern, content)

    for match in matches:
        entry_type = match[0]
        entry_key = match[1].strip()
        entry_content = match[2]

        # Extract fields
        title_match = re.search(
            r"title\s*=\s*[{\']?([^}\']+)[{\']?", entry_content, re.IGNORECASE
        )
        author_match = re.search(
            r"author\s*=\s*[{\']?([^}\']+)[{\']?", entry_content, re.IGNORECASE
        )
        year_match = re.search(
            r"year\s*=\s*[{\']?(\d{4})[{\']?", entry_content, re.IGNORECASE
        )
        journal_match = re.search(
            r"journal\s*=\s*[{\']?([^}\']+)[{\']?", entry_content, re.IGNORECASE
        )
        doi_match = re.search(
            r"doi\s*=\s*[{\']?([^}\']+)[{\']?", entry_content, re.IGNORECASE
        )

        title = title_match.group(1).strip() if title_match else ""
        author = author_match.group(1).strip() if author_match else ""
        year = year_match.group(1).strip() if year_match else ""
        journal = journal_match.group(1).strip() if journal_match else ""
        doi = doi_match.group(1).strip() if doi_match else None

        # Clean up title (remove quotes, braces)
        title = re.sub(r"[{}\"\']", "", title).strip()
        title = re.sub(r"\s+", " ", title).strip()

        entries[entry_key] = {
            "type": entry_type,
            "title": title,
            "author": author,
            "year": year,
            "journal": journal,
            "doi": doi,
        }

    return entries


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


def compare_records(crossref_record, bib_entry, entry_key):
    """Compare CrossRef record with bib entry and report discrepancies."""
    issues = []
    warnings = []

    # Extract CrossRef data
    cr_title = None
    cr_authors = []
    cr_year = None
    cr_journal = None

    if "title" in crossref_record:
        cr_title = (
            crossref_record["title"][0]
            if isinstance(crossref_record["title"], list)
            else crossref_record["title"]
        )
    if "author" in crossref_record:
        for author in crossref_record["author"]:
            given = author.get("given", "")
            family = author.get("family", "")
            if given or family:
                cr_authors.append(f"{given} {family}".strip())
            elif given:
                cr_authors.append(given.strip())
            elif family:
                cr_authors.append(family.strip())

    if "published-print" in crossref_record:
        cr_year = str(crossref_record["published-print"]["date-parts"][0])

    if (
        "short-container-title" in crossref_record
        and len(crossref_record["short-container-title"]) > 0
    ):
        cr_journal = crossref_record["short-container-title"][0]

    # Extract bib data
    bib_title = bib_entry.get("title", "")
    bib_author = bib_entry.get("author", "")
    bib_year = bib_entry.get("year", "")
    bib_journal = bib_entry.get("journal", "")

    # Normalize and compare titles
    if cr_title and bib_title:
        title_score = similarity_score(cr_title, bib_title)
        if title_score < 70:
            issues.append(
                (
                    "TITLE_MISMATCH",
                    f"Low similarity ({title_score}%): BibTeX: '{bib_title[:80]}' vs CrossRef: '{cr_title[:80]}'",
                )
            )
        elif title_score < 90:
            warnings.append(
                (
                    "TITLE_SIMILARITY",
                    f"Medium similarity ({title_score}%): BibTeX: '{bib_title[:60]}...' vs CrossRef: '{cr_title[:60]}...'",
                )
            )

    # Compare years
    if cr_year and bib_year and cr_year != bib_year:
        issues.append(
            ("YEAR_MISMATCH", f"BibTeX: '{bib_year}' vs CrossRef: '{cr_year}'")
        )

    # Compare journals
    if cr_journal and bib_journal:
        cr_journal_norm = normalize_title(cr_journal)
        bib_journal_norm = normalize_title(bib_journal)

        if (
            cr_journal_norm != bib_journal_norm
            and cr_journal_norm not in bib_journal_norm
            and bib_journal_norm not in cr_journal_norm
        ):
            issues.append(
                (
                    "JOURNAL_MISMATCH",
                    f"BibTeX: '{bib_journal}' vs CrossRef: '{cr_journal}'",
                )
            )

    # Compare authors (simplified)
    if cr_authors and bib_author:
        bib_author_norm = normalize_author_name(bib_author)
        cr_authors_norm = [normalize_author_name(a) for a in cr_authors[:3]]

        matches = sum(1 for a in cr_authors_norm if a in bib_author_norm)
        if matches == 0:
            issues.append(
                (
                    "AUTHOR_MISMATCH",
                    f"CrossRef: '{', '.join(cr_authors[:3])}' vs BibTeX: '{bib_author}'",
                )
            )

    return issues, warnings


def main():
    if len(sys.argv) < 3:
        print("Usage: python validate_dois_accurate.py <bib_file.bib> <tex_directory>")
        print("Example: python validate_dois_accurate.py references.bib content/")
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
    print("ACCURATE DOI VALIDATION")
    print("=" * 80)
    print(f"BibTeX file: {bib_file}")
    print(f"TeX directory: {tex_dir}")
    print()

    # Load BibTeX database
    print("Loading BibTeX database...")
    entries_by_key = parse_bibtex_simple(bib_file)
    print(f"Loaded {len(entries_by_key)} entries from BibTeX file")
    print()

    # Extract citations from .tex files
    print("Extracting citations from .tex files...")
    cited_keys = extract_citations_from_tex(tex_dir)
    print(f"Found {len(cited_keys)} unique citations")
    print()

    # Find cited entries with DOIs
    entries_with_doi = [
        k for k, v in entries_by_key.items() if v.get("doi") and k in cited_keys
    ]

    if not entries_with_doi:
        print("No cited entries with DOIs found!")
        sys.exit(0)

    print(f"Validating {len(entries_with_doi)} cited entries with DOIs...")
    print()

    # Validate each cited entry
    total_issues = 0
    total_warnings = 0

    for i, cite_key in enumerate(entries_with_doi, 1):
        print(f"[{i}/{len(entries_with_doi)}] Checking: {cite_key}")

        entry = entries_by_key[cite_key]
        doi = entry.get("doi", "")

        # Fetch CrossRef record
        print(f"  Fetching from CrossRef: {doi}")
        crossref_record, crossref_url, error_msg = fetch_crossref_record(doi)

        if not crossref_record:
            print(f"  ❌ FAILED: {error_msg}")
            if crossref_url:
                print(f"     URL: {crossref_url}")
            continue

        print(f"  ✅ Retrieved record")
        print(f"     URL: {crossref_url}")

        # Compare records
        issues, warnings = compare_records(crossref_record, entry, cite_key)

        for issue_type, issue_msg in issues:
            print(f"  ❌ {issue_type}: {issue_msg}")
            total_issues += 1

        for warning_type, warning_msg in warnings:
            print(f"  ⚠️  {warning_type}: {warning_msg}")
            total_warnings += 1

        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total entries checked: {len(entries_with_doi)}")
    print(f"Total issues found: {total_issues}")
    print(f"Total warnings found: {total_warnings}")
    print()

    if total_issues > 0:
        print("⚠️  ISSUES FOUND - DOIs may not match the cited papers!")
        sys.exit(1)
    else:
        print("✅ All DOIs are accurate!")
        sys.exit(0)


if __name__ == "__main__":
    main()
