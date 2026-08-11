# OpenHCS documentation audit

This directory records editorial review evidence for the active Read the Docs
corpus. It is a quality-assurance ledger, not an authority for product facts.
Configuration, callable, runtime, and package semantics remain on their owning
declarations or in the owning package documentation.

Each JSON file contains an array with one object for every reviewed RST source:

- `path`: repository-relative source path;
- `source_sha256`: digest of the exact reviewed RST content;
- `audience`: the practitioners the page serves;
- `user_need`: one concrete need, phrased from the reader's perspective;
- `diataxis`: `tutorial`, `how-to`, `reference`, or `explanation`;
- `authority`: current declarations, implementation, tests, or owner files used
  to verify the page. Each entry records `path`, `sha256`, and the evidence
  `role` that file served;
- `findings`: accuracy, freshness, duplication, boundary, flow, and navigation
  findings discovered during review;
- `disposition`: `keep`, `revise`, `split`, `merge`, `remove`, or `redirect`;
- `validation`: evidence used to validate the final page.

An empty `findings` array means the reviewer challenged the page against all
of those dimensions and found no actionable defect. It must not mean that the
page received only a build check. Generated reference should name its
declaration authority rather than copy generated field facts into this ledger.

The audit validator requires exactly one entry for every active RST source and
rejects entries for Sphinx-excluded archive material. It also rejects a digest
after the reviewed page changes, forcing the editorial assessment and its
evidence to be revisited with the prose. Authority digests provide the reverse
gate: changing an implementation, declaration, test, package manifest, or
owner file invalidates every dependent page until its claims are reassessed.
Every page requires at least one non-documentation authority; another OpenHCS
page cannot prove a product claim.

## Review workflow

When a page or one of its declared authorities changes:

1. Find every dependent record, for example with
   `rg -l '"path": "openhcs/core/config.py"' docs/audits`.
2. Re-read the changed owner and the complete dependent page. Correct or remove
   claims that the owner no longer supports.
3. Update the evidence role, findings, disposition, and validation where the
   review changes their meaning.
4. Refresh page and authority digests only after that comparison is complete.
5. Run `python scripts/validate_docs.py docs/source` and the page's listed
   behavioral validation before committing.

Do not mass-refresh digests merely to make the gate pass. A changed digest is
the signal that the source comparison must be repeated.
