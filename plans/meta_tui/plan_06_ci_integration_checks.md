# Plan 06: TUI Automated Architectural Validation in CI

**Version**: 1.0
**Date**: 2025-05-23
**Author**: MasterMind Architect

## 1. Introduction & Goal

**Problem**: Architectural principles and design decisions, even when well-defined and initially implemented, can erode over time as a codebase evolves. Manual reviews can catch some deviations, but automated checks are crucial for consistently maintaining architectural integrity and preventing "architectural drift." Without such checks, issues like high coupling, circular dependencies, or violations of module boundaries can re-emerge.

**Goal**: To integrate automated architectural validation into the Continuous Integration (CI) pipeline for the `openhcs.tui` package. This will provide early feedback on pull requests and ensure that changes adhere to the established architectural guidelines. The checks will leverage the existing `tools/code_analysis/` scripts.

**Architectural Principles**:
*   **Continuous Verification**: Architectural conformance should be checked continuously, not just periodically.
*   **Early Feedback**: Developers should receive feedback on architectural violations as early as possible in the development cycle.
*   **Automation**: Manual architectural reviews should be supplemented by automated checks to ensure consistency and reduce reviewer burden.
*   **Explicit Architectural Rules**: The CI checks will codify and enforce key architectural rules.

## 2. Proposed CI Integration Steps

This plan assumes a CI system like GitHub Actions, GitLab CI, or Jenkins is in use. The specific implementation details will vary based on the CI system, but the general steps are:

### 2.1. Scripting for CI Execution

*   **Action**: Ensure the existing analysis tools in `tools/code_analysis/` can be run non-interactively and produce machine-parseable output or clear success/failure exit codes.
    *   `meta_analyzer.py`: This script already orchestrates multiple analyzers. Ensure it has a mode that:
        *   Outputs a summary report (e.g., to a file or stdout).
        *   Exits with a non-zero code if critical violations are found.
    *   Individual analyzers (`call_graph_analyzer.py`, `import_validator.py`, `interface_classifier.py`, `semantic_role_analyzer.py`, `code_analyzer_cli.py` for dependencies and async patterns):
        *   Confirm they can be targeted at `openhcs/tui`.
        *   Ensure they can output results to specified files or stdout.
        *   Modify them to exit with a non-zero status code if specific error conditions are met (e.g., `import_validator.py` finding missing/unused imports, `call_graph_analyzer.py` finding new critical circular dependencies).

### 2.2. Defining CI Workflow/Job

*   **Action**: Create a new CI workflow (e.g., a YAML file for GitHub Actions) or add a new job/stage to an existing workflow that runs on every pull request targeting the main development branch.
*   **Workflow Steps**:
    1.  **Checkout Code**: Fetch the latest code from the pull request branch.
    2.  **Setup Python Environment**: Install Python and project dependencies (including those needed by the analysis tools).
    3.  **Run Architectural Checks**:
        *   Execute `python tools/code_analysis/meta_analyzer.py comprehensive openhcs/tui --ci-mode ...` (a new `--ci-mode` flag might be added to `meta_analyzer.py` to control output and exit codes for CI).
        *   Alternatively, run individual analyzers if more granular control over failure conditions is needed:
            *   `python tools/code_analysis/import_validator.py openhcs/tui --fail-on-issues`
            *   `python tools/code_analysis/code_analyzer_cli.py dependencies openhcs/tui --check-cycles --fail-on-new-cycles` (requires enhancing `code_analyzer_cli.py` to detect *new* cycles against a baseline or defined allowed cycles).
            *   `python tools/code_analysis/code_analyzer_cli.py async-patterns openhcs/tui --fail-on-unawaited`
            *   Other checks as deemed critical (e.g., a custom script to check for direct core imports from UI modules after Plan 01 is implemented).
    4.  **Report Results**:
        *   If checks produce report files (e.g., Markdown, CSV), upload them as CI artifacts for review.
        *   The CI job should fail if any of the critical checks return a non-zero exit code.

### 2.3. Defining Failure Conditions and Thresholds

*   **Action**: Define what constitutes a critical architectural violation that should fail a CI build. This requires careful consideration to avoid being overly restrictive initially.
*   **Initial Critical Violations to Consider Failing CI For**:
    *   **New Missing or Unused Imports**: `import_validator.py` should fail if it finds any.
    *   **New Circular Dependencies**: `code_analyzer_cli.py dependencies` (or `call_graph_analyzer.py`) needs a mechanism to compare against a baseline or a list of known/accepted cycles. New, unapproved cycles should fail the build.
    *   **New Unawaited Coroutines**: `code_analyzer_cli.py async-patterns` should fail if it finds any.
    *   **(Post-Plan 01)** **Direct Core Imports from UI Layer**: A custom script could grep for forbidden import patterns from specific TUI modules (e.g., `from openhcs.core.orchestrator import ...` in a `*.ui.component.py` file).
*   **Non-Failing Reports (for now)**:
    *   Changes in semantic roles, interface implementations, or overall definition counts might initially just generate reports for review, rather than failing the build, unless specific undesirable patterns are identified as critical.
    *   Call graph hotspots: Report changes but don't fail initially.

### 2.4. Baseline and Iteration

*   **Action**:
    1.  **Establish a Baseline**: Before enabling strict CI failures, run all analysis tools on the current main branch to understand the existing state of architectural "debt."
    2.  **Iterative Rollout**:
        *   Start by having the CI job run the analyzers and upload reports, but not fail the build for most checks. This allows developers to get used to seeing the reports.
        *   Gradually enable failure conditions, starting with the most critical and unambiguous violations (e.g., new unused imports).
        *   For issues like circular dependencies, a list of currently accepted cycles might be maintained, and the CI would only fail if *new* unlisted cycles are introduced. The long-term goal would be to reduce this accepted list.

### 2.5. Documentation and Developer Guidance

*   **Action**:
    1.  Document the architectural principles being enforced by the CI checks.
    2.  Provide clear guidance to developers on how to interpret the reports generated by the analysis tools.
    3.  Explain how to fix common violations.
    4.  Document the process for addressing accepted architectural debt or requesting exceptions if a deviation is justified.

## 3. Example CI Job Snippet (Conceptual GitHub Actions)

```yaml
name: TUI Architectural Validation

on:
  pull_request:
    paths:
      - 'openhcs/tui/**'
      - 'tools/code_analysis/**'

jobs:
  validate-tui-architecture:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v3
        with:
          python-version: '3.9' # Or project's Python version

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt # Or project-specific setup

      - name: Run Import Validator
        run: |
          python tools/code_analysis/import_validator.py openhcs/tui --output-file reports_ci/import_issues.md --fail-on-issues
        # continue-on-error: false # Default, will fail job on non-zero exit

      - name: Run Async Pattern Check
        run: |
          python tools/code_analysis/code_analyzer_cli.py async-patterns openhcs/tui --output-dir reports_ci/async_analysis --fail-on-unawaited
        # continue-on-error: false

      - name: Run Dependency Cycle Check
        # This step needs enhancement in the tool to compare against a baseline or fail on any cycle
        run: |
          python tools/code_analysis/code_analyzer_cli.py dependencies openhcs/tui -o reports_ci/dependencies.md # --fail-on-new-cycles (hypothetical flag)
        # continue-on-error: true # Initially, might not fail for this

      # Add other analyzers as needed

      - name: Upload Analysis Reports
        if: always() # Upload reports even if previous steps failed
        uses: actions/upload-artifact@v3
        with:
          name: tui-architecture-reports
          path: reports_ci/
```

## 4. Verification

1.  **CI Workflow Execution**: Confirm that the CI job triggers on relevant pull requests.
2.  **Tool Execution in CI**: Verify that all analysis scripts run correctly in the CI environment and produce the expected output/artifacts.
3.  **Failure Conditions**: Test that the CI job correctly fails when a defined critical architectural violation is introduced in a pull request.
4.  **Report Accessibility**: Ensure that reports uploaded as artifacts are accessible and understandable.
5.  **Developer Feedback Loop**: Monitor developer feedback on the CI checks to refine thresholds, improve error messages, and ensure the process is helpful rather than overly burdensome.

By implementing these CI checks, the project can proactively maintain and improve the architectural quality of the `openhcs.tui` package.