from scripts.resolve_successful_job_baseline import (
    WorkflowJob,
    WorkflowRun,
    select_successful_job_baseline,
)


def test_baseline_uses_job_success_independently_of_workflow_outcome() -> None:
    current = WorkflowRun(run_id=4, head_sha="head")
    cancelled_after_success = WorkflowRun(run_id=3, head_sha="candidate")
    older = WorkflowRun(run_id=2, head_sha="older")
    jobs = {
        3: (WorkflowJob(name="code-quality", successful=True),),
        2: (WorkflowJob(name="code-quality", successful=True),),
    }

    baseline = select_successful_job_baseline(
        (current, cancelled_after_success, older),
        current_head_sha=current.head_sha,
        job_name="code-quality",
        jobs_for_run=jobs.__getitem__,
        is_ancestor=lambda candidate, _head: candidate in {"candidate", "older"},
    )

    assert baseline == cancelled_after_success


def test_baseline_skips_failed_jobs_and_non_ancestor_runs() -> None:
    runs = (
        WorkflowRun(run_id=3, head_sha="other-branch"),
        WorkflowRun(run_id=2, head_sha="failed"),
        WorkflowRun(run_id=1, head_sha="successful"),
    )
    jobs = {
        3: (WorkflowJob(name="code-quality", successful=True),),
        2: (WorkflowJob(name="code-quality", successful=False),),
        1: (WorkflowJob(name="code-quality", successful=True),),
    }

    baseline = select_successful_job_baseline(
        runs,
        current_head_sha="head",
        job_name="code-quality",
        jobs_for_run=jobs.__getitem__,
        is_ancestor=lambda candidate, _head: candidate != "other-branch",
    )

    assert baseline == runs[2]


def test_baseline_requires_the_named_job() -> None:
    run = WorkflowRun(run_id=1, head_sha="candidate")

    baseline = select_successful_job_baseline(
        (run,),
        current_head_sha="head",
        job_name="code-quality",
        jobs_for_run=lambda _run_id: (WorkflowJob(name="unit-tests", successful=True),),
        is_ancestor=lambda _candidate, _head: True,
    )

    assert baseline is None
