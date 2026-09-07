"""Every job in a filtered workflow follows its `changes` job.

A job that forgets the gate runs on every pull request, and a filter list that
omits the workflow file skips the run that would test a change to it.
"""

import pytest
import yaml

from hdf5_pure_scripts import repo_root

WORKFLOWS = repo_root() / ".github" / "workflows"


def load(name):
    return yaml.safe_load((WORKFLOWS / name).read_text())


def filtered():
    return sorted(p.name for p in WORKFLOWS.glob("*.yml") if "changes" in load(p.name)["jobs"])


def needs(job):
    declared = job.get("needs", [])
    return [declared] if isinstance(declared, str) else declared


@pytest.mark.parametrize("name", filtered())
def test_the_filter_names_the_workflow_file(name):
    step = load(name)["jobs"]["changes"]["steps"][0]
    assert f"- .github/workflows/{name}\n" in step["with"]["filters"]


@pytest.mark.parametrize("name", filtered())
def test_every_job_follows_changes(name):
    jobs = load(name)["jobs"]

    def gated(job_id):
        job = jobs[job_id]
        if "changes" in needs(job):
            return (
                job_id.endswith("-guard") or job.get("if") == "needs.changes.outputs.run == 'true'"
            )
        return any(gated(n) for n in needs(job))

    for job_id in jobs:
        if job_id != "changes":
            assert gated(job_id), f"{name}: {job_id} does not follow changes"
