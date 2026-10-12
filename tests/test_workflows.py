"""What the CI and release workflows are allowed to do, and what they run.

Neither question had an answer in the files. `ci.yml` declared no permissions at
all, so its token was whatever the repository settings happened to say — and it
runs code from the pull request head: `pip install -e` executes the PR's build
backend and pytest imports its conftest. In `release.yml` only the build job
declared any, so the smoke test and the canary deploy inherited the default,
and the canary job is the one holding cluster credentials.

Every action was also on a moving tag. `v4` is a pointer: whoever controls the
action's repository can re-point it, and the next run would execute whatever it
pointed at — with the token, and in the release workflow alongside a kubeconfig.
"""
import re
import unittest
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
WORKFLOWS = sorted((REPO / ".github/workflows").glob("*.yml"))

# A commit, not a tag or a branch.
PINNED = re.compile(r"^[0-9a-f]{40}$")

# The one scope that has to be writable, and the single job that needs it:
# pushing the image to GHCR.
EXPECTED_WRITERS = {("release.yml", "build"): {"packages"}}


def _parsed(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _jobs(path: Path) -> dict:
    return _parsed(path).get("jobs") or {}


class TestEveryWorkflowSaysWhatItMayDo(unittest.TestCase):
    def test_there_are_workflows_to_check(self):
        """So a rename cannot make this file vacuously pass."""
        self.assertTrue(WORKFLOWS)
        self.assertIn("ci.yml", [p.name for p in WORKFLOWS])
        self.assertIn("release.yml", [p.name for p in WORKFLOWS])

    def test_no_job_inherits_the_repository_default(self):
        """Declared at workflow level or on the job itself; `{}` counts, absent
        does not."""
        for path in WORKFLOWS:
            workflow = _parsed(path)
            at_workflow_level = "permissions" in workflow
            for name, job in (workflow.get("jobs") or {}).items():
                with self.subTest(workflow=path.name, job=name):
                    self.assertTrue(
                        at_workflow_level or "permissions" in job,
                        f"{path.name}:{name} runs with whatever the repository "
                        "default is")

    def test_only_the_image_push_may_write_anything(self):
        for path in WORKFLOWS:
            workflow = _parsed(path)
            scopes = {}
            if isinstance(workflow.get("permissions"), dict):
                scopes[(path.name, "<workflow>")] = workflow["permissions"]
            for name, job in (workflow.get("jobs") or {}).items():
                if isinstance(job.get("permissions"), dict):
                    scopes[(path.name, name)] = job["permissions"]
            for where, declared in scopes.items():
                writable = {scope for scope, level in declared.items()
                            if level == "write"}
                with self.subTest(where=where):
                    self.assertEqual(writable, EXPECTED_WRITERS.get(where, set()),
                                     f"{where} may write {sorted(writable)}")

    def test_the_credential_holding_job_may_do_nothing_here(self):
        """canary-deploy has no checkout and no action: it talks to the cluster
        with KUBECONFIG_B64. The job holding credentials should be the one able
        to do least with the repository."""
        job = _jobs(REPO / ".github/workflows/release.yml")["canary-deploy"]
        self.assertEqual(job.get("permissions"), {})

    def test_the_pull_request_workflow_is_read_only(self):
        """It runs PR-head code, which is the one place an escalation would be
        somebody else's to trigger."""
        workflow = _parsed(REPO / ".github/workflows/ci.yml")
        self.assertEqual(workflow.get("permissions"), {"contents": "read"})


class TestEveryActionIsPinnedToACommit(unittest.TestCase):
    @staticmethod
    def _uses(path: Path):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip().lstrip("- ")
            if stripped.startswith("uses:"):
                yield number, stripped[len("uses:"):].strip()

    def test_nothing_runs_from_a_moving_tag(self):
        found = 0
        for path in WORKFLOWS:
            for number, spec in self._uses(path):
                found += 1
                reference = spec.split("#")[0].strip().split("@")[-1]
                with self.subTest(workflow=path.name, line=number, uses=spec):
                    self.assertTrue(
                        PINNED.match(reference),
                        f"{path.name}:{number} runs {reference!r}, which whoever "
                        "owns that action can re-point")
        self.assertGreater(found, 0, "no `uses:` lines found at all")

    def test_each_pin_says_which_version_it_is(self):
        """A bare 40-character hash tells a reader nothing about whether it is
        current. The comment is how the pin stays reviewable."""
        for path in WORKFLOWS:
            for number, spec in self._uses(path):
                with self.subTest(workflow=path.name, line=number):
                    self.assertIn("#", spec,
                                  "pinned with no version comment beside it")
                    self.assertTrue(spec.split("#", 1)[1].strip(),
                                    "empty version comment")

    def test_the_convention_is_explained_once_per_workflow(self):
        """Otherwise the next person to add a step copies a tag, and the pins
        rot into a mixed set that looks deliberate."""
        for path in WORKFLOWS:
            with self.subTest(workflow=path.name):
                text = path.read_text(encoding="utf-8")
                self.assertIn("moving pointer", text)

class TestTheTestedEnvironmentIsTheShippedOne(unittest.TestCase):
    """CI installed pyproject's floors; the Dockerfile installs the lock. For
    four days that difference made the suite red on a commit that touched
    neither file.

    `unstructured>=0.18` resolved to 0.27.25, and unstructured 0.21.0 replaced
    nltk with spaCy. Three tests that read the INSTALLED tokenizer's source to
    check the Dockerfile's corpora bake were therefore reading a module that no
    longer mentions nltk, and a fourth asked importlib for the version of a
    package the floors no longer install at all. The assertions were right. The
    environment was not the one they were written about, and nothing said so.

    These are about the pipeline, not about the packages: that the job which
    gates a push installs the pinned closure, and that the nightly job does
    NOT, because resolving the floors freely is how drift like this surfaces.
    """

    CI = REPO / ".github/workflows/ci.yml"

    @classmethod
    def setUpClass(cls):
        cls.jobs = _jobs(cls.CI)

    @staticmethod
    def _install_step(job) -> dict:
        [step] = [s for s in job["steps"]
                  if "pip install" in str(s.get("run", ""))]
        return step

    def test_the_push_path_installs_under_the_lock(self):
        step = self._install_step(self.jobs["test"])
        self.assertIn(
            "-c requirements.txt", step["run"],
            "the job that gates every push resolves pyproject's floors, so it "
            "tests whatever PyPI offers today rather than what the Dockerfile "
            "installs. Constrain it to the lock.")

    def test_the_nightly_job_is_left_unconstrained_on_purpose(self):
        """The other half. Constraining both would make the floors
        unobserved, and an unobserved floor is how this happened."""
        step = self._install_step(self.jobs["full-deps"])
        self.assertNotIn("-c requirements.txt", step["run"])
        self.assertIn('-e ".[all]"', step["run"])

    def test_the_nightly_job_is_not_on_the_push_path(self):
        """Because it is the one allowed to go red on somebody else's release.
        If it ever gates a push, drift stops being information and starts
        being a blocked merge."""
        condition = self.jobs["full-deps"]["if"]
        self.assertIn("schedule", condition)
        self.assertIn("workflow_dispatch", condition)

    def test_the_pip_cache_key_notices_a_lock_change(self):
        """It was keyed on pyproject alone, so editing requirements.txt
        reused a cache built from different pins."""
        [setup] = [s for s in self.jobs["test"]["steps"]
                   if "setup-python" in str(s.get("uses", ""))]
        key = setup["with"]["cache-dependency-path"]
        for name in ("pyproject.toml", "requirements.txt"):
            with self.subTest(file=name):
                self.assertIn(name, key)

    def test_the_corpora_download_is_off_where_the_lock_is_installed(self):
        """unstructured 0.18.21 calls nltk.download() at module scope unless
        AUTO_DOWNLOAD_NLTK is false, and tests/test_packaging.py imports that
        module. Under the constraint CI gets that unstructured, so without this
        every run reaches the network inside a `quiet=True` call that says
        nothing when it fails. The image sets the same variable."""
        [step] = [s for s in self.jobs["test"]["steps"]
                  if "pytest" in str(s.get("run", ""))]
        self.assertEqual(step["env"].get("AUTO_DOWNLOAD_NLTK"), "false")



if __name__ == "__main__":                    # pragma: no cover
    unittest.main()
