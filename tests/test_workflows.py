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
import json
import re
import tempfile
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

    # vMAJOR.MINOR at least. `# v4` was what every pin here carried, and it
    # cannot answer the only question a reader has — is this current? —
    # because the major has not moved in two years while the actions behind it
    # went from Node 20 to Node 24.
    EXACT = re.compile(r"^v\d+\.\d+(\.\d+)?$")

    def test_each_pin_says_exactly_which_version_it_is(self):
        """A bare 40-character hash tells a reader nothing about whether it is
        current, and `# v4` tells them almost nothing: it is a moving major,
        so it stays true while the pin rots underneath it. The comment is the
        only thing that makes these reviewable."""
        for path in WORKFLOWS:
            for number, spec in self._uses(path):
                with self.subTest(workflow=path.name, line=number, uses=spec):
                    self.assertIn("#", spec,
                                  "pinned with no version comment beside it")
                    comment = spec.split("#", 1)[1].strip()
                    self.assertRegex(
                        comment, self.EXACT,
                        f"{path.name}:{number} says {comment!r}. Give the exact "
                        "release the SHA came from, so the next reader can tell "
                        "whether it has moved.")

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

    def test_the_corpora_download_is_off_in_the_nightly_job_too(self):
        """It resolves the floors, and the floors are now bounded to the nltk
        era (see the note at the declaration in pyproject), so it lands on an
        unstructured that downloads at import just like the pinned one."""
        [step] = [s for s in self.jobs["full-deps"]["steps"]
                  if "pytest" in str(s.get("run", ""))]
        self.assertEqual(step["env"].get("AUTO_DOWNLOAD_NLTK"), "false")

    def test_the_corpora_download_is_off_where_the_lock_is_installed(self):
        """unstructured 0.18.21 calls nltk.download() at module scope unless
        AUTO_DOWNLOAD_NLTK is false, and tests/test_packaging.py imports that
        module. Under the constraint CI gets that unstructured, so without this
        every run reaches the network inside a `quiet=True` call that says
        nothing when it fails. The image sets the same variable."""
        [step] = [s for s in self.jobs["test"]["steps"]
                  if "pytest" in str(s.get("run", ""))]
        self.assertEqual(step["env"].get("AUTO_DOWNLOAD_NLTK"), "false")


class TestABuildThatExportsACacheCanExportACache(unittest.TestCase):
    """Four tagged releases, four failures, no image ever pushed — and the
    cause was two lines of cache configuration with no builder behind them.

    buildx's default driver on a GitHub runner is `docker`, the BuildKit inside
    dockerd, which reports CacheExport=false unless the containerd image store
    is on. buildx validates `cache-to` against the driver's features while it
    assembles the solve options, before it ever reads the Dockerfile, so the
    build died in one to three seconds with a single line of stderr. The step
    list of all four runs contains no "Set up Docker Buildx".

    The assertion is conditional on exporting a cache, so the last test here
    keeps it from passing because nothing does.
    """

    @staticmethod
    def _steps(job) -> list:
        return job.get("steps") or []

    def _exporting_jobs(self):
        for path in WORKFLOWS:
            for name, job in _jobs(path).items():
                steps = self._steps(job)
                exporters = [index for index, step in enumerate(steps)
                             if "cache-to" in str(step.get("with") or {})]
                if exporters:
                    yield path.name, name, steps, exporters

    def test_every_cache_exporting_job_sets_up_a_builder(self):
        for workflow, job, steps, _at in self._exporting_jobs():
            with self.subTest(workflow=workflow, job=job):
                self.assertTrue(
                    any("setup-buildx-action" in str(step.get("uses") or "")
                        for step in steps),
                    f"{workflow}:{job} passes cache-to with no "
                    "docker/setup-buildx-action step, so buildx runs on the "
                    "docker driver and rejects the export before building "
                    "anything")

    def test_the_builder_is_set_up_before_the_build_that_needs_it(self):
        """Steps run in order; a builder created afterwards is not the one the
        build used."""
        for workflow, job, steps, exporters in self._exporting_jobs():
            builder = [index for index, step in enumerate(steps)
                       if "setup-buildx-action" in str(step.get("uses") or "")]
            with self.subTest(workflow=workflow, job=job):
                self.assertTrue(builder)
                self.assertLess(min(builder), min(exporters))

    def test_something_actually_exports_a_cache(self):
        """Otherwise the two tests above hold over an empty set, which is how a
        conditional guard quietly stops guarding."""
        found = [(workflow, job) for workflow, job, _s, _a in self._exporting_jobs()]
        self.assertEqual(found, [("release.yml", "build")], found)


class TestTheSmokeTestCouldEverHavePassed(unittest.TestCase):
    """It could not, for three independent reasons, and it has never run —
    the build job has failed on every tagged release, so job 2 was skipped
    every time.

    1. It asked for `ghcr.io/NickSanft/MisterFritz`, built by hand from
       `IMAGE_NAME: ${{ github.repository }}`. metadata-action lowercases the
       names it pushes, so the registry holds `nicksanft/misterfritz`; and an
       OCI repository name may not contain uppercase at all, so docker rejects
       that string before contacting the registry.
    2. It asked for tag `sha-<40 characters>`, while `type=sha` produces the
       SHORT sha. That tag has never existed.
    3. It passed `python local-sim/mock_fritz.py` to an image whose exec-form
       ENTRYPOINT APPENDS arguments rather than replacing them, so the command
       was `python main_discord.py python local-sim/mock_fritz.py` and
       main_discord read none of it. It booted the real bot, failed the Discord
       login on a placeholder token, and was gone before the curl.
    """

    RELEASE = REPO / ".github/workflows/release.yml"

    @classmethod
    def setUpClass(cls):
        cls.jobs = _jobs(cls.RELEASE)
        cls.smoke = cls.jobs["test-image"]
        cls.runs = [step for step in cls.smoke["steps"] if "run" in step]

    def test_the_build_publishes_a_reference_rather_than_a_list_of_tags(self):
        """`steps.meta.outputs.tags` is newline-separated — four tags for a
        semver push — so it could never be interpolated into a shell
        command. Nothing read it, which is the only reason that was not a
        fourth bug."""
        image = self.jobs["build"]["outputs"]["image"]
        self.assertNotIn("meta.outputs.tags", image)
        self.assertIn("steps.reference.outputs.image", image)

    def test_the_reference_is_lowercased_and_pinned_to_a_digest(self):
        [step] = [s for s in self.jobs["build"]["steps"]
                  if s.get("id") == "reference"]
        self.assertIn("tr '[:upper:]'", step["run"],
                      "github.repository is mixed case and a repository name "
                      "may not be")
        self.assertIn("steps.push.outputs.digest", str(step["env"]),
                      "pin the digest, so the job that boots the image and the "
                      "job that deploys it get the same bytes")

    def test_the_build_step_is_identified_so_its_digest_can_be_read(self):
        [build] = [s for s in self.jobs["build"]["steps"]
                   if "build-push-action" in str(s.get("uses", ""))]
        self.assertEqual(build.get("id"), "push")

    def test_the_smoke_test_boots_the_image_through_its_own_entrypoint(self):
        """The step that SUPPLIES configuration, specifically: asserting over
        every run script joined together let the real boot be changed while the
        no-configuration check kept the word `--preflight` in the file."""
        [boot] = [step for step in self.runs
                  if "-e DISCORD_BOT_TOKEN" in step["run"]]
        self.assertIn("--preflight", boot["run"])
        commands = " ".join(step["run"] for step in self.runs)
        self.assertNotIn("--entrypoint", commands,
                         "overriding the entrypoint tests a program the image "
                         "does not start with")
        self.assertNotIn("mock_fritz", commands,
                         "the mock shares nothing with the application except "
                         "metric names, so it can pass against a broken image")

    def test_it_names_the_image_the_build_job_published(self):
        for step in self.runs:
            with self.subTest(step=step.get("name")):
                self.assertIn("needs.build.outputs.image", str(step.get("env") or {}))
                self.assertNotIn("IMAGE_NAME", step["run"])
                self.assertNotIn("github.sha", step["run"])

    def test_the_reference_reaches_the_shell_through_the_environment(self):
        """Not interpolated into the script. A value with a newline in it
        would otherwise become additional shell commands."""
        for step in self.runs:
            with self.subTest(step=step.get("name")):
                self.assertIn('"$IMAGE"', step["run"])

    def test_it_proves_the_image_carries_no_configuration_of_its_own(self):
        """.dockerignore excludes .env and config.json, so with neither
        variable set validate_config must refuse to boot. A pass there means
        somebody baked a config.json in, and every container is carrying a
        token."""
        [negative] = [step for step in self.runs
                      if "exit 1" in step["run"] and "if docker run" in step["run"]]
        self.assertNotIn("-e DISCORD_BOT_TOKEN", negative["run"],
                         "the point is that it runs with NO configuration")
        self.assertIn("dockerignore", negative["run"].lower())

    def test_it_is_allowed_to_pull_what_the_build_job_pushed(self):
        """An explicit permissions block sets every scope it does not name to
        none, and a newly published GHCR package is private by default."""
        self.assertEqual(self.smoke["permissions"], {"packages": "read"})
        self.assertTrue(any("login-action" in str(step.get("uses", ""))
                            for step in self.smoke["steps"]),
                        "nothing logs in, so a private package cannot be pulled")

    def test_there_is_no_waiting_on_a_sleep(self):
        """`sleep 5` was shorter than the image's own import time, so even a
        working container might not have been listening yet. preflight probes
        in-process and needs no window at all."""
        for step in self.runs:
            with self.subTest(step=step.get("name")):
                self.assertNotIn("sleep", step["run"])


class TestNothingBuildsAnImageReferenceByHand(unittest.TestCase):
    """`IMAGE_NAME: ${{ github.repository }}` is `NickSanft/MisterFritz`, and
    an OCI repository name is lowercase-only — `path-component` is
    `alpha-numeric [separator alpha-numeric]*` where `alpha-numeric` is
    `[a-z0-9]+`, and the reference parser returns "repository name must be
    lowercase" rather than looking anything up.

    metadata-action lowercases the names it pushes, so the registry is right
    and every hand-built reference was wrong. Two jobs did it: the smoke test,
    which has never run, and the canary, which hands the result to `kubectl
    argo rollouts set image`. The second is the one that would have reached a
    cluster.

    The rule is about the shape, because the next reference somebody writes by
    hand will have the same defect: a script that mentions IMAGE_NAME must
    lowercase it.
    """

    @staticmethod
    def _run_steps():
        for path in WORKFLOWS:
            for job, definition in _jobs(path).items():
                for step in definition.get("steps") or []:
                    if "run" in step:
                        yield path.name, job, step

    def test_the_rule_has_something_to_check(self):
        self.assertGreater(len(list(self._run_steps())), 5)

    def test_a_script_that_names_the_repository_lowercases_it(self):
        for workflow, job, step in self._run_steps():
            script = step["run"]
            mentioned = "IMAGE_NAME" in script or "IMAGE_NAME" in str(step.get("env") or {})
            if not mentioned:
                continue
            with self.subTest(workflow=workflow, job=job, step=step.get("name")):
                self.assertIn(
                    "tr '[:upper:]'", script,
                    f"{workflow}:{job} builds a registry reference from "
                    "github.repository without lowercasing it, which is not a "
                    "parseable reference at all")

    def test_no_script_asks_for_a_tag_named_after_the_full_commit_sha(self):
        """`type=sha` publishes the SHORT sha. `sha-${{ github.sha }}` names a
        tag that has never existed and cannot."""
        for workflow, job, step in self._run_steps():
            with self.subTest(workflow=workflow, job=job):
                self.assertNotIn("sha-${{ github.sha }}", step["run"])

    def test_the_canary_deploys_what_the_smoke_test_booted(self):
        """Not the same version tag — the same digest. Otherwise the two
        jobs can disagree about which bytes they mean, and the one that
        reaches the cluster is the one nobody tested."""
        canary = _jobs(REPO / ".github/workflows/release.yml")["canary-deploy"]
        [deploy] = [step for step in canary["steps"]
                    if "kubectl" in str(step.get("run", ""))]
        self.assertIn("needs.build.outputs.image", str(deploy["env"]))
        self.assertIn('misterfritz="$IMAGE"', deploy["run"])


class TestTheImageIsBuiltAndBootedSomewhere(unittest.TestCase):
    """Batch F's actual finding: `ENTRYPOINT ["python", "main_discord.py"]` was
    never executed by any automation at all.

    ci.yml had no docker step, so the image was first built on a `v*` tag —
    where the build has failed every time, four for four. The local-sim canary
    the README calls a full pipeline simulation builds local-sim/Dockerfile.mock,
    thirteen lines with one dependency. Nothing ever ran the thing that ships.
    """

    CI = REPO / ".github/workflows/ci.yml"

    @classmethod
    def setUpClass(cls):
        cls.job = _jobs(cls.CI)["build-image"]

    def test_the_real_dockerfile_is_what_gets_built(self):
        [build] = [step for step in self.job["steps"]
                   if "build-push-action" in str(step.get("uses", ""))]
        self.assertEqual(build["with"]["context"], ".")
        self.assertNotIn("file", build["with"],
                         "a `file:` here would point at something other than "
                         "the Dockerfile the release builds")
        self.assertIs(build["with"]["push"], False)

    def test_it_boots_the_image_through_the_entrypoint(self):
        """The step that SUPPLIES configuration, specifically. Asserting over
        every run script joined together let the real boot be gutted while the
        no-configuration check below kept the word `--preflight` in the file —
        found by mutating this step's argument to `--version`."""
        [boot] = [step for step in self.job["steps"]
                  if "-e DISCORD_BOT_TOKEN" in str(step.get("run", ""))]
        self.assertIn("--preflight", boot["run"])
        self.assertNotIn("--entrypoint", boot["run"])

    def test_it_proves_the_image_carries_no_configuration(self):
        [negative] = [step for step in self.job["steps"]
                      if "if docker run" in str(step.get("run", ""))]
        self.assertNotIn("-e DISCORD_BOT_TOKEN", negative["run"])
        self.assertIn("exit 1", negative["run"])

    def test_it_is_not_on_the_push_path(self):
        """Measured, not assumed: the lock pulls CUDA wheels from PyPI, about
        4.5 GiB down and 8.5 GiB installed, plus a 464 MiB Whisper model. That
        is 15-30 minutes, which would dominate every push for a signal the
        constrained install in `test` gives in three minutes."""
        condition = self.job["if"]
        self.assertIn("schedule", condition)
        self.assertIn("workflow_dispatch", condition)

    def test_the_dockerfile_still_has_the_entrypoint_this_all_assumes(self):
        dockerfile = (REPO / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn('ENTRYPOINT ["python", "main_discord.py"]', dockerfile)


# Node ranges as package.json and pnpm-lock.yaml write them: `>=` and `^`
# clauses (`>= 0.4`, `>=v12.22.7` too) joined by `||`, which is every form the
# desktop lock holds. Anything else is refused rather than guessed at.
_CLAUSE = re.compile(r"(\^|>=)v?(\d+)(?:\.(\d+))?(?:\.(\d+))?")


def _version(text: str) -> tuple:
    return tuple(int(part) for part in (text.split(".") + ["0", "0"])[:3])


def _branches(spec: str) -> list:
    return [re.sub(r"(>=|\^)\s+", r"\1", branch.strip()) for branch in spec.split("||")]


def _floor(clause: str) -> tuple:
    match = _CLAUSE.fullmatch(clause)
    if not match:
        raise AssertionError(f"cannot read the Node range clause {clause!r}")
    return match.group(1), tuple(int(part or 0) for part in match.groups()[1:])


def _admits(spec: str, version: tuple) -> bool:
    for branch in _branches(spec):
        op, floor = _floor(branch)
        if version >= floor and (op == ">=" or version[0] == floor[0]):
            return True
    return False


def _edges(spec: str) -> set:
    """Where a range's stretches of Node begin (its floors) and where its `^`
    branches stop (the next major)."""
    edges = set()
    for branch in _branches(spec):
        op, floor = _floor(branch)
        edges.add(floor)
        if op == "^":
            edges.add((floor[0] + 1, 0, 0))
    return edges


def _promised_beyond(engines: str, locked: set) -> list:
    """Every (Node, locked range) where `engines` admits a Node the range does
    not. Each range is a union of stretches that begin at a floor and end at a
    `^` branch's next major, so the first Node of any gap is one of these
    edges: checking them checks every version."""
    edges = _edges(engines).union(*(_edges(spec) for spec in locked))
    return [(edge, spec) for edge in sorted(edges) if _admits(engines, edge)
            for spec in sorted(locked) if not _admits(spec, edge)]


def _locked_node_ranges(lock: Path) -> set:
    """pnpm writes each engines map on one line with its keys sorted, so bun,
    deno or iojs can come before node: node is read wherever it falls. The
    field is read at pnpm's package-field indent, four spaces; dependency
    names sit at six, so a package called `check-engines` is not one."""
    text = lock.read_text(encoding="utf-8")
    entries = re.findall(r"^    engines:", text, re.M)
    one_line = re.findall(r"^    engines: \{.*\}$", text, re.M)
    if len(entries) != len(one_line):
        raise AssertionError("an engines entry in the lock is not on one line")
    return set(re.findall(r"^    engines: \{(?:[^}]*?, )?node: '?([^',}]+?)'?[,}]",
                          text, re.M))


class TestTheDesktopJobBuildsWhatTheReadmeSays(unittest.TestCase):
    """desktop/README.md says how the shell is built and tested, and its
    Cargo.toml which Rust it builds with. The job is only worth its minutes if
    it does those things, so the parts that could drift apart are held
    together here: the toolchain against rust-version, every compiled build
    against custom-protocol (without which the shell is a dev build that loads
    a dev server), and the pages built before the Rust that embeds them.
    """

    CI = REPO / ".github/workflows/ci.yml"
    DESKTOP = REPO / "desktop"

    @classmethod
    def setUpClass(cls):
        cls.job = _jobs(cls.CI)["desktop"]
        cls.steps = cls.job["steps"]
        cls.runs = [str(step.get("run", "")) for step in cls.steps]

    def _first(self, needle: str) -> int:
        found = [index for index, run in enumerate(self.runs) if needle in run]
        self.assertTrue(found, f"no step runs {needle!r}")
        return found[0]

    def test_it_runs_on_windows_from_the_app_folder(self):
        self.assertTrue(self.job["runs-on"].startswith("windows-"))
        self.assertEqual(self.job["defaults"]["run"]["working-directory"], "desktop")

    def test_it_builds_with_the_rust_the_crate_declares(self):
        manifest = (self.DESKTOP / "src-tauri/Cargo.toml").read_text(encoding="utf-8")
        [declared] = re.findall(r'^rust-version\s*=\s*"([^"]+)"', manifest, re.M)
        self.assertEqual(self.job["env"]["RUSTUP_TOOLCHAIN"], declared,
                         "the job builds with another Rust than Cargo.toml's "
                         "rust-version, so the MSRV it promises goes untested")
        install = self.runs[self._first("rustup toolchain install")]
        self.assertIn("RUSTUP_TOOLCHAIN", install)

    def test_pnpm_is_the_version_the_app_pins(self):
        """Read from packageManager rather than restated, so the two cannot
        disagree."""
        [setup] = [step for step in self.steps
                   if "pnpm/action-setup" in str(step.get("uses", ""))]
        self.assertNotIn("version", setup.get("with") or {})
        self.assertEqual(setup["with"]["package_json_file"], "desktop/package.json")
        package = json.loads((self.DESKTOP / "package.json").read_text(encoding="utf-8"))
        self.assertTrue(package["packageManager"].startswith("pnpm@"))

    def test_node_is_one_every_locked_package_runs_on(self):
        """Against the lockfile, not only the app's own `engines`: that said
        >=20.19 while the vitest it locks runs on ^22.12 || ^24 || >=26, so a
        job on Node 20 would have held to it and run vitest on a Node it
        excludes. node-version "26" is read as 26.0.0, the lowest it can
        resolve to."""
        [setup] = [step for step in self.steps
                   if "actions/setup-node" in str(step.get("uses", ""))]
        wanted = str(setup["with"]["node-version"])
        self.assertRegex(wanted, r"^\d+(\.\d+){0,2}$")
        lowest = _version(wanted)
        locked = _locked_node_ranges(self.DESKTOP / "pnpm-lock.yaml")
        self.assertIn("^22.12.0 || ^24.0.0 || >=26.0.0", locked,
                      "vitest 5's range is gone from the lock: is the parse right?")
        engines = json.loads((self.DESKTOP / "package.json")
                             .read_text(encoding="utf-8"))["engines"]["node"]
        for spec in sorted(locked | {engines}):
            with self.subTest(range=spec):
                self.assertTrue(_admits(spec, lowest),
                                f"Node {wanted} is outside {spec}")

    def test_the_apps_engines_promise_only_what_the_lock_runs_on(self):
        """desktop/README.md repeats `engines`, so any Node it admits that a
        locked package excludes (a floor too low, or 23 and 25 inside
        `>=22.12`) is a promise the app cannot keep. Every Node it admits is
        checked, not only its floors."""
        engines = json.loads((self.DESKTOP / "package.json")
                             .read_text(encoding="utf-8"))["engines"]["node"]
        locked = _locked_node_ranges(self.DESKTOP / "pnpm-lock.yaml")
        self.assertEqual(_promised_beyond(engines, locked), [],
                         f"engines {engines!r} admits a Node a locked package excludes")

    def test_the_range_reader_reads_ranges(self):
        """So the two tests above cannot pass on a reader that admits
        everything."""
        self.assertTrue(_admits("^22.12.0 || ^24.0.0 || >=26.0.0", (24, 1, 0)))
        self.assertTrue(_admits(">= 0.4", (26, 0, 0)))
        self.assertTrue(_admits(">=v12.22.7", (12, 22, 7)))
        for outside in ((20, 19, 0), (22, 11, 9), (23, 0, 0), (25, 9, 0)):
            with self.subTest(version=outside):
                self.assertFalse(_admits("^22.12.0 || ^24.0.0 || >=26.0.0", outside))
        with self.assertRaises(AssertionError):
            _admits("<26", (24, 0, 0))
        vitest = {"^22.12.0 || ^24.0.0 || >=26.0.0"}
        self.assertEqual(_promised_beyond("^22.12.0 || ^24.0.0 || >=26.0.0", vitest), [])
        for over in (">=22.12.0", "^22.12.0 || >=24.0.0", ">=20.19"):
            with self.subTest(engines=over):
                self.assertTrue(_promised_beyond(over, vitest))

    def test_the_lock_reader_finds_node_wherever_pnpm_sorts_it(self):
        lock = "\n".join([
            "  a@1:",
            "    engines: {bun: '>=1', node: '>=18'}",
            "  b@1:",
            "    engines: {node: ^20.19.0 || >=22.12.0}",
            "  c@1:",
            "    engines: {node: '>=14', npm: '>=6', yarn: '>=1'}",
            "  d@1:",
            "    engines: {bun: '>=1'}",
            "    dependencies:",
            "      check-engines: 1.0.0",
        ])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "pnpm-lock.yaml"
            path.write_text(lock, encoding="utf-8")
            self.assertEqual(_locked_node_ranges(path),
                             {">=18", "^20.19.0 || >=22.12.0", ">=14"})
            path.write_text(lock + "\n    engines:\n      node: '>=8'\n", encoding="utf-8")
            with self.assertRaises(AssertionError):
                _locked_node_ranges(path)

    def test_the_lockfile_is_installed_as_it_is(self):
        self._first("pnpm install --frozen-lockfile")

    def test_every_cargo_run_keeps_to_the_lock(self):
        cargo = [run for run in self.runs if "cargo " in run]
        self.assertTrue(cargo)
        for run in cargo:
            with self.subTest(run=run):
                self.assertIn("--locked", run)

    def test_every_compiled_build_is_the_shipped_one(self):
        """`cargo check` only type-checks, and the README calls it safe without
        the feature; anything that compiles does not."""
        compiled = [run for run in self.runs
                    if re.search(r"\bcargo (build|test|run)\b", run)]
        self.assertTrue(compiled, "nothing compiles or tests the shell")
        for run in compiled:
            with self.subTest(run=run):
                self.assertIn("--features custom-protocol", run)

    def test_the_pages_are_tested_and_built_before_the_shell_embeds_them(self):
        tested = self._first("pnpm test:run")
        built = self._first("pnpm build")
        shell = self._first("cargo test")
        self.assertLess(tested, shell)
        self.assertLess(built, shell,
                        "the custom-protocol build embeds dist/, so the pages "
                        "must be built first")

    def test_it_checks_out_the_files_as_committed(self):
        """Windows images default to core.autocrlf=true, which would test CRLF
        copies of files committed, and tested, as LF."""
        autocrlf = self._first("core.autocrlf false")
        [checkout] = [index for index, step in enumerate(self.steps)
                      if "actions/checkout" in str(step.get("uses", ""))]
        self.assertLess(autocrlf, checkout)



if __name__ == "__main__":                    # pragma: no cover
    unittest.main()
