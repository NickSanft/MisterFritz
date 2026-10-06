"""Both deployments must persist the state the bot creates.

Neither did. Compose mounted `./chat_history.db` - a path nothing in this
codebase ever opens, and one that Docker turns into a *directory* on the host
because the file does not exist - while `/app/fritz.db` sat on the container's
writable layer, so `docker compose down`, a rebuild or any image update
destroyed the whole conversation history, every schedule, the relay's messages
AND its block list, and the identity aliases. Kubernetes had the same hole in a
different shape: a PVC with subPath mounts for input, output and chroma_store,
but `CHAT_DB_NAME` pointing at `/app/chat_history.db`, which is on no mount,
and `DB_NAME` left at its working-directory default.

The relay blocks are what make this more than an inconvenience. DECISIONS.md
#22 keeps a block out of `/forget all` so that a privacy command can never be a
safety regression - and a container restart did exactly what that decision
forbade, with no command run and no audit line.

So these tests read the manifests and check them against what the code actually
resolves its paths to, under each deployment's own environment. The last class
is the one that matters most over time: it discovers configurable paths from the
source, so a new one has to be classified here before the suite goes green.
"""
import ast
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Every configurable path the bot writes to, and the module that resolves it.
# TestTheInventoryIsComplete below fails if a new one appears and is not here.
STATE_PATHS = {
    "DB_NAME": "fritz_utils",            # checkpoints, schedules, relay, aliases
    "CHAT_DB_NAME": "fritz_utils",       # defaults to DB_NAME
    "SCHEDULE_DB": "fritz_utils",        # defaults to DB_NAME
    "CHROMA_DB_PATH": "fritz_utils",     # memories and the document index
    "INDEXED_FILES_PATH": "fritz_utils",  # manifest, inside CHROMA_DB_PATH
    "WORKSPACES_ROOT": "fritz_utils",    # per-user sandboxes
    "DOC_FOLDER": "fritz_utils",         # the watched folder
    "AUDIT_LOG_PATH": "observability",   # /forget and /export events
}

# Paths the bot writes that are NOT configurable, so they cannot be discovered
# by reading env defaults. Both are a bare relative "output" resolved against
# the working directory: image_generator.py:9 and tts.py:27.
UNCONFIGURABLE = {"output"}

# Constants that hold a path but are not state: these name an executable to
# RUN, and their "./ffmpeg.exe" fallback is a bundled binary, not something the
# bot writes. The Dockerfile installs ffmpeg via apt and sets both to the
# system binary.
NOT_STATE = {"FFMPEG_PATH", "FFPROBE_PATH"}

# State that is deliberately NOT mounted, with the reason.
#
#   .chat_cookie_secret - generated when CHAT_COOKIE_SECRET is unset. Mounting
#   it would mean binding a single file that does not exist yet, which is the
#   trap this whole test file exists to prevent. For a container the answer is
#   to set the secret explicitly; .env.example says so.


def _resolved(env: dict) -> dict:
    """What the path constants come out as under `env`.

    In a subprocess on purpose: these are module-level constants, so reading
    them under a different environment means importing the modules again, and
    reloading fritz_utils in-process would hand every other test module a
    different object than the one it imported.
    """
    program = (
        "import json, fritz_utils, observability;"
        "print(json.dumps({"
        + ", ".join(f"{name!r}: {module}.{name}" for name, module in STATE_PATHS.items())
        + "}))"
    )
    # conftest.py points every one of these at a temp sandbox so the suite
    # never touches real data, and a subprocess inherits that. Anything the
    # deployment does not set has to be UNSET here, or the test reads the test
    # environment's paths and passes whatever the manifest says.
    inherited = {k: v for k, v in os.environ.items()
                 if k not in STATE_PATHS or k in env}
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=str(REPO), env={**inherited, **env},
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:                      # pragma: no cover
        raise AssertionError(f"could not resolve paths: {result.stderr[-2000:]}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def _posix(path: str) -> str:
    return path.replace("\\", "/").rstrip("/")


def _is_under(path: str, mount: str) -> bool:
    """Is `path` the mount itself, or inside it?

    Container paths, so posix semantics regardless of the host running the
    tests. A relative path is resolved against /app, which is the WORKDIR.
    """
    path, mount = _posix(path), _posix(mount)
    if not path.startswith("/"):
        path = "/app/" + path.lstrip("./")
    return path == mount or path.startswith(mount + "/")


class Deployment:
    """The two things a deployment says: where state goes, and what is mounted."""

    def __init__(self, env: dict, mounts: list):
        self.env = env
        self.mounts = mounts

    @property
    def paths(self) -> dict:
        return _resolved(self.env)


def compose() -> Deployment:
    spec = yaml.safe_load((REPO / "docker-compose.yml").read_text(encoding="utf-8"))
    service = spec["services"]["misterfritz"]
    env = {}
    for entry in service.get("environment") or []:
        name, _, value = str(entry).partition("=")
        if "${" not in value:                       # skip compose interpolation
            env[name] = value
    mounts = [str(v).split(":")[1] for v in service.get("volumes") or []]
    return Deployment(env, mounts)


def kubernetes() -> Deployment:
    configmap = yaml.safe_load((REPO / "infra/k8s/configmap.yaml").read_text(encoding="utf-8"))
    env = {k: str(v) for k, v in configmap["data"].items()}
    mounts = []
    for document in yaml.safe_load_all(
            (REPO / "infra/k8s/deployment.yaml").read_text(encoding="utf-8")):
        if not document or document.get("kind") != "Deployment":
            continue
        for container in document["spec"]["template"]["spec"]["containers"]:
            for mount in container.get("volumeMounts") or []:
                mounts.append(mount["mountPath"])
    return Deployment(env, mounts)


class DeploymentCase(unittest.TestCase):
    """Resolved once per class: each call is a subprocess import."""

    @classmethod
    def setUpClass(cls):
        cls.compose = compose()
        cls.kubernetes = kubernetes()
        cls.resolved = {"compose": cls.compose.paths,
                        "kubernetes": cls.kubernetes.paths}

    def deployments(self):
        return (("compose", self.compose), ("kubernetes", self.kubernetes))


class TestEveryStatePathSurvivesARestart(DeploymentCase):
    def test_nothing_the_bot_writes_lands_on_the_ephemeral_layer(self):
        for label, deployment in self.deployments():
            for name, path in self.resolved[label].items():
                with self.subTest(deployment=label, setting=name, path=path):
                    self.assertTrue(
                        any(_is_under(path, mount) for mount in deployment.mounts),
                        f"{label}: {name}={path} is on no mount, so it is "
                        f"destroyed on every restart. Mounts: {deployment.mounts}",
                    )

    def test_the_database_is_one_file_in_both_deployments(self):
        """CHAT_DB_NAME and SCHEDULE_DB default to DB_NAME. Overriding one of
        them separately - which the k8s configmap used to do - splits the state
        across files and is how half of it ended up unmounted."""
        for label in ("compose", "kubernetes"):
            with self.subTest(deployment=label):
                paths = self.resolved[label]
                self.assertEqual(paths["DB_NAME"], paths["CHAT_DB_NAME"])
                self.assertEqual(paths["DB_NAME"], paths["SCHEDULE_DB"])

    def test_the_databases_are_not_left_at_their_native_defaults(self):
        """A relative default resolves against the WORKDIR, which is exactly
        the layer that does not survive. Catching it by value rather than by
        mount means it fails even if someone mounts /app itself."""
        for label in ("compose", "kubernetes"):
            with self.subTest(deployment=label):
                self.assertTrue(
                    self.resolved[label]["DB_NAME"].startswith("/"),
                    "the database path is relative, so it follows the working "
                    "directory rather than the volume",
                )


class TestTheMountsAreTheOnesTheAppUses(DeploymentCase):
    def test_no_mount_names_a_path_nothing_writes(self):
        """`./chat_history.db:/app/chat_history.db` was mounted for a path no
        module opens, which is worse than a missing mount: it reads as though
        the database were persisted."""
        for label, deployment in self.deployments():
            paths = list(self.resolved[label].values()) + sorted(UNCONFIGURABLE)
            for mount in deployment.mounts:
                with self.subTest(deployment=label, mount=mount):
                    self.assertTrue(
                        any(_is_under(path, mount) for path in paths),
                        f"{label}: {mount} is mounted but nothing the bot "
                        "writes resolves under it",
                    )

    def test_compose_binds_directories_only(self):
        """Docker creates a missing bind source, and for a path that looks like
        a file it creates a DIRECTORY - which is how the old chat_history.db
        bind produced one on every first `up`. Binding directories sidesteps
        the whole question."""
        for volume in yaml.safe_load(
                (REPO / "docker-compose.yml").read_text(encoding="utf-8")
        )["services"]["misterfritz"]["volumes"]:
            host = str(volume).split(":")[0]
            with self.subTest(volume=volume):
                self.assertNotIn(".", Path(host).name,
                                 f"{volume} binds something that looks like a "
                                 "file; bind its directory instead")


class TestTheInventoryIsComplete(unittest.TestCase):
    """The guard that outlives this change.

    STATE_PATHS is hand-written, so it can go stale the moment someone adds a
    path. This finds the candidates in the source instead: a module-level
    constant whose default value is a relative path or names a .db or .log
    file. A new one fails here until it is classified - and classifying it
    means deciding whether it has to survive a restart.
    """

    @staticmethod
    def _candidates(module: str) -> set:
        tree = ast.parse((REPO / f"{module}.py").read_text(encoding="utf-8"))
        found = set()
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [t.id for t in targets if isinstance(t, ast.Name)]
            if not names or names[0].startswith("_"):
                continue
            literals = [n.value for n in ast.walk(node)
                        if isinstance(n, ast.Constant) and isinstance(n.value, str)]
            for value in literals:
                looks_like_a_path = (
                    value.startswith("./")
                    or value.endswith(".db")
                    or value.endswith(".log")
                    or value.endswith(".txt")
                )
                if looks_like_a_path:
                    found.add(names[0])
                    break
        return found

    def test_no_configurable_path_is_missing_from_the_inventory(self):
        for module in ("fritz_utils", "observability"):
            with self.subTest(module=module):
                missing = self._candidates(module) - set(STATE_PATHS) - NOT_STATE
                self.assertEqual(
                    missing, set(),
                    f"{module} resolves {sorted(missing)} to a path. Add each to "
                    "STATE_PATHS so the deployment tests cover it, or, if it "
                    "does not need to survive a restart, say why beside "
                    "UNCONFIGURABLE.",
                )

    def test_the_inventory_names_things_that_exist(self):
        """The other direction: a renamed constant must not leave a dead entry
        quietly passing every test above."""
        for name, module in STATE_PATHS.items():
            with self.subTest(setting=name):
                self.assertIn(name, self._candidates(module) | {"CHAT_DB_NAME",
                                                                "SCHEDULE_DB",
                                                                "INDEXED_FILES_PATH"},
                              f"{module}.{name} is in STATE_PATHS but no longer "
                              "looks like a path constant there")


if __name__ == "__main__":                    # pragma: no cover
    unittest.main()
