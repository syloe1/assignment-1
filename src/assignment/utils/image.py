"""Build a task's testbed image on Modal from a local checkout.

The repository under test is copied in from `task.source` — normally the
`chess_app/` submodule — rather than cloned inside the build. That keeps the
build offline, works with a private repository without any credential, and
avoids a network round trip on every rebuild.

The cost is that the image reflects whatever is on disk, so `verify_source`
checks the `chess_app` source before every build to confirm it matches the
source the assignment intends people to fix. A testbed silently built from an
already-fixed working tree would make a broken agent look like it passed.
"""

from __future__ import annotations

import logging
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

import modal

from assignment.local_sandbox import docker_build, sandbox_backend
from assignment.task import Task

logger = logging.getLogger(__name__)

ASSIGNMENT_DIR = "/opt/assignment"

# Local caches, credentials, and git metadata never belong in the testbed. The
# .git directory in particular is a gitlink file in a submodule checkout and
# would be meaningless inside the container; the build makes a fresh repository
# instead.
IGNORED_NAMES = {".git", ".venv", ".pytest_cache", "__pycache__", ".env", ".DS_Store"}

class SourceMismatch(Exception):
    """The local checkout is not the commit the task is defined against."""

def _git(source: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(source), *args], capture_output=True, text=True)

def is_ignored(path: Path) -> bool:
    """Whether a file should be kept out of the build context."""
    return any(part in IGNORED_NAMES for part in path.parts) or path.suffix == ".pyc"

def verify_source(task: Task, strict: bool = True) -> None:
    """Check that the local checkout matches the task's base commit.

    Args:
        task: The task whose `source` to check.
        strict: Raise on a mismatch. When False, mismatches are logged as
            warnings and the build proceeds — useful when deliberately testing
            a modified checkout.

    Raises:
        SourceMismatch: The checkout is missing, on the wrong commit, or dirty.
    """

    def complain(message: str) -> None:
        if strict:
            raise SourceMismatch(message)
        logger.warning("%s (continuing: strict=False)", message)

    if not task.source.is_dir():
        raise SourceMismatch(
            f"{task.source} does not exist. If it is a submodule, run `git submodule update --init`."
        )

    head = _git(task.source, "rev-parse", "HEAD")
    if head.returncode != 0:
        complain(f"{task.source} is not a git checkout: {head.stderr.strip()}")
        return

    if head.stdout.strip() != task.base_commit:
        complain(
            f"{task.source} is at {head.stdout.strip()[:12]}, but {task.id} is defined against "
            f"{task.base_commit[:12]}. The testbed would not be the task's base commit."
        )

    dirty = _git(task.source, "status", "--porcelain")
    if dirty.stdout.strip():
        # Each line is a status code then the path. The code's width varies, so
        # split on whitespace rather than slicing at a fixed offset.
        paths = [line.strip().split(None, 1)[-1] for line in dirty.stdout.strip().splitlines()[:5]]
        changed = ", ".join(paths)
        complain(
            f"{task.source} has uncommitted changes ({changed}). The testbed would contain them, "
            "which silently invalidates the evaluation if one of them is the fix."
        )

def build_testbed_image(
    task: Task,
    strict: bool = True,
    force_build: bool = False,
    assignment_files: tuple[Path, ...] = (),
) -> "modal.Image | str":
    """Build the image holding the repository under test at its base commit.

    Args:
        task: The task whose Dockerfile and source checkout to build from.
        strict: Refuse to build when the checkout does not match the task's
            base commit. See `verify_source`.
        force_build: Skip the builder's layer cache.
        assignment_files: Files to place in `/opt/assignment` inside the image.
            The chess sandbox needs these; a plain testbed does not.

    Returns:
        A Modal image, or a Docker tag when `SANDBOX_BACKEND=docker` is set.
        Either one is accepted by `assignment.env.Environment`.
    """
    verify_source(task, strict=strict)

    logger.info("Building %s from %s at %s", task.id, task.source, task.base_commit[:12])
    if sandbox_backend() == "docker":
        return build_testbed_docker_image(
            task, assignment_files=assignment_files, force_build=force_build
        )

    image = modal.Image.from_dockerfile(
        str(task.dockerfile),
        context_dir=str(task.source),
        force_build=force_build,
        ignore=is_ignored,
    )

    # Pin last, deliberately. Modal appends its own dependency install after the
    # Dockerfile's commands on image builder versions <= 2024.10, and the 2023.12
    # requirements file pins fastapi==0.88.0, which pulls starlette down to 0.22
    # and breaks every test using TestClient. Chaining here puts this layer on
    # top of that one, so the task's versions are the ones that survive.
    if task.pins:
        logger.info("Pinning %d packages on top of the built image", len(task.pins))
        image = image.pip_install(*task.pins)

    for source in assignment_files:
        image = image.add_local_file(
            str(source),
            f"{ASSIGNMENT_DIR}/{Path(source).name}",
            copy=True,  # SWE-ReX adds its runtime build layer afterwards.
        )

    return image


def _copy_source(source: Path, destination: Path) -> None:
    """Copy a checkout into a build context, minus caches and git metadata."""

    def ignore(directory: str, names: list[str]) -> set[str]:
        return {
            name
            for name in names
            if is_ignored(Path(directory) / name)
        }

    shutil.copytree(source, destination, ignore=ignore, symlinks=True)


def build_testbed_docker_image(
    task: Task,
    assignment_files: tuple[Path, ...] = (),
    force_build: bool = False,
) -> str:
    """Build the testbed with the local Docker daemon and return its tag.

    Two builds, because the task's Dockerfile cannot be amended: the second one
    starts `FROM` the first and adds what has to come afterwards. The build
    context is a temporary copy of the checkout rather than the checkout
    itself, so a `.dockerignore` never has to be written into a submodule this
    assignment forbids touching.

    Args:
        task: The task whose Dockerfile and source checkout to build from.
        assignment_files: Files to place in `/opt/assignment` inside the image.
        force_build: Skip the layer cache.

    Returns:
        The tag of the finished image.
    """
    digest = task.base_commit[:12]
    final_tag = f"assignment-{task.id}:{digest}"
    if assignment_files:
        # Otherwise two callers sharing a tag would silently get whichever
        # image was built first.
        final_tag += "-chess"

    # Nothing to add on top, so the task's Dockerfile is the whole story and
    # the build can be tagged as final directly.
    if not task.pins and not assignment_files:
        with tempfile.TemporaryDirectory(prefix="assignment-build-") as context_dir:
            context = Path(context_dir)
            _copy_source(task.source, context / "checkout")
            # The Dockerfile sits beside the context, not in it, so its own
            # `COPY . /testbed` cannot pull a stray file into the image.
            docker_build(
                context / "checkout",
                final_tag,
                dockerfile=task.dockerfile,
                force=force_build,
            )
        return final_tag

    base_tag = f"{final_tag}-base"
    with tempfile.TemporaryDirectory(prefix="assignment-build-") as context_dir:
        context = Path(context_dir)
        _copy_source(task.source, context / "checkout")
        docker_build(
            context / "checkout",
            base_tag,
            dockerfile=task.dockerfile,
            force=force_build,
        )

    additions = [f"FROM {base_tag}"]
    if task.pins:
        logger.info("Pinning %d packages on top of the built image", len(task.pins))
        additions.append(
            "RUN pip install --no-cache-dir "
            + " ".join(shlex.quote(pin) for pin in task.pins)
        )
    if assignment_files:
        additions += [
            f"RUN mkdir -p {ASSIGNMENT_DIR}",
            f"COPY files/ {ASSIGNMENT_DIR}/",
        ]

    with tempfile.TemporaryDirectory(prefix="assignment-layer-") as layer_dir:
        layer = Path(layer_dir)
        (layer / "Dockerfile").write_text("\n".join(additions) + "\n")
        if assignment_files:
            (layer / "files").mkdir()
            for source in assignment_files:
                shutil.copy(source, layer / "files" / Path(source).name)
        docker_build(layer, final_tag, force=force_build)

    return final_tag
