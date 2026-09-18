"""A local Docker sandbox, standing in for Modal.

The assignment hosts every testbed in a Modal sandbox reached through SWE-ReX.
That needs a Modal account and credits, which not every reader has. This module
provides the same capability with nothing but a local Docker daemon.

It deliberately does not go through SWE-ReX. SWE-ReX's own Docker deployment
bootstraps a runtime *inside* the container, so the image must be able to reach
a package index and install Python packages. The images used here cannot always
do that: `python:3.12-slim` has no runtime preinstalled, and the SWE-bench
images install their repository into a conda environment that the bootstrap
would not see. Talking to the container with `docker exec` instead puts no
requirements on the image at all.

What this does *not* provide is Modal's server-side reclamation. A container
launched here keeps running if the process that started it dies, so
`container_name` is reported at startup and `stop()` is the only thing that
ends it.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Iterable, Sequence

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

BACKEND_ENV = "SANDBOX_BACKEND"
DEFAULT_BACKEND = "modal"
BACKENDS = ("modal", "docker")

# Modal's `Sandbox.create` argument names, reused so callers can keep passing
# the same kwargs to either backend.
MODAL_ENCRYPTED_PORTS = "encrypted_ports"


def sandbox_backend() -> str:
    """Which sandbox backend this process should use.

    Reads `SANDBOX_BACKEND`, defaulting to `modal` so a checkout without a
    `.env` behaves exactly as the starter does.

    Raises:
        ValueError: If the variable names a backend that does not exist, rather
            than silently falling back to Modal and failing later with a
            credential error that says nothing about the real mistake.
    """
    value = os.environ.get(BACKEND_ENV, DEFAULT_BACKEND).strip().lower()
    if value not in BACKENDS:
        raise ValueError(
            f"{BACKEND_ENV}={value!r} is not one of: {', '.join(BACKENDS)}"
        )
    return value


class DockerUnavailable(RuntimeError):
    """The docker CLI is missing or its daemon is not answering."""


def _docker_argv(*args: str) -> list[str]:
    return ["docker", *args]


def _run(
    argv: Sequence[str],
    timeout: float | None = None,
) -> subprocess.CompletedProcess:
    """Run a docker command, capturing both streams as text.

    Returns:
        The completed process. A non-zero exit is not raised, because the
        caller reports the command's own failure to the model as an observation.
    """
    return subprocess.run(
        list(argv),
        capture_output=True,
        text=True,
        errors="replace",
        timeout=timeout,
    )


def require_docker() -> None:
    """Fail early, and with a fixable message, when docker is unusable."""
    if shutil.which("docker") is None:
        raise DockerUnavailable(
            "The docker CLI was not found on PATH. Install Docker and make sure "
            "`docker --version` works, or set "
            f"{BACKEND_ENV}=modal to use Modal instead."
        )
    probe = _run(_docker_argv("info"), timeout=30)
    if probe.returncode != 0:
        raise DockerUnavailable(
            "The docker CLI is installed but the daemon is not answering "
            f"(`docker info` exited {probe.returncode}): "
            f"{(probe.stderr or probe.stdout).strip()}"
        )


def published_ports(modal_sandbox_kwargs: dict[str, Any] | None) -> list[int]:
    """Ports a caller asked Modal to forward, for the Docker backend to publish.

    Lets `ChessSandbox`, which describes its port with Modal's own kwarg, work
    under either backend without branching at the call site.
    """
    if not modal_sandbox_kwargs:
        return []
    ports = modal_sandbox_kwargs.get(MODAL_ENCRYPTED_PORTS) or []
    return [int(port) for port in ports]


class DockerSandbox:
    """A long-lived container that commands are executed in.

    Mimics the small part of the sandbox surface `assignment.env.Environment`
    relies on: `execute`, `is_alive`, `stop`, and `tunnel_url`.
    """

    def __init__(
        self,
        image: str,
        cwd: str = "/",
        ports: Iterable[int] = (),
        deployment_timeout: float | None = None,
        env_defaults: dict[str, str] | None = None,
        name: str | None = None,
    ):
        """Start a detached container from `image`.

        Args:
            image: A local tag or a registry reference, pulled if absent.
            cwd: Working directory for commands that do not name one.
            ports: Container ports to publish on the same host port, bound to
                loopback only. Publishing 8000 makes it reachable at
                `http://127.0.0.1:8000`, which is what Modal's tunnel URL is
                replaced by.
            deployment_timeout: How long a Modal sandbox would live. Recorded
                and logged, not enforced; see the module docstring.
            env_defaults: Environment merged into every command. Held by
                reference, so the `Environment` that owns it can keep adding to
                it (as `activate_conda_env` does).
            name: Container name. Generated when not given.

        Raises:
            DockerUnavailable: If docker is missing or its daemon is down.
            RuntimeError: If the container does not come up.
        """
        require_docker()
        self.image = image
        self.cwd = cwd
        self.ports = list(dict.fromkeys(int(port) for port in ports))
        self.deployment_timeout = deployment_timeout
        self.env_defaults: dict[str, str] = (
            env_defaults if env_defaults is not None else {}
        )
        self.container = name or f"assignment-{uuid.uuid4().hex[:10]}"
        self._shell: str | None = None
        self._start()

    # -- Lifecycle ----------------------------------------------------------

    def _start(self) -> None:
        # The image's own ENTRYPOINT is overridden so a task image that ships
        # one (a server, a shell wrapper) cannot decide how the sandbox idles.
        argv = _docker_argv(
            "run",
            "-d",
            "--name",
            self.container,
            "--entrypoint",
            "/bin/sh",
            "-w",
            self.cwd,
        )
        for port in self.ports:
            argv += ["-p", f"127.0.0.1:{port}:{port}"]
        argv += [self.image, "-c", "sleep infinity"]

        result = _run(argv, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(
                f"Could not start container from {self.image!r}: "
                f"{(result.stderr or result.stdout).strip()}"
            )

        logger.info("Container %s running (%s)", self.container, self.image)
        if self.deployment_timeout is not None:
            logger.info(
                "Container %s will not reclaim itself; stop it with "
                "`docker rm -f %s` if this process dies",
                self.container,
                self.container,
            )

        self._shell = self._detect_shell()

    def _detect_shell(self) -> str:
        """Pick the shell commands run under, preferring bash where it exists.

        The `execute` tool documents bash idioms (`cat <<'EOF'`, `sed -i`) and
        the harness runs `pytest` and `git apply`, so a POSIX `sh` fallback is
        kept for images that genuinely lack bash rather than assumed away.
        """
        for candidate in ("/bin/bash", "/bin/sh"):
            probe = _run(
                _docker_argv("exec", self.container, "test", "-x", candidate),
                timeout=60,
            )
            if probe.returncode == 0:
                return candidate
        raise RuntimeError(f"Container {self.container} has neither bash nor sh")

    def is_alive(self) -> bool:
        """Whether the container is still running."""
        probe = _run(
            _docker_argv(
                "inspect", "-f", "{{.State.Running}}", self.container
            ),
            timeout=60,
        )
        return probe.returncode == 0 and probe.stdout.strip() == "true"

    def stop(self, timeout: float = 10) -> None:
        """Force-remove the container. Safe to call more than once."""
        _run(_docker_argv("rm", "-f", self.container), timeout=timeout)
        logger.info("Container %s removed", self.container)

    def tunnel_url(self, port: int) -> str:
        """The host URL a published port is reachable at.

        Raises:
            ValueError: If the port was not published, matching the Modal
                backend's behaviour so callers need no special case.
        """
        if port not in self.ports:
            forwarded = ", ".join(str(item) for item in self.ports) or "none"
            raise ValueError(
                f"Port {port} was not forwarded. Available forwarded ports: {forwarded}."
            )
        return f"http://127.0.0.1:{port}"

    # -- Execution ----------------------------------------------------------

    def execute(
        self,
        command: str | list[str],
        timeout: float | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        shell: bool | None = True,
        check: bool = False,
    ) -> dict:
        """Run a command in the container.

        Mirrors `assignment.env.Environment.execute`'s contract exactly,
        including that a failed command is reported rather than raised.

        Args:
            command: A shell string, or an argv list when `shell` is False.
            timeout: Seconds before the `docker exec` client is killed. The
                command itself is not signalled, since the container holds it
                and there is no handle to it from here.
            cwd: Working directory, defaulting to the sandbox's.
            env: Extra environment variables for this command.
            shell: Run through a shell. None is treated as True, because a model
                calling this as a tool may send null for an omitted argument.
            check: Raise on a non-zero exit instead of reporting it.
        """
        shell = True if shell is None else shell
        workdir = cwd or self.cwd
        merged_env = {**self.env_defaults, **(env or {})}

        argv = _docker_argv("exec", "-w", workdir)
        for key, value in merged_env.items():
            argv += ["-e", f"{key}={value}"]
        argv.append(self.container)
        if shell:
            assert self._shell is not None, "the container is not running"
            argv += [self._shell, "-c", command]
        else:
            # An argv list is what shell=False means; a bare string would be a
            # single word, which is almost never what the caller intended.
            argv += list(command) if isinstance(command, list) else [command]

        try:
            completed = _run(argv, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            if check:
                raise
            return self._failure(f"Command timed out after {timeout}s", exc)
        except OSError as exc:
            if check:
                raise
            return self._failure(f"Could not run docker: {exc}", exc)

        # A command that merely failed is a fact worth reporting to the model.
        # A container that is gone is terminal: every later command fails the
        # same way, so raise instead of letting the caller keep going, matching
        # what the Modal backend does when its sandbox dies.
        if completed.returncode != 0 and not self.is_alive():
            raise RuntimeError(
                "The sandbox is no longer running, so no further commands can "
                f"be executed. Last error: {(completed.stderr or completed.stdout).strip()}"
            )

        output = {
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            # Same shape as the Modal backend: one merged string for the agent,
            # with the streams kept separate for the evaluation harness.
            "output": completed.stdout,
            "returncode": completed.returncode,
            "exception_info": "",
        }
        if completed.stderr:
            output["output"] += completed.stderr

        if check and completed.returncode != 0:
            raise subprocess.CalledProcessError(
                completed.returncode,
                command,
                output=completed.stdout,
                stderr=completed.stderr,
            )
        return output

    @staticmethod
    def _failure(message: str, exc: BaseException) -> dict:
        """A failed call, in the same shape as a command that simply exited."""
        return {
            "stdout": "",
            "stderr": message,
            "output": message,
            "returncode": -1,
            "exception_info": f"An error occurred while executing the command: {exc}",
            "extra": {"exception_type": type(exc).__name__, "exception": str(exc)},
        }


def docker_build(
    context: str | Path,
    tag: str,
    dockerfile: str | Path | None = None,
    force: bool = False,
) -> str:
    """Build an image, streaming the builder's output.

    Args:
        context: The build context directory.
        tag: Tag for the resulting image.
        dockerfile: Path to the Dockerfile. May live outside the context, which
            is what keeps the Dockerfile itself from being copied into the
            image by a `COPY . /testbed`.
        force: Skip the layer cache.

    Returns:
        The tag.

    Raises:
        RuntimeError: If the build fails.
    """
    require_docker()
    argv = _docker_argv("build", "-t", tag)
    if dockerfile is not None:
        argv += ["-f", str(dockerfile)]
    if force:
        argv.append("--no-cache")
    argv.append(str(context))

    logger.info("Building %s", tag)
    # Not captured: a first build pulls a base image and installs packages, and
    # minutes of silence looks like a hang.
    completed = subprocess.run(argv)
    if completed.returncode != 0:
        raise RuntimeError(f"docker build failed for {tag} (exit {completed.returncode})")
    return tag
