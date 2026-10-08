"""Local APPS execution in short-lived, resource-limited Docker containers."""

from __future__ import annotations

import asyncio
import json
import math
import tempfile
import uuid
from pathlib import Path

from .core import AppsProblem, ExecutionGrade

_CALL_RUNNER = """\
import json
import runpy
import sys

arguments = json.load(sys.stdin)
namespace = runpy.run_path('/task/solution.py', run_name='apps_solution')
name = sys.argv[1]
function = namespace.get(name)
if function is None:
    function = getattr(namespace['Solution'](), name)
result = function(*arguments)
json.dump(result, sys.stdout, allow_nan=False)
"""


class _OutputLimitError(RuntimeError):
    pass


class DockerAppsEvaluator:
    def __init__(
        self,
        *,
        image: str = "python:3.13-slim",
        timeout_seconds: float = 10,
        max_output_bytes: int = 65536,
        temp_root: Path | None = None,
    ) -> None:
        if (
            not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
            or max_output_bytes < 1
        ):
            raise ValueError("positive timeout and output limit required")
        self.image = image
        self.timeout_seconds = timeout_seconds
        self.max_output_bytes = max_output_bytes
        self.temp_root = temp_root

    def _command(self, name: str, directory: Path, fn_name: str | None) -> list[str]:
        command = [
            "docker",
            "run",
            "--rm",
            "--pull=never",
            "--name",
            name,
            "--network",
            "none",
            "--memory",
            "256m",
            "--memory-swap",
            "256m",
            "--cpus",
            "1",
            "--pids-limit",
            "64",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            "65534:65534",
            "--log-driver",
            "none",
            "--tmpfs",
            "/tmp:rw,nosuid,noexec,size=16m,mode=1777",
            "--mount",
            f"type=bind,src={directory},dst=/task,readonly",
            "--interactive",
            self.image,
            "python",
            "-I",
        ]
        if fn_name is None:
            return command + ["/task/solution.py"]
        return command + ["/task/call_runner.py", fn_name]

    async def _read_limited(self, stream: asyncio.StreamReader) -> bytes:
        chunks = []
        total = 0
        while chunk := await stream.read(4096):
            total += len(chunk)
            if total > self.max_output_bytes:
                raise _OutputLimitError("program output limit exceeded")
            chunks.append(chunk)
        return b"".join(chunks)

    @staticmethod
    async def _remove_container(name: str) -> None:
        cleanup = await asyncio.create_subprocess_exec(
            "docker",
            "rm",
            "--force",
            name,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(cleanup.communicate(), timeout=5)
        except TimeoutError:
            cleanup.kill()
            await cleanup.wait()
            raise RuntimeError("Docker container cleanup timed out") from None
        if cleanup.returncode and b"No such container" not in stderr:
            raise RuntimeError("Docker container cleanup failed")

    async def _run(
        self, command: list[str], name: str, input_text: str
    ) -> tuple[int, bytes, bytes]:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdin, stdout_stream, stderr_stream = (
            process.stdin,
            process.stdout,
            process.stderr,
        )
        assert stdin and stdout_stream and stderr_stream

        async def feed() -> None:
            try:
                stdin.write(input_text.encode())
                await stdin.drain()
                stdin.close()
            except (BrokenPipeError, ConnectionResetError):
                pass

        tasks = [
            asyncio.create_task(self._read_limited(stdout_stream)),
            asyncio.create_task(self._read_limited(stderr_stream)),
            asyncio.create_task(feed()),
        ]
        try:
            stdout, stderr, _ = await asyncio.wait_for(
                asyncio.gather(*tasks), timeout=self.timeout_seconds
            )
            returncode = await asyncio.wait_for(process.wait(), timeout=2)
            return returncode, stdout, stderr
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            try:
                # Killing only the Docker CLI would leave the program running.
                await self._remove_container(name)
            finally:
                if process.returncode is None:
                    process.kill()
                await process.wait()

    async def grade(self, problem: AppsProblem, code: str) -> ExecutionGrade:
        try:
            with tempfile.TemporaryDirectory(dir=self.temp_root) as temp:
                directory = Path(temp)
                directory.chmod(0o755)  # The container runs as an unprivileged UID.
                solution = directory / "solution.py"
                solution.write_text(code)
                solution.chmod(0o644)
                if problem.fn_name:
                    runner = directory / "call_runner.py"
                    runner.write_text(_CALL_RUNNER)
                    runner.chmod(0o644)
                for index, (input_value, expected) in enumerate(
                    zip(problem.inputs, problem.outputs, strict=True)
                ):
                    name = f"apps-grouped-rl-{uuid.uuid4().hex}"
                    command = self._command(name, directory, problem.fn_name)
                    input_text = (
                        json.dumps(input_value, allow_nan=False)
                        if problem.fn_name
                        else input_value
                    )
                    try:
                        returncode, stdout, stderr = await self._run(
                            command, name, input_text
                        )
                    except (TimeoutError, _OutputLimitError) as exc:
                        return ExecutionGrade(
                            False, f"test {index}: {type(exc).__name__}"
                        )
                    if (
                        stderr.startswith(b"docker:")
                        or b"Cannot connect to the Docker daemon" in stderr
                    ):
                        return ExecutionGrade(
                            False, infrastructure_error=stderr.decode(errors="replace")
                        )
                    if returncode:
                        return ExecutionGrade(
                            False, f"test {index}: program exited {returncode}"
                        )
                    if problem.fn_name:
                        try:
                            actual = json.loads(stdout)
                        except (ValueError, UnicodeDecodeError):
                            return ExecutionGrade(
                                False, f"test {index}: invalid function result"
                            )
                        matches = actual == expected or (
                            isinstance(expected, list)
                            and len(expected) == 1
                            and actual == expected[0]
                        )
                    else:
                        try:
                            actual_text = stdout.decode()
                        except UnicodeDecodeError:
                            return ExecutionGrade(
                                False, f"test {index}: invalid UTF-8 output"
                            )
                        matches = actual_text.strip() == expected.strip()
                    if not matches:
                        return ExecutionGrade(False, f"test {index}: wrong answer")
                return ExecutionGrade(True, "all APPS tests passed")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return ExecutionGrade(
                False, infrastructure_error=f"sandbox infrastructure failure: {exc}"
            )
