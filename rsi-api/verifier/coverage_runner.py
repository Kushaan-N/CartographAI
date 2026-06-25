"""
Branch coverage instrumentation harness.
INVARIANT: run_client() must return in under 200ms.
Tested in tests/test_coverage.py.
"""
import subprocess
import sys
import tempfile
import os
import time
import json
import shutil
import socket
import signal
from dataclasses import dataclass


@dataclass
class CoverageResult:
    branch_coverage: float      # 0.0-1.0
    branches_hit: int
    total_branches: int
    execution_time_ms: float    # must be < 200


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class InstrumentedApp:
    def __init__(self, source_code: str, port: int):
        self.tmpdir = tempfile.mkdtemp()
        self.source_path = os.path.join(self.tmpdir, "flask_app.py")
        self.data_file = os.path.join(self.tmpdir, ".coverage")
        self.json_file = os.path.join(self.tmpdir, "coverage.json")

        with open(self.source_path, "w") as f:
            f.write(source_code)

        # Use a fresh port so it doesn't conflict with the original RunningAPI
        self.port = _find_free_port()
        self.url = f"http://127.0.0.1:{self.port}"

        self.process = subprocess.Popen(
            [
                sys.executable, "-m", "coverage", "run",
                "--branch",
                f"--data-file={self.data_file}",
                self.source_path,
            ],
            env={**os.environ, "PORT": str(self.port)},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Wait for the instrumented app to boot before any client runs.
        self.ready = self._await_health(timeout=5.0)

    def _await_health(self, timeout: float) -> bool:
        """Poll /health until the instrumented app responds. Returns readiness.
        Tracked so run_client can extend the wait under load rather than running
        the client against a not-yet-listening app (which would yield 0% coverage
        from a boot race, not from the client's behaviour)."""
        import requests as req
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.process.poll() is not None:
                return False  # process died during boot
            try:
                if req.get(f"{self.url}/health", timeout=0.5).status_code == 200:
                    return True
            except Exception:
                pass
            time.sleep(0.1)
        return False

    def run_client(self, client_code: str) -> CoverageResult:
        t0 = time.monotonic()

        # Guard against a slow boot under load: if the app wasn't ready at
        # construction, give it one more bounded chance before running the client.
        if not getattr(self, "ready", False):
            self.ready = self._await_health(timeout=3.0)

        client_tmp = tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w")
        client_tmp.write(client_code)
        client_tmp.close()

        try:
            subprocess.run(
                [sys.executable, client_tmp.name],
                timeout=5,
                capture_output=True,
                env={**os.environ, "API_BASE_URL": self.url},
            )
        except subprocess.TimeoutExpired:
            pass
        except Exception:
            pass
        finally:
            try:
                os.unlink(client_tmp.name)
            except Exception:
                pass

        # Stop the Flask app so coverage data is flushed.
        # IMPORTANT: use SIGINT, not SIGTERM. coverage.py writes its data file
        # in an atexit handler, which runs on SIGINT (raised as KeyboardInterrupt
        # in the werkzeug server) but NOT on SIGTERM. Sending SIGTERM here causes
        # the .coverage file to never be written -> num_branches=0 -> 0% coverage
        # on every episode, zeroing all GRPO advantages.
        self.process.send_signal(signal.SIGINT)
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()

        elapsed_ms = (time.monotonic() - t0) * 1000

        # Parse coverage data
        branch_coverage = 0.0
        branches_hit = 0
        total_branches = 0
        try:
            result = subprocess.run(
                [
                    sys.executable, "-m", "coverage", "json",
                    f"--data-file={self.data_file}",
                    f"--include={self.source_path}",
                    "-o", self.json_file,
                ],
                capture_output=True,
                timeout=10,
            )
            if result.returncode == 0 and os.path.exists(self.json_file):
                with open(self.json_file) as jf:
                    report = json.load(jf)
                files = report.get("files", {})
                if files:
                    summary = list(files.values())[0].get("summary", {})
                    covered = summary.get("covered_branches", 0)
                    total = summary.get("num_branches", 0)
                    if total > 0:
                        branch_coverage = covered / total
                        branches_hit = covered
                        total_branches = total
        except Exception:
            pass

        return CoverageResult(
            branch_coverage=branch_coverage,
            branches_hit=branches_hit,
            total_branches=total_branches,
            execution_time_ms=elapsed_ms,
        )

    def shutdown(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
        shutil.rmtree(self.tmpdir, ignore_errors=True)


def instrument(source_code: str, port: int) -> InstrumentedApp:
    app = InstrumentedApp(source_code, port)
    return app
