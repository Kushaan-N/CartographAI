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

        # Wait for health check
        deadline = time.time() + 5.0
        import requests as req
        while time.time() < deadline:
            try:
                r = req.get(f"{self.url}/health", timeout=0.5)
                if r.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(0.1)

    def run_client(self, client_code: str) -> CoverageResult:
        t0 = time.monotonic()

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

        # Stop the Flask app so coverage data is flushed
        self.process.terminate()
        try:
            self.process.wait(timeout=3)
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
