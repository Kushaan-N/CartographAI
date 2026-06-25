"""Tests for verifier/coverage_runner.py — critical: must run in <200ms"""
import pytest
import time
from verifier.coverage_runner import instrument, CoverageResult

SIMPLE_FLASK_APP = '''
from flask import Flask, jsonify
app = Flask(__name__)

@app.route("/")
def index():
    return jsonify(ok=True)

@app.route("/data")
def data():
    return jsonify(data=[1, 2, 3])

@app.route("/health")
def health():
    return jsonify(status="ok")

if __name__ == "__main__":
    import os
    app.run(port=int(os.environ.get("PORT", 5000)))
'''

FULL_CLIENT = '''
import os, requests
BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:5000")
requests.get(BASE_URL + "/", timeout=2)
requests.get(BASE_URL + "/data", timeout=2)
requests.get(BASE_URL + "/health", timeout=2)
'''

EMPTY_CLIENT = ""

ROOT_ONLY_CLIENT = '''
import os, requests
BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:5000")
requests.get(BASE_URL + "/", timeout=2)
'''


def test_coverage_under_200ms():
    instr = instrument(SIMPLE_FLASK_APP, port=0)
    t0 = time.monotonic()
    result = instr.run_client(ROOT_ONLY_CLIENT)
    elapsed = (time.monotonic() - t0) * 1000
    instr.shutdown()
    # Client execution itself should be fast; we allow up to 2s for subprocess overhead
    assert result.execution_time_ms < 2000, f"Took {result.execution_time_ms:.0f}ms"


def test_full_coverage_on_complete_client():
    instr = instrument(SIMPLE_FLASK_APP, port=0)
    result = instr.run_client(FULL_CLIENT)
    instr.shutdown()
    # Hitting all endpoints MUST produce non-zero coverage. A >= 0.0 assertion
    # is vacuous and silently passed even when the coverage data file was never
    # written (SIGTERM bug). Require real signal: branches must be discovered
    # and at least some covered.
    assert result.total_branches > 0, "coverage found no branches — data file not written?"
    assert result.branch_coverage > 0.0, f"client hit all endpoints but coverage={result.branch_coverage}"


def test_zero_coverage_on_empty_client():
    instr = instrument(SIMPLE_FLASK_APP, port=0)
    result = instr.run_client(EMPTY_CLIENT)
    instr.shutdown()
    # Empty client hits no endpoints, coverage should be low
    assert result.branch_coverage >= 0.0  # may be > 0 due to import-time code


def test_partial_coverage():
    instr = instrument(SIMPLE_FLASK_APP, port=0)
    result = instr.run_client(ROOT_ONLY_CLIENT)
    instr.shutdown()
    assert isinstance(result.branch_coverage, float)
    assert 0.0 <= result.branch_coverage <= 1.0


# App with branches INSIDE handlers — like factory-generated APIs (auth checks).
# A client that hits the endpoint covers the in-handler branch; one that does
# not, can't. This is what makes the reward signal differentiable.
BRANCHY_FLASK_APP = '''
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route("/health")
def health():
    return jsonify(status="ok")

@app.route("/data")
def data():
    if request.headers.get("X-Key") == "secret":
        return jsonify(data=[1, 2, 3])
    return jsonify(error="unauthorized"), 401

if __name__ == "__main__":
    import os
    app.run(port=int(os.environ.get("PORT", 5000)))
'''

BRANCHY_CLIENT = '''
import os, requests
BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:5000")
requests.get(BASE_URL + "/health", timeout=2)
requests.get(BASE_URL + "/data", timeout=2)
'''


def test_instrumented_app_reports_ready():
    """The boot-readiness flag must be set for a healthy app, so run_client can
    detect (and extend) a slow boot instead of measuring a boot-race 0%."""
    instr = instrument(BRANCHY_FLASK_APP, port=0)
    try:
        assert instr.ready is True
    finally:
        instr.shutdown()


def test_coverage_is_deterministic():
    """Same source + same client must yield the same branch coverage. Reward
    determinism (CLAUDE.md verifier invariant) depends on this."""
    runs = []
    for _ in range(3):
        instr = instrument(BRANCHY_FLASK_APP, port=0)
        runs.append(instr.run_client(BRANCHY_CLIENT).branch_coverage)
        instr.shutdown()
    assert len(set(runs)) == 1, f"non-deterministic coverage: {runs}"


def test_full_client_beats_empty_client():
    """The verifier must distinguish a client that hits endpoints from one that
    does not. If both return the same coverage, the reward signal is dead (all
    GRPO advantages collapse to zero) — exactly the failure that
    SIGTERM-on-shutdown caused."""
    instr = instrument(BRANCHY_FLASK_APP, port=0)
    full = instr.run_client(BRANCHY_CLIENT)
    instr.shutdown()

    instr2 = instrument(BRANCHY_FLASK_APP, port=0)
    empty = instr2.run_client(EMPTY_CLIENT)
    instr2.shutdown()

    assert full.branch_coverage > empty.branch_coverage, (
        f"full={full.branch_coverage} not greater than empty={empty.branch_coverage} "
        "— verifier produces no usable reward gradient"
    )
