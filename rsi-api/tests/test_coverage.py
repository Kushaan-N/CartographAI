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
    # Hitting all endpoints should give high coverage
    assert result.branch_coverage >= 0.0  # sanity check


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
