"""Tests for curriculum/factory.py"""
import re
import pytest
import requests
from curriculum.factory import generate_api


CONFIG = {
    "curriculum": {
        "levels": {
            "1": {"endpoints": 3, "auth_schemes": 1, "dependencies": 0, "red_herrings": 0},
            "2": {"endpoints": 6, "auth_schemes": 2, "dependencies": 1, "red_herrings": 1},
            "3": {"endpoints": 10, "auth_schemes": 3, "dependencies": 3, "red_herrings": 2},
            "4": {"endpoints": 15, "auth_schemes": 4, "dependencies": 5, "red_herrings": 3},
        },
        "zpd_advance_threshold": 0.70,
        "zpd_retreat_threshold": 0.40,
        "rolling_window": 30,
    }
}


def test_generate_returns_alive_api():
    """generate_api() returns a RunningAPI that responds to GET /health"""
    api = generate_api(level=1, config=CONFIG)
    try:
        assert api.is_alive(), "API should be alive after generation"
        r = requests.get(api.url + "/health", timeout=2)
        assert r.status_code == 200
        assert r.json().get("status") == "ok"
    finally:
        api.shutdown()


def test_each_call_produces_unique_api():
    """Two calls produce different api_ids and different endpoint sets"""
    api1 = generate_api(level=1, config=CONFIG)
    api2 = generate_api(level=1, config=CONFIG)
    try:
        assert api1.api_id != api2.api_id
        # Different secrets/tokens make the source codes differ
        assert api1.source_code != api2.source_code
    finally:
        api1.shutdown()
        api2.shutdown()


def test_shutdown_kills_subprocess():
    """api.shutdown() terminates process, is_alive() returns False after"""
    api = generate_api(level=1, config=CONFIG)
    assert api.is_alive()
    api.shutdown()
    assert not api.is_alive()


def test_level_1_endpoint_count():
    """Level 1 API has exactly 3 non-health endpoints"""
    api = generate_api(level=1, config=CONFIG)
    try:
        routes = re.findall(r'@app\.route\("(/[^"]*)"', api.source_code)
        # "/" is the discovery index, a system route like /health and /_inject.
        non_system = [r for r in routes if r not in ("/", "/health", "/_inject")]
        assert len(non_system) == 3, f"Expected 3 non-system routes, got {len(non_system)}: {non_system}"
    finally:
        api.shutdown()


def test_level_3_has_red_herrings():
    """Level 3 API source code contains red herring endpoint markers"""
    api = generate_api(level=3, config=CONFIG)
    try:
        assert "# RED_HERRING" in api.source_code, "Expected RED_HERRING marker in source"
    finally:
        api.shutdown()
