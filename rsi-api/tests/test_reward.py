"""Tests for verifier/reward.py"""
import pytest
from verifier.reward import (
    compute_episode_reward,
    compute_step_reward,
    is_syntax_valid,
    StepInfo,
)
from verifier.coverage_runner import CoverageResult


CONFIG = {
    "reward": {
        "coverage_weight": 1.0,
        "destructive_penalty": 0.3,
        "new_endpoint_bonus": 0.05,
        "auth_token_bonus": 0.15,
        "repeated_request_penalty": 0.1,
    }
}


def test_invalid_syntax_returns_zero():
    assert not is_syntax_valid("def foo(: pass")
    result = CoverageResult(branch_coverage=0.9, branches_hit=9, total_branches=10, execution_time_ms=50)
    r = compute_episode_reward(result, "def foo(: pass", [], CONFIG)
    assert r == 0.0


def test_valid_syntax_gate_passes():
    assert is_syntax_valid("def foo(): pass")
    assert is_syntax_valid("import os\nprint(os.getcwd())")


def test_auth_token_bonus():
    info = StepInfo(auth_token_acquired=True)
    r = compute_step_reward(info, CONFIG)
    assert abs(r - 0.15) < 1e-6


def test_repeated_request_penalty():
    info = StepInfo(request_was_repeated=True)
    r = compute_step_reward(info, CONFIG)
    assert abs(r - (-0.1)) < 1e-6


def test_episode_reward_formula():
    result = CoverageResult(branch_coverage=0.8, branches_hit=8, total_branches=10, execution_time_ms=50)
    r = compute_episode_reward(result, "def foo(): pass", [], CONFIG)
    assert abs(r - 0.8) < 1e-6


def test_step_reward_new_endpoint():
    info = StepInfo(new_endpoints_discovered=2)
    r = compute_step_reward(info, CONFIG)
    assert abs(r - 0.10) < 1e-6


def test_step_reward_combined():
    info = StepInfo(auth_token_acquired=True, new_endpoints_discovered=1)
    r = compute_step_reward(info, CONFIG)
    assert abs(r - 0.20) < 1e-6
