"""Integration test: one full episode end-to-end (no GPU, random policy)"""
import pytest
import yaml
from training.episode import run_episode, build_observation, generate_client
from agent.actions import ActionSpace
from agent.memory import WorkingMemory


CONFIG = {
    "curriculum": {
        "levels": {
            "1": {"endpoints": 3, "auth_schemes": 1, "dependencies": 0, "red_herrings": 0},
        },
        "zpd_advance_threshold": 0.70,
        "zpd_retreat_threshold": 0.40,
        "rolling_window": 30,
    },
    "training": {
        "max_episode_steps": 5,
    },
    "reward": {
        "coverage_weight": 1.0,
        "destructive_penalty": 0.3,
        "new_endpoint_bonus": 0.05,
        "auth_token_bonus": 0.15,
        "repeated_request_penalty": 0.1,
    },
}


class RandomPolicy:
    def __init__(self):
        self.action_space = ActionSpace()

    def sample_action(self, observation):
        return self.action_space.sample_random()

    def log_prob(self, obs, action):
        import torch
        return torch.tensor(-1.0, requires_grad=True)


def test_episode_returns_required_fields():
    from curriculum.factory import generate_api
    api = generate_api(level=1, config=CONFIG)
    try:
        result = run_episode(RandomPolicy(), api, CONFIG)
    finally:
        api.shutdown()

    required = ["trajectory", "episode_reward", "branch_coverage",
                "success", "failure_mode", "metadata"]
    for key in required:
        assert key in result, f"Missing key: {key}"


def test_episode_respects_max_steps():
    from curriculum.factory import generate_api
    api = generate_api(level=1, config=CONFIG)
    try:
        result = run_episode(RandomPolicy(), api, CONFIG)
    finally:
        api.shutdown()
    assert len(result["trajectory"]) <= CONFIG["training"]["max_episode_steps"]


def test_generate_client_valid_python():
    from verifier.reward import is_syntax_valid
    memory = WorkingMemory()
    memory.add_hypothesis("/users")
    memory.update_node("/users", "discovered")
    memory.add_hypothesis("/orders")
    memory.update_node("/orders", "discovered")
    client = generate_client(memory, "http://127.0.0.1:9999")
    assert is_syntax_valid(client), f"Invalid Python:\n{client}"


def test_build_observation_structure():
    memory = WorkingMemory()
    memory.add_hypothesis("/users")
    obs = build_observation(memory, {"status_code": 200, "body": {}}, step=3)
    assert "discovered_endpoints" in obs
    assert "hypothesized_endpoints" in obs
    assert "episode_step" in obs
    assert obs["episode_step"] == 3
    assert "last_response" in obs
