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


def test_generate_client_handles_tricky_endpoint_names():
    """Regression: endpoint paths with dots / dashes / slashes / leading digits
    (which can arrive via response-body parsing) must not break client codegen.
    A single bad function name (e.g. call_v1.2_data) would make the WHOLE client
    a SyntaxError, gating the episode reward to 0.0 despite real discovery."""
    from verifier.reward import is_syntax_valid
    memory = WorkingMemory()
    for ep in ["/v1.2/data", "/foo.bar", "/a-b", "/2024", "/x/y/z", "/health"]:
        memory.add_hypothesis(ep)
        memory.update_node(ep, "discovered", last_status_code=200)
    client = generate_client(memory, "http://127.0.0.1:9999")
    assert is_syntax_valid(client), f"Invalid Python:\n{client}"


class _DiscoverProbePolicy:
    """Legitimate (observation-only) policy: probe the discovery index, then probe
    the endpoints the observation surfaces, with auth headers. Mirrors the
    epsilon-greedy exploration behaviour without an LLM."""
    def __init__(self):
        self.action_space = ActionSpace()

    def sample_action(self, observation):
        from agent.actions import Action
        hyp = observation.get("hypothesized_endpoints", []) or []
        disc = set(observation.get("discovered_endpoints", []) or [])
        h = {"Authorization": "Bearer t", "X-API-Key": "t", "X-Service-Token": "t"}
        if "/" in hyp and "/" not in disc:
            return Action("GET", "/", {}, None)
        unprobed = [e for e in hyp if e not in disc and e != "/"]
        if unprobed:
            return Action("GET", unprobed[0], h, None)
        # Nothing new to probe: re-hit /health (always 200) rather than spraying
        # 404 guesses that would trip the fragility lock.
        return Action("GET", "/health", {}, None)

    def log_prob(self, obs, action):
        import torch
        return torch.tensor(-1.0, requires_grad=True)


def test_discovering_policy_gets_nonzero_coverage_via_index():
    """End-to-end: with the discoverable index, a policy that probes "/" and then
    the surfaced endpoints achieves non-zero coverage and a positive reward.
    Guards the whole reward chain (factory index -> discovery -> client -> coverage)."""
    from curriculum.factory import generate_api
    # Use the real configs' softened fragility (10): the always-404 seeded
    # hypotheses (/api, /status, /login, /auth) burn fragility budget, and at the
    # default threshold of 5 a deterministic probe order can lock before reaching
    # the real endpoints. Real training uses threshold 10 + randomized exploration.
    cfg = dict(CONFIG)
    cfg["training"] = {"max_episode_steps": 20, "fragility_threshold": 10}
    api = generate_api(level=1, config=cfg)
    try:
        result = run_episode(_DiscoverProbePolicy(), api, cfg)
    finally:
        api.shutdown()
    assert not result["metadata"]["api_locked"], "unexpectedly fragility-locked at threshold 10"
    assert result["branch_coverage"] > 0.07, f"coverage {result['branch_coverage']:.3f} not above floor"
    assert result["episode_reward"] > 0


def test_api_locked_yields_fragility_penalty():
    """A policy that only ever 4xxs trips the fragility lock; episode_reward must
    equal the fragility penalty and failure_mode must be api_locked."""
    from curriculum.factory import generate_api
    from agent.actions import Action

    class AlwaysMissPolicy:
        def __init__(self): self.action_space = ActionSpace()
        def sample_action(self, obs):
            return Action("GET", "/definitely_not_a_real_endpoint_xyz", {}, None)
        def log_prob(self, obs, a):
            import torch; return torch.tensor(-1.0, requires_grad=True)

    cfg = dict(CONFIG)
    cfg["training"] = {"max_episode_steps": 20, "fragility_threshold": 5, "fragility_penalty": -1.0}
    api = generate_api(level=1, config=cfg)
    try:
        result = run_episode(AlwaysMissPolicy(), api, cfg)
    finally:
        api.shutdown()
    assert result["metadata"]["api_locked"] is True
    assert result["failure_mode"] == "api_locked"
    assert result["episode_reward"] == -1.0
