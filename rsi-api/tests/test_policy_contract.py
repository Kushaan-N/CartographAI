"""Lightweight contract guards for agent/policy.py that don't load the 1B model.

These guard a SILENT, CRITICAL behaviour: PeftModel.from_pretrained loads adapters
frozen by default, so Policy.load() must pass is_trainable=True or train_step()
reloads a checkpoint and GRPO updates nothing. The real behaviour only manifests
on a full model load (too heavy for the fast test suite), so we assert the source
contract to prevent accidental regression.
"""
import inspect
from agent.policy import Policy, decayed_epsilon


def test_epsilon_decays_linearly():
    cfg = {"training": {"exploration_epsilon_start": 0.5,
                        "exploration_epsilon_end": 0.05,
                        "exploration_decay_episodes": 100}}
    assert abs(decayed_epsilon(cfg, 0) - 0.5) < 1e-9       # start
    assert abs(decayed_epsilon(cfg, 50) - 0.275) < 1e-9    # midpoint
    assert abs(decayed_epsilon(cfg, 100) - 0.05) < 1e-9    # end
    assert abs(decayed_epsilon(cfg, 999) - 0.05) < 1e-9    # clamped past horizon


def test_epsilon_falls_back_to_constant():
    # No start/end -> use the constant exploration_epsilon, unchanged over time.
    cfg = {"training": {"exploration_epsilon": 0.3, "total_episodes": 100}}
    assert abs(decayed_epsilon(cfg, 0) - 0.3) < 1e-9
    assert abs(decayed_epsilon(cfg, 100) - 0.3) < 1e-9


def test_epsilon_defaults_to_zero_when_unset():
    assert decayed_epsilon({"training": {}}, 10) == 0.0
    assert decayed_epsilon({}, 10) == 0.0


def test_load_passes_is_trainable():
    src = inspect.getsource(Policy.load)
    assert "is_trainable=True" in src, (
        "Policy.load() must load the adapter with is_trainable=True, else GRPO "
        "train_step reloads a FROZEN model and silently updates no weights."
    )


def test_init_checkpoint_path_is_trainable():
    src = inspect.getsource(Policy.__init__)
    assert "is_trainable=True" in src, (
        "Policy(checkpoint_path=...) must load the adapter trainable for GRPO."
    )


def test_sample_action_supports_epsilon_exploration():
    src = inspect.getsource(Policy.sample_action)
    assert "exploration_epsilon" in src and "_explore_action" in src, (
        "sample_action must honour epsilon-greedy structured exploration."
    )
