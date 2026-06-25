"""Lightweight contract guards for agent/policy.py that don't load the 1B model.

These guard a SILENT, CRITICAL behaviour: PeftModel.from_pretrained loads adapters
frozen by default, so Policy.load() must pass is_trainable=True or train_step()
reloads a checkpoint and GRPO updates nothing. The real behaviour only manifests
on a full model load (too heavy for the fast test suite), so we assert the source
contract to prevent accidental regression.
"""
import inspect
from agent.policy import Policy


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
