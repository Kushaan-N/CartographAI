"""Tests for training/buffer.py — GRPO grouping and advantage computation."""
from training.buffer import GRPOBatch, RolloutBuffer


def _ep(api_id, reward):
    return {"api_id": api_id, "episode_reward": reward, "trajectory": [{"x": 1}]}


def test_advantages_are_group_centered():
    """advantage_i = reward_i - mean(group). Each group must be zero-sum, and the
    flat advantage list must align with the flat group->episode order GRPO uses."""
    batch = GRPOBatch(groups=[
        [_ep("a", 0.2), _ep("a", 0.8)],   # mean 0.5 -> [-0.3, +0.3]
        [_ep("b", 0.1), _ep("b", 0.1)],   # mean 0.1 -> [0.0, 0.0]
    ])
    adv = batch.compute_advantages()
    assert len(adv) == 4
    assert abs(adv[0] + 0.3) < 1e-9 and abs(adv[1] - 0.3) < 1e-9
    assert abs(adv[2]) < 1e-9 and abs(adv[3]) < 1e-9
    # zero-sum within each group
    assert abs(adv[0] + adv[1]) < 1e-9
    assert abs(adv[2] + adv[3]) < 1e-9


def test_uniform_group_gives_zero_advantages():
    """A group with identical rewards yields zero advantages (GRPO skips it)."""
    batch = GRPOBatch(groups=[[_ep("a", 0.25), _ep("a", 0.25)]])
    assert all(abs(a) < 1e-9 for a in batch.compute_advantages())


def test_buffer_groups_by_api_and_readiness():
    config = {"training": {"episodes_per_update": 4, "grpo_group_size": 2}}
    buf = RolloutBuffer(config)
    assert not buf.is_ready()
    buf.add_episode(_ep("a", 0.1)); buf.add_episode(_ep("a", 0.2))
    assert not buf.is_ready()  # one complete group (2) < batch_size (4)
    buf.add_episode(_ep("b", 0.3)); buf.add_episode(_ep("b", 0.4))
    assert buf.is_ready()      # two complete groups == batch_size
    batch = buf.get_batch()
    assert len(batch.groups) == 2
    assert all(len(g) == 2 for g in batch.groups)
    # consumed groups are cleared
    assert buf.total_collected >= 0 and len(buf.by_api) == 0


def test_batch_json_roundtrip():
    batch = GRPOBatch(groups=[[_ep("a", 0.5), _ep("a", 0.5)]])
    restored = GRPOBatch.from_json(batch.to_json())
    assert restored.compute_advantages() == batch.compute_advantages()
