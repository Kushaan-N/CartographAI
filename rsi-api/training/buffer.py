"""
GRPO rollout buffer. Groups episodes by api_id for contrastive reward.
Needs grpo_group_size (default 4) episodes per api_id to form a group.
"""
from collections import defaultdict
from dataclasses import dataclass
import json


@dataclass
class GRPOBatch:
    groups: list[list[dict]]    # list of groups; each group = grpo_group_size episodes on same API

    def compute_advantages(self) -> list[float]:
        """
        For each group: advantage_i = reward_i - mean(group rewards).
        Return flat list aligned with flat list of all trajectories.

        NOTE: deliberately NOT std-normalized. With this env's quantized,
        near-equal rewards, dividing by a tiny group std would amplify trivial
        reward noise into order-1 advantages — a destabilizer. The KL anchor to
        the frozen reference policy (training/grpo.py) is what controls update
        scale and prevents the policy collapse.
        """
        advantages = []
        for group in self.groups:
            rewards = [ep.get("episode_reward", 0.0) for ep in group]
            mean_r = sum(rewards) / len(rewards) if rewards else 0.0
            for r in rewards:
                advantages.append(r - mean_r)
        return advantages

    def to_json(self) -> str:
        """Serialize to JSON string for Modal transport."""
        return json.dumps({"groups": self.groups})

    @classmethod
    def from_json(cls, s: str) -> "GRPOBatch":
        """Deserialize from JSON string."""
        data = json.loads(s)
        return cls(groups=data["groups"])


class RolloutBuffer:
    def __init__(self, config: dict):
        self.batch_size = config["training"]["episodes_per_update"]
        self.group_size = config["training"]["grpo_group_size"]
        self.by_api: dict[str, list] = defaultdict(list)
        self.total_collected = 0

    def add_episode(self, result: dict):
        """
        Add EpisodeResult dict to buffer keyed by result["api_id"].
        Increment total_collected.
        """
        api_id = result.get("api_id", "unknown")
        self.by_api[api_id].append(result)
        self.total_collected += 1

    def is_ready(self) -> bool:
        """
        True when we have enough complete groups to fill a batch.
        A complete group = group_size episodes with same api_id.
        Count complete groups * group_size >= batch_size.
        """
        complete_groups = sum(1 for eps in self.by_api.values() if len(eps) >= self.group_size)
        return complete_groups * self.group_size >= self.batch_size

    def get_batch(self) -> GRPOBatch:
        """
        Extract complete groups into GRPOBatch. Clear used episodes.
        Only include groups that have exactly group_size episodes.
        """
        groups = []
        used_apis = []
        for api_id, eps in self.by_api.items():
            if len(eps) >= self.group_size:
                groups.append(eps[:self.group_size])
                used_apis.append(api_id)
            if len(groups) * self.group_size >= self.batch_size:
                break
        for api_id in used_apis:
            del self.by_api[api_id]
        return GRPOBatch(groups=groups)

    def clear(self):
        """Reset buffer."""
        self.by_api = defaultdict(list)
        self.total_collected = 0
