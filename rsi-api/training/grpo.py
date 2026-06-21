"""
GRPO trainer. Owns all gradient updates.
policy.update() is a stub — this file does the actual weight changes.
"""
import torch
import torch.nn.functional as F
from training.buffer import GRPOBatch


class GRPOTrainer:
    def __init__(self, policy, config: dict):
        self.policy = policy
        self.config = config
        self.step_count = 0
        self.optimizer = torch.optim.AdamW(
            policy.model.parameters(),
            lr=float(config["training"]["learning_rate"]),
            weight_decay=0.01,
        )

    def compute_loss(self, batch: GRPOBatch, advantages: list[float]) -> torch.Tensor:
        """
        GRPO loss using one sampled step per episode.
        Running log_prob on all steps causes OOM with large batches.
        We sample one step per episode instead.
        """
        from agent.actions import Action as ActionClass
        import random

        all_log_probs = []
        flat_trajectories = [ep for group in batch.groups for ep in group]

        for ep, advantage in zip(flat_trajectories, advantages):
            trajectory = ep.get("trajectory", [])
            if not trajectory:
                continue

            # Sample ONE step per episode instead of all steps
            step = random.choice(trajectory)
            obs = step.get("obs")
            action = step.get("action")

            if not obs or not action:
                continue

            if isinstance(action, dict):
                action = ActionClass(
                    method=action["method"],
                    endpoint=action["endpoint"],
                    headers=action.get("headers", {}),
                    body=action.get("body"),
                )

            torch.cuda.empty_cache()
            lp = self.policy.log_prob(obs, action)
            all_log_probs.append(lp * advantage)
        if not all_log_probs:
            return torch.tensor(0.0, requires_grad=True, device=next(self.policy.model.parameters()).device)

        loss = -torch.stack(all_log_probs).mean()

        # All advantages were zero — no grad_fn survives the multiply-by-zero
        if not loss.requires_grad:
            loss = loss + torch.tensor(0.0, requires_grad=True, device=loss.device)

        return loss

    def step(self, batch: GRPOBatch) -> float:
        """One gradient update."""
        advantages = batch.compute_advantages()
        self.optimizer.zero_grad()
        loss = self.compute_loss(batch, advantages)

        if loss.requires_grad:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.model.parameters(), 1.0)
            self.optimizer.step()
        else:
            print("[GRPO] Skipping update: all advantages zero (uniform rewards)")

        self.step_count += 1
        return loss.item()
