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
        tcfg = config["training"]
        # KL anchor to the frozen reference (warmup) policy and an entropy bonus —
        # together these stop the bare policy-gradient update from collapsing the
        # policy onto a single action. kl_coef=0 -> plain REINFORCE (old behaviour).
        self.kl_coef = float(tcfg.get("kl_coef", 0.05))
        self.entropy_coef = float(tcfg.get("entropy_coef", 0.0))
        # Only optimize trainable params — the frozen reference adapter must not
        # be handed to the optimizer.
        trainable = [p for p in policy.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=float(tcfg["learning_rate"]),
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

        loss_terms = []
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
            # Policy-gradient term (REINFORCE with group-relative advantage).
            lp, entropy = self.policy.score_policy(obs, action)
            term = -(lp * advantage) - self.entropy_coef * entropy

            # KL anchor to the frozen reference policy (DeepSeekMath GRPO k3
            # estimator: kl = exp(r) - r - 1 >= 0, with r = log(pi_ref/pi_theta)).
            # Computed on the sampled action; the log-ratio is clamped so the
            # sequence-level sum can't blow exp() up.
            if self.kl_coef > 0:
                lp_ref = self.policy.ref_log_prob(obs, action)
                logr = torch.clamp(lp_ref - lp, -5.0, 5.0)
                kl = torch.exp(logr) - logr - 1.0
                term = term + self.kl_coef * kl

            loss_terms.append(term)

        if not loss_terms:
            return torch.tensor(0.0, requires_grad=True, device=next(self.policy.model.parameters()).device)

        loss = torch.stack(loss_terms).mean()

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
