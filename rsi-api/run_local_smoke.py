"""
Local (no-Modal) GRPO smoke/progress runner.

Runs the real GRPO loop on CPU/MPS so you can watch progress before committing to
Modal. Mirrors training/loop.py semantics (same-seed contrastive groups,
RolloutBuffer, GRPOTrainer) but collects locally and:
  - anneals exploration epsilon over training (decayed_epsilon), and
  - periodically runs an EXPLORATION-OFF eval (epsilon=0) to measure what the LLM
    actually learned, separate from the structured-exploration scaffold.

Usage:
  python run_local_smoke.py [total_episodes] [--config configs/train_config_local.yaml] [--checkpoint checkpoints/warmup]
"""
import sys, json, time, random, argparse, yaml
from curriculum.factory import generate_api
from training.episode import run_episode
from training.buffer import RolloutBuffer
from training.grpo import GRPOTrainer
from agent.policy import Policy, decayed_epsilon


def collect_group(policy, level, cfg):
    """One GRPO group: grpo_group_size episodes on the SAME API (same seed)."""
    gs = cfg["training"]["grpo_group_size"]
    seed = random.randint(0, 2**31 - 1)
    api_id = f"g{seed}"
    eps_out = []
    for _ in range(gs):
        c = dict(cfg); c["seed"] = seed; c["api_id"] = api_id
        api = generate_api(level=level, config=c)
        try:
            r = run_episode(policy, api, c); r["api_id"] = api_id
            eps_out.append(r)
        finally:
            api.shutdown()
    return eps_out


def eval_pure_policy(policy, level, n, cfg):
    """Eval with exploration OFF — measures the LLM's learned behaviour."""
    saved = policy.exploration_epsilon
    saved_greedy = getattr(policy, "eval_greedy", False)
    policy.exploration_epsilon = 0.0
    policy.eval_greedy = True   # measure the policy's mode, not a temp-0.7 sample
    try:
        covs = []
        for _ in range(n):
            api = generate_api(level=level, config=cfg)
            try:
                covs.append(run_episode(policy, api, cfg)["branch_coverage"])
            finally:
                api.shutdown()
        return sum(covs) / max(len(covs), 1), (max(covs) if covs else 0.0)
    finally:
        policy.exploration_epsilon = saved
        policy.eval_greedy = saved_greedy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("total_episodes", nargs="?", type=int, default=40)
    ap.add_argument("--config", default="configs/train_config_local.yaml")
    ap.add_argument("--checkpoint", default="checkpoints/warmup")
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--eval-every-updates", type=int, default=3,
                    help="run an exploration-off eval on the first update and every N updates")
    ap.add_argument("--metrics", default="local_metrics.jsonl")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    import os
    ckpt = args.checkpoint if args.checkpoint and os.path.exists(args.checkpoint) else None
    print(f"=== Local GRPO smoke: {args.total_episodes} episodes, checkpoint={ckpt} ===", flush=True)
    policy = Policy(cfg, checkpoint_path=ckpt)
    trainer = GRPOTrainer(policy, cfg)
    buffer = RolloutBuffer(cfg)

    open(args.metrics, "w").close()
    episode_count = 0
    update_idx = 0
    t0 = time.time()
    while episode_count < args.total_episodes:
        # Anneal exploration as training progresses.
        eps = decayed_epsilon(cfg, episode_count)
        policy.exploration_epsilon = eps

        n_groups = max(1, cfg["training"]["episodes_per_update"] // cfg["training"]["grpo_group_size"])
        batch_covs = []
        for _ in range(n_groups):
            grp = collect_group(policy, args.level, cfg)
            for r in grp:
                buffer.add_episode(r); batch_covs.append(r["branch_coverage"])
                episode_count += 1

        loss = trainer.step(buffer.get_batch()) if buffer.is_ready() else None
        update_idx += 1
        mean_cov = sum(batch_covs) / max(len(batch_covs), 1)

        rec = {"episode": episode_count, "update": update_idx, "epsilon": round(eps, 4),
               "collect_mean_coverage": round(mean_cov, 4),
               "loss": round(loss, 6) if loss is not None else None}

        # Exploration-off eval on the first update and every N updates — measures
        # the LLM's learned behaviour (should rise over training as it internalizes
        # discovery and epsilon decays).
        if update_idx == 1 or update_idx % args.eval_every_updates == 0:
            pe_mean, pe_max = eval_pure_policy(policy, args.level, 3, cfg)
            rec["pure_policy_mean_coverage"] = round(pe_mean, 4)
            rec["pure_policy_max_coverage"] = round(pe_max, 4)

        with open(args.metrics, "a") as f:
            f.write(json.dumps(rec) + "\n")
        pe = rec.get("pure_policy_mean_coverage")
        print(f"[ep {episode_count}/{args.total_episodes}] eps={eps:.3f} "
              f"collect_cov={mean_cov:.1%} loss={loss if loss is None else round(loss,5)}"
              + (f" | pure_policy_cov={pe:.1%}" if pe is not None else ""), flush=True)

    print(f"\n=== DONE in {round(time.time()-t0)}s — metrics in {args.metrics} ===", flush=True)
    policy.save("checkpoints/grpo_local")


if __name__ == "__main__":
    main()
