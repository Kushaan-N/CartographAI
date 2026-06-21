"""
Automated evaluation script.

Runs N episodes per curriculum level with BOTH the base model and
the trained model, collects coverage results, and produces the
results table.

Usage:
  # Run eval on Modal (recommended — faster, no local GPU needed)
  python eval/run_eval.py --checkpoint /checkpoints/latest --n-episodes 10 --modal

  # Run eval locally (requires local GPU)
  python eval/run_eval.py --checkpoint checkpoints/latest --n-episodes 10

  # Just reformat existing results
  python eval/run_eval.py --results-only

Output:
  Prints table to stdout
  Saves to eval/results.json
  Saves markdown version to eval/results.md
"""
import argparse
import json
import yaml
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_random_policy():
    from agent.actions import ActionSpace
    import torch

    class RandomPolicy:
        def __init__(self):
            self.action_space = ActionSpace()

        def sample_action(self, obs):
            return self.action_space.sample_random()

        def log_prob(self, obs, action):
            return torch.tensor(-1.0, requires_grad=True)

    return RandomPolicy()


def _run_episodes_with_policy(policy, level: int, n_episodes: int, config: dict) -> list[dict]:
    from curriculum.factory import generate_api
    from training.episode import run_episode

    results = []
    for _ in range(n_episodes):
        api = generate_api(level=level, config=config)
        try:
            results.append(run_episode(policy, api, config))
        finally:
            api.shutdown()
    return results


def run_base_model_episodes(level: int, n_episodes: int, config: dict) -> list[dict]:
    """
    Run n_episodes with the BASE model (no LoRA, no training).
    Uses ActionSpace.sample_random() as a stand-in for zero-shot base model.

    Returns list of episode result dicts with keys:
    branch_coverage, success, trajectory (list of steps), metadata
    """
    return _run_episodes_with_policy(_make_random_policy(), level, n_episodes, config)


def run_trained_model_episodes(
    checkpoint: str,
    level: int,
    n_episodes: int,
    config: dict,
) -> list[dict]:
    """
    Run n_episodes with the TRAINED model (LoRA checkpoint).
    Load Policy from checkpoint, run run_episode() for each.
    Returns list of episode result dicts.
    """
    from agent.policy import Policy

    policy = Policy(config)
    policy.load(checkpoint)
    return _run_episodes_with_policy(policy, level, n_episodes, config)


def aggregate_results(episodes: list[dict]) -> dict:
    """
    Aggregate list of episode results into summary stats.

    Returns:
    {
      "mean_coverage": float,
      "success_rate": float,        # fraction of episodes with coverage > 0.5
      "mean_steps": float,          # mean steps per episode
      "n_episodes": int,
      "coverage_distribution": list[float]  # all coverage values for histogram
    }
    """
    if not episodes:
        return {
            "mean_coverage": 0.0,
            "success_rate": 0.0,
            "mean_steps": 0.0,
            "n_episodes": 0,
            "coverage_distribution": [],
        }
    coverages = [e["branch_coverage"] for e in episodes]
    successes = [1 if e["success"] else 0 for e in episodes]
    steps = [len(e.get("trajectory", [])) for e in episodes]
    return {
        "mean_coverage": sum(coverages) / len(coverages),
        "success_rate": sum(successes) / len(successes),
        "mean_steps": sum(steps) / len(steps),
        "n_episodes": len(episodes),
        "coverage_distribution": coverages,
    }


def run_full_eval(
    checkpoint: str,
    n_episodes: int,
    config: dict,
    use_modal: bool = False,
) -> dict:
    """
    Run full evaluation across all 4 curriculum levels.

    For each level:
    1. Run n_episodes with base model -> aggregate_results
    2. Run n_episodes with trained model -> aggregate_results
    3. Compute improvement = trained.mean_coverage - base.mean_coverage

    Returns full results dict matching format expected by table.py.

    If use_modal=True: call run_eval_modal.remote() from modal_app.py
    If use_modal=False: run locally (loads trained policy once for efficiency)
    """
    if use_modal:
        from modal_app import run_eval_modal
        return run_eval_modal.remote(checkpoint, n_episodes, config)

    from agent.policy import Policy

    base_policy = _make_random_policy()
    trained_policy = Policy(config)
    trained_policy.load(checkpoint)

    curriculum_levels = config.get("curriculum", {}).get("levels", {})
    levels = {}

    for lvl in [1, 2, 3, 4]:
        lvl_cfg = curriculum_levels.get(str(lvl)) or curriculum_levels.get(lvl, {})
        ep_count = lvl_cfg.get("endpoints")

        print(f"  Level {lvl}: running base model ({n_episodes} episodes)...", flush=True)
        base_eps = _run_episodes_with_policy(base_policy, lvl, n_episodes, config)
        base_stats = aggregate_results(base_eps)

        print(f"  Level {lvl}: running trained model ({n_episodes} episodes)...", flush=True)
        trained_eps = _run_episodes_with_policy(trained_policy, lvl, n_episodes, config)
        trained_stats = aggregate_results(trained_eps)

        improvement = trained_stats["mean_coverage"] - base_stats["mean_coverage"]
        levels[str(lvl)] = {
            "endpoints": ep_count,
            "base": base_stats,
            "trained": trained_stats,
            "improvement": improvement,
        }

    return {
        "episodes_trained": 0,
        "curriculum_level_reached": 1,
        "levels": levels,
    }


def main():
    parser = argparse.ArgumentParser(description="RSI-API Evaluation")
    parser.add_argument("--checkpoint", default="checkpoints/latest",
                        help="Path to trained model checkpoint")
    parser.add_argument("--n-episodes", type=int, default=10,
                        help="Episodes per level per model (10 recommended)")
    parser.add_argument("--modal", action="store_true",
                        help="Run eval on Modal instead of locally")
    parser.add_argument("--results-only", action="store_true",
                        help="Just reformat existing eval/results.json")
    parser.add_argument("--config", default="configs/train_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    from eval.table import (
        format_terminal_table, format_markdown_table,
        save_results, load_results,
    )

    if args.results_only:
        results = load_results()
    else:
        episodes_trained = 0
        level_reached = 1
        try:
            with open("logs/metrics.jsonl") as f:
                lines = f.readlines()
                if lines:
                    last = json.loads(lines[-1])
                    episodes_trained = last.get("episode", 0)
                    level_reached = last.get("curriculum_level", 1)
        except FileNotFoundError:
            pass

        results = run_full_eval(
            checkpoint=args.checkpoint,
            n_episodes=args.n_episodes,
            config=config,
            use_modal=args.modal,
        )
        results["episodes_trained"] = episodes_trained
        results["curriculum_level_reached"] = level_reached

        save_results(results)
        Path("eval/results.md").write_text(format_markdown_table(results))

    print(format_terminal_table(results))


if __name__ == "__main__":
    main()
