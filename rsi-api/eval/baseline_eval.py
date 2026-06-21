"""
Baseline evaluation — Gemma-3-4B-IT with zero training.
Establishes "before" numbers for the results table.
Run this before training starts.

Usage: python eval/baseline_eval.py
       python eval/baseline_eval.py --n-episodes 3 --levels 1 2
       python eval/baseline_eval.py --random-only  # fast check, no GPU needed
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def run_random_baseline(levels: list[int], n_episodes: int, config: dict) -> dict:
    """
    Fast baseline using random policy — no GPU needed.
    Use this first to confirm the full pipeline works
    before loading the actual model.
    """
    from curriculum.factory import generate_api
    from training.episode import run_episode
    from agent.actions import ActionSpace

    class RandomPolicy:
        def __init__(self): self.action_space = ActionSpace()
        def sample_action(self, obs): return self.action_space.sample_random()
        def log_prob(self, obs, action):
            import torch; return torch.tensor(-1.0, requires_grad=True)

    results = {}
    for level in levels:
        print(f"\nRunning {n_episodes} random policy episodes on Level {level}...")
        level_results = []
        for i in range(n_episodes):
            api = generate_api(level=level, config=config)
            try:
                t0 = time.monotonic()
                result = run_episode(RandomPolicy(), api, config)
                elapsed = time.monotonic() - t0
                level_results.append({
                    "coverage": result["branch_coverage"],
                    "success": result["success"],
                    "steps": result["metadata"]["total_steps_taken"],
                    "failure_mode": result["failure_mode"],
                    "elapsed_s": elapsed
                })
                print(f"  Episode {i+1}: coverage={result['branch_coverage']:.1%} "
                      f"success={result['success']} "
                      f"steps={result['metadata']['total_steps_taken']} "
                      f"({elapsed:.1f}s)")
            except Exception as e:
                print(f"  Episode {i+1}: ERROR — {e}")
                level_results.append({"coverage": 0.0, "success": False, "steps": 0, "error": str(e)})
            finally:
                api.shutdown()

        coverages = [r["coverage"] for r in level_results]
        successes = [r["success"] for r in level_results]
        results[str(level)] = {
            "policy": "random",
            "mean_coverage": sum(coverages) / len(coverages),
            "success_rate": sum(successes) / len(successes),
            "mean_steps": sum(r.get("steps", 0) for r in level_results) / len(level_results),
            "n_episodes": len(level_results),
            "episodes": level_results
        }

    return results


def run_base_model_baseline(levels: list[int], n_episodes: int, config: dict) -> dict:
    """
    True zero-shot baseline using Gemma-3-4B-IT with NO LoRA.
    Requires GPU. Shows what the model knows about APIs before any training.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from agent.actions import ActionSpace, Action
    from curriculum.factory import generate_api
    from training.episode import run_episode

    MODEL_NAME = "google/gemma-3-1b-it"

    print(f"Loading {MODEL_NAME} base model (no LoRA)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    device = next(model.parameters()).device
    print(f"Model loaded on {device}")

    action_space = ActionSpace()

    class BaseModelPolicy:
        """Gemma-3-4B-IT with no LoRA — pure zero-shot performance."""

        def sample_action(self, observation: dict) -> Action:
            messages = [{
                "role": "user",
                "content": (
                    f"You are an API exploration agent. Discover all endpoints "
                    f"of an undocumented API.\n\n"
                    f"Step: {observation.get('episode_step', 0)}\n"
                    f"Discovered: {observation.get('discovered_endpoints', [])}\n"
                    f"Hypothesized: {observation.get('hypothesized_endpoints', [])}\n"
                    f"Auth tokens acquired: {observation.get('auth_tokens_acquired', 0)}\n"
                    f"Last response: "
                    f"status={observation.get('last_response', {}).get('status_code', 'none')} "
                    f"body={str(observation.get('last_response', {}).get('body', ''))[:300]}\n\n"
                    f"Output JSON only, no explanation:\n"
                    f'{"{"}"method": "GET", "endpoint": "/path", '
                    f'"headers": {"{}"}, "body": null{"}"}'
                )
            }]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            generated = outputs[0][inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(generated, skip_special_tokens=True)
            action = action_space.from_model_output(text)
            if action is None:
                print(f"    [parse fail] model output: {text[:100]}")
                action = action_space.sample_random()
            return action

        def log_prob(self, obs, action):
            import torch; return torch.tensor(-1.0, requires_grad=True)

    policy = BaseModelPolicy()
    results = {}

    for level in levels:
        print(f"\nRunning {n_episodes} base model episodes on Level {level}...")
        level_results = []

        for i in range(n_episodes):
            api = generate_api(level=level, config=config)
            try:
                t0 = time.monotonic()
                result = run_episode(policy, api, config)
                elapsed = time.monotonic() - t0
                level_results.append({
                    "coverage": result["branch_coverage"],
                    "success": result["success"],
                    "steps": result["metadata"]["total_steps_taken"],
                    "failure_mode": result["failure_mode"],
                    "auth_tokens": result["metadata"]["auth_tokens_acquired"],
                    "elapsed_s": elapsed
                })
                print(f"  Episode {i+1}: coverage={result['branch_coverage']:.1%} "
                      f"success={result['success']} "
                      f"steps={result['metadata']['total_steps_taken']} "
                      f"auth_tokens={result['metadata']['auth_tokens_acquired']} "
                      f"({elapsed:.1f}s)")
            except Exception as e:
                print(f"  Episode {i+1}: ERROR — {e}")
                import traceback; traceback.print_exc()
                level_results.append({
                    "coverage": 0.0, "success": False,
                    "steps": 0, "error": str(e)
                })
            finally:
                api.shutdown()

        coverages = [r["coverage"] for r in level_results]
        successes = [r["success"] for r in level_results]
        results[str(level)] = {
            "policy": "gemma-3-4b-it-base",
            "mean_coverage": sum(coverages) / len(coverages),
            "success_rate": sum(successes) / len(successes),
            "mean_steps": sum(r.get("steps", 0) for r in level_results) / len(level_results),
            "n_episodes": len(level_results),
            "episodes": level_results
        }

    return results


def print_results_table(results: dict, policy_name: str):
    """Print a clean results table to stdout."""
    endpoint_counts = {"1": 3, "2": 6, "3": 10, "4": 15}

    print("\n" + "="*60)
    print(f"  BASELINE RESULTS — {policy_name}")
    print("="*60)
    print(f"{'Level':<20} {'Coverage':>10} {'Success':>10} {'Avg Steps':>10}")
    print("-"*60)

    total_coverage = []
    for level, data in sorted(results.items()):
        ep_count = endpoint_counts.get(level, "?")
        label = f"Level {level} ({ep_count} endpoints)"
        cov = data["mean_coverage"]
        sr = data["success_rate"]
        steps = data["mean_steps"]
        n = data["n_episodes"]
        total_coverage.append(cov)
        print(f"{label:<20} {cov:>9.1%} {sr:>9.1%} {steps:>10.1f}  (n={n})")

    print("-"*60)
    mean = sum(total_coverage) / len(total_coverage)
    print(f"{'Mean':.<20} {mean:>9.1%}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Baseline evaluation for RSI-API")
    parser.add_argument("--n-episodes", type=int, default=5,
                        help="Episodes per level (default 5, use 3 for quick check)")
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3, 4],
                        help="Curriculum levels to evaluate (default: 1 2 3 4)")
    parser.add_argument("--random-only", action="store_true",
                        help="Use random policy only — no GPU needed, fast pipeline check")
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--output", default="eval/baseline_results.json")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    Path("eval").mkdir(exist_ok=True)

    if args.random_only:
        print("Running RANDOM POLICY baseline (no GPU needed)...")
        print("Use this to verify the pipeline works before loading the model.")
        results = run_random_baseline(args.levels, args.n_episodes, config)
        policy_name = "RANDOM POLICY"
    else:
        print("Running GEMMA-3-4B-IT BASE MODEL baseline (requires GPU)...")
        print("This shows true zero-shot performance before any RL training.")
        results = run_base_model_baseline(args.levels, args.n_episodes, config)
        policy_name = "GEMMA-3-4B-IT (no training)"

    print_results_table(results, policy_name)

    output = {
        "policy": policy_name,
        "levels": results,
        "config": {
            "n_episodes": args.n_episodes,
            "levels_tested": args.levels,
            "max_steps": config["training"]["max_episode_steps"],
            "max_total_steps": config["training"].get("max_total_steps", 35),
        }
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
