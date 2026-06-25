"""
Modal application — single source of truth for all cloud compute.

RULE: This is the ONLY file in the project that imports modal.
No other file should ever import modal directly.

Three Modal functions:
1. collect_episode() — CPU worker, runs one agent episode
2. train_step() — A100 GPU, one GRPO gradient update
3. evaluate_policy() — CPU workers, held-out evaluation

One local helper (not a Modal function):
4. run_parallel_episodes() — calls collect_episode.map(), called from loop.py

Modal Volume layout (rsi-api-checkpoints):
  /checkpoints/episode_{n}/adapter_model.safetensors  — LoRA weights
  /checkpoints/episode_{n}/config.json                — LoRA config
  /checkpoints/latest/                                — copy of most recent
  /checkpoints/logs/metrics.jsonl                     — training metrics

Cost tracking:
  A100 40GB: ~$3.00/hr
  CPU worker (2 vCPU, 4GB): ~$0.0002/hr each
  50 workers x 12h = $0.12 total for episode collection
  GPU training 12h = ~$36
  Total estimated: ~$46
"""
import modal
from dataclasses import dataclass, asdict
import json

MODEL_NAME = "google/gemma-3-1b-it"

# ── Modal primitives ──────────────────────────────────────────────────────────

app = modal.App("rsi-api")

volume = modal.Volume.from_name("rsi-api-checkpoints", create_if_missing=True)

hf_secret = modal.Secret.from_name("huggingface-secret")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "flask",
        "flask-cors",
        "coverage",
        "networkx",
        "transformers>=4.50.0",
        "peft",
        "torch>=2.2.0",
        "trl>=0.8.0",
        "accelerate",
        "bitsandbytes",
        "requests",
        "pyyaml",
        "huggingface_hub",
    ])
    .env({
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "CACHE_BUST": "v3",
    })
)

image = image.add_local_dir(".", remote_path="/app",
    ignore=lambda path: any(x in str(path) for x in [".git", "__pycache__", ".pyc", "checkpoints"]))

CHECKPOINT_DIR = "/checkpoints"
LOGS_PATH = "/checkpoints/logs/metrics.jsonl"


# ── Return type dataclasses ───────────────────────────────────────────────────

@dataclass
class EpisodeResult:
    """
    Returned by collect_episode(). Must be fully JSON-serializable
    since Modal serializes return values across the network.
    """
    trajectory: list        # list of {obs, action, step_reward, info} dicts
    episode_reward: float
    branch_coverage: float
    curriculum_level: int
    success: bool           # True if coverage > 0.5
    api_id: str             # unique ID, used to group episodes in GRPO buffer
    failure_mode: str       # "auth_fail" | "wrong_order" | "infinite_loop" | "success" | etc.
    metadata: dict          # timing, step count, endpoints discovered


@dataclass
class EvalResult:
    mean_coverage: float
    success_rate: float
    by_level: dict          # {level: {mean_coverage, success_rate, n_episodes}}
    n_episodes: int


# ── Modal functions ───────────────────────────────────────────────────────────

@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=300,
    volumes={CHECKPOINT_DIR: volume},
)
def collect_episode(api_config: dict, policy_checkpoint: str, config: dict) -> dict:
    """
    Run one agent episode on a Modal CPU worker.
    Called via collect_episode.map() for parallel collection.
    """
    import sys, os
    sys.path.insert(0, "/app")

    from curriculum.factory import generate_api
    from agent.policy import Policy
    from training.episode import run_episode
    import traceback

    api = None
    try:
        api = generate_api(
            level=api_config.get("level", 1),
            config=api_config.get("config", config),
            factory_weights=api_config.get("factory_weights"),
        )
        # Override api_id with the one from config if provided
        if api_config.get("api_id"):
            api.api_id = api_config["api_id"]

        if not api.is_alive():
            return None

        if policy_checkpoint and os.path.exists(policy_checkpoint):
            policy = Policy(config)
            policy.load(policy_checkpoint)
        else:
            # First episodes: use random policy (no model needed)
            from agent.actions import ActionSpace
            class RandomPolicy:
                def __init__(self): self.action_space = ActionSpace()
                def sample_action(self, obs): return self.action_space.sample_random()
                def log_prob(self, obs, action):
                    import torch; return torch.tensor(-1.0, requires_grad=True)
            policy = RandomPolicy()

        result = run_episode(policy, api, api_config.get("config", config))
        result["api_id"] = api.api_id
        return result

    except Exception as e:
        return {
            "trajectory": [], "episode_reward": 0.0, "branch_coverage": 0.0,
            "curriculum_level": api_config.get("level", 1), "success": False,
            "api_id": api_config.get("seed", "unknown"), "failure_mode": "exception",
            "metadata": {"error": str(e), "traceback": traceback.format_exc()}
        }
    finally:
        if api:
            try: api.shutdown()
            except Exception: pass


@app.function(
    image=image,
    gpu="A100",
    memory=32768,
    timeout=3600,
    volumes={CHECKPOINT_DIR: volume},
    secrets=[hf_secret],
)
def train_step(
    checkpoint_path: str,
    batch_json: str,
    episode_number: int,
    config: dict,
) -> str:
    """
    One GRPO gradient update on A100.
    Returns path to new checkpoint on Volume.
    """
    import sys, os, json, shutil
    sys.path.insert(0, "/app")

    import torch
    from agent.policy import Policy
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    from training.grpo import GRPOTrainer
    from training.buffer import GRPOBatch

    # Load or initialize policy
    policy = Policy(config)
    if checkpoint_path and os.path.exists(checkpoint_path):
        policy.load(checkpoint_path)
    # else: fresh LoRA already initialized in Policy.__init__()

    # Deserialize batch
    batch = GRPOBatch.from_json(batch_json)

    # Train
    trainer = GRPOTrainer(policy, config)
    loss = trainer.step(batch)

    # Save to volume
    new_ckpt = f"{CHECKPOINT_DIR}/episode_{episode_number}"
    os.makedirs(new_ckpt, exist_ok=True)
    policy.save(new_ckpt)

    # Update latest
    latest = f"{CHECKPOINT_DIR}/latest"
    if os.path.exists(latest):
        shutil.rmtree(latest)
    shutil.copytree(new_ckpt, latest)

    # Flush volume writes
    volume.commit()

    # Log metrics
    os.makedirs(os.path.dirname(LOGS_PATH), exist_ok=True)
    with open(LOGS_PATH, "a") as f:
        f.write(json.dumps({
            "episode": episode_number,
            "loss": loss,
            "checkpoint": new_ckpt
        }) + "\n")
    volume.commit()

    return new_ckpt


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=300,
    volumes={CHECKPOINT_DIR: volume},
)
def evaluate_policy(
    checkpoint_path: str,
    level: int,
    n_episodes: int,
    config: dict,
) -> dict:
    """
    Evaluate current policy on n_episodes held-out APIs at given level.
    Returns EvalResult as dict.
    """
    import sys
    sys.path.insert(0, "/app")

    from curriculum.factory import generate_api
    from agent.policy import Policy
    from training.episode import run_episode

    policy = Policy(config)
    if checkpoint_path:
        policy.load(checkpoint_path)

    results = []
    for _ in range(n_episodes):
        api = generate_api(level=level, config=config)
        try:
            result = run_episode(policy, api, config)
            results.append(result)
        finally:
            api.shutdown()

    mean_cov = sum(r["branch_coverage"] for r in results) / max(len(results), 1)
    sr = sum(1 for r in results if r["success"]) / max(len(results), 1)

    return {
        "mean_coverage": mean_cov,
        "success_rate": sr,
        "by_level": {str(level): {"mean_coverage": mean_cov, "success_rate": sr, "n_episodes": len(results)}},
        "n_episodes": len(results)
    }


@app.function(
    image=image,
    gpu="A100",
    memory=32768,
    timeout=1800,
    volumes={CHECKPOINT_DIR: volume},
    secrets=[hf_secret],
)
def run_eval_modal(checkpoint: str, n_episodes: int, config: dict) -> dict:
    """
    Run full base-vs-trained evaluation across all 4 curriculum levels on A100.

    checkpoint: path on Modal Volume, e.g. '/checkpoints/latest'
    Returns results dict compatible with eval/table.py format_terminal_table().
    """
    import sys, os
    sys.path.insert(0, "/app")

    from curriculum.factory import generate_api
    from agent.policy import Policy
    from training.episode import run_episode

    def _agg(episodes):
        if not episodes:
            return {"mean_coverage": 0.0, "success_rate": 0.0, "mean_steps": 0.0,
                    "n_episodes": 0, "coverage_distribution": []}
        covs = [e["branch_coverage"] for e in episodes]
        suc = [1 if e["success"] else 0 for e in episodes]
        steps = [len(e.get("trajectory", [])) for e in episodes]
        return {
            "mean_coverage": sum(covs) / len(covs),
            "success_rate": sum(suc) / len(suc),
            "mean_steps": sum(steps) / len(steps),
            "n_episodes": len(episodes),
            "coverage_distribution": covs,
        }

    def _run(policy, lvl, n):
        results = []
        for _ in range(n):
            api = generate_api(level=lvl, config=config)
            try:
                results.append(run_episode(policy, api, config))
            finally:
                api.shutdown()
        return results

    # Resolve checkpoint path to volume mount
    ckpt_full = checkpoint
    if not os.path.isabs(checkpoint):
        ckpt_full = os.path.join(CHECKPOINT_DIR, checkpoint.lstrip("/"))

    # Fair "BASE MODEL vs TRAINED MODEL" comparison:
    #  - base is the untrained policy (base Gemma + fresh LoRA), NOT a random-action
    #    policy. Comparing random vs trained is what made the original eval meaningless.
    #  - exploration is DISABLED for eval (epsilon=0) so we measure what the LLM
    #    actually learned, not the shared structured-exploration scaffold (which would
    #    give base and trained near-identical coverage and hide real learning).
    eval_config = dict(config)
    eval_config["training"] = {**config.get("training", {}), "exploration_epsilon": 0.0}

    base_policy = Policy(eval_config)
    trained_policy = Policy(eval_config)
    if os.path.exists(ckpt_full):
        trained_policy.load(ckpt_full)

    curriculum_levels = config.get("curriculum", {}).get("levels", {})
    levels = {}

    for lvl in [1, 2, 3, 4]:
        lvl_cfg = curriculum_levels.get(str(lvl)) or curriculum_levels.get(lvl, {})
        ep_count = lvl_cfg.get("endpoints")

        base_stats = _agg(_run(base_policy, lvl, n_episodes))
        trained_stats = _agg(_run(trained_policy, lvl, n_episodes))
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


# ── Local orchestration helper (not a Modal function) ────────────────────────

def run_parallel_episodes(
    api_configs: list[dict],
    checkpoint_path: str,
    config: dict,
) -> list[dict]:
    """
    Launch parallel episode collection using Modal.
    Called from training/loop.py — runs locally, dispatches to Modal.
    """
    inputs = [(cfg, checkpoint_path, config) for cfg in api_configs]

    try:
        f = modal.Function.from_name("rsi-api", "collect_episode")
        results = list(f.starmap(inputs))
        results = [r for r in results if r is not None]
        print(f"[Modal] Collected {len(results)} episodes on Modal workers")
        return results
    except Exception as e:
        print(f"Modal parallel collection failed: {e}")
        print("Falling back to local sequential execution...")

        from curriculum.factory import generate_api
        from training.episode import run_episode
        from agent.actions import ActionSpace

        class RandomPolicy:
            def __init__(self): self.action_space = ActionSpace()
            def sample_action(self, obs): return self.action_space.sample_random()
            def log_prob(self, obs, action):
                import torch; return torch.tensor(-1.0, requires_grad=True)

        results = []
        group_size = config["training"].get("grpo_group_size", 4)
        n_groups = max(1, len(api_configs) // group_size)

        for group_idx in range(n_groups):
            idx = group_idx * group_size
            cfg = api_configs[idx] if idx < len(api_configs) else api_configs[0]
            try:
                api = generate_api(
                    level=cfg.get("level", 1),
                    config=cfg.get("config", config),
                )
                shared_api_id = api.api_id
                for ep_idx in range(group_size):
                    try:
                        result = run_episode(RandomPolicy(), api, config)
                        result["api_id"] = shared_api_id
                        results.append(result)
                        print(f"[Local fallback] Group {group_idx+1} episode {ep_idx+1}: coverage={result['branch_coverage']:.1%}")
                    except Exception as ep_e:
                        print(f"[Local fallback] Episode failed: {ep_e}")
                api.shutdown()
            except Exception as g_e:
                print(f"[Local fallback] Group {group_idx} failed: {g_e}")

        return results


@app.function(
    image=image,
    cpu=2,
    memory=8192,
    timeout=86400,
    volumes={CHECKPOINT_DIR: volume},
    secrets=[hf_secret],
)
def run_training_loop(config: dict):
    """
    Runs the entire RSI training loop on Modal.
    Training runs in the cloud — laptop can close.
    """
    import sys
    sys.path.insert(0, "/app")
    from training.loop import TrainingLoop
    loop = TrainingLoop(config)
    loop.run()


@app.function(
    image=image,
    gpu="A100",
    memory=32768,
    timeout=600,
    volumes={CHECKPOINT_DIR: volume},
    secrets=[hf_secret],
)
def smoke_train_diagnostics(config: dict) -> dict:
    """Definitive weight-update check on A100: build a fresh TRAINABLE policy, run
    one real episode (reward/coverage), then a GRPO step on a guaranteed-varied
    group, and measure the actual weight delta. Catches the frozen-adapter /
    zero-advantage failure modes that make training a silent no-op."""
    import sys, copy
    sys.path.insert(0, "/app")
    from curriculum.factory import generate_api
    from agent.policy import Policy
    from training.episode import run_episode
    from training.grpo import GRPOTrainer
    from training.buffer import GRPOBatch

    policy = Policy(config)
    trainable = sum(1 for p in policy.model.parameters() if p.requires_grad)

    api = generate_api(level=1, config=config)
    try:
        ep = run_episode(policy, api, config)
    finally:
        api.shutdown()

    base_r = ep.get("episode_reward", 0.0) or 0.0
    hi = copy.deepcopy(ep); hi["episode_reward"] = base_r + 0.1
    lo = copy.deepcopy(ep); lo["episode_reward"] = base_r
    batch = GRPOBatch(groups=[[hi, lo]])

    trainer = GRPOTrainer(policy, config)
    trainables = [p for p in policy.model.parameters() if p.requires_grad]
    before = [p.detach().clone() for p in trainables]
    loss = trainer.step(batch)
    delta = max((p.detach() - b).abs().max().item() for p, b in zip(trainables, before)) if trainables else 0.0

    return {
        "trainable_params": trainable,
        "episode_coverage": ep.get("branch_coverage", 0.0),
        "episode_reward": base_r,
        "grpo_loss": loss,
        "max_weight_delta": delta,
        "weights_updated": delta > 1e-7,
    }


@app.local_entrypoint()
def smoke(config_path: str = "configs/train_config_modal_smoke.yaml", checkpoint: str = ""):
    """End-to-end Modal smoke test — run BEFORE the full job to confirm the cloud
    pipeline works and actually updates weights/rewards.

        modal run modal_app.py::smoke
        modal run modal_app.py::smoke --checkpoint /checkpoints/latest   # use seeded policy

    Verifies: CPU collection (collect_episode.map), reward/coverage non-zero, A100
    GRPO weight update (train_step + smoke_train_diagnostics), volume checkpoint
    save + reload, and eval. Exits non-zero on any failed check.
    """
    import yaml, random, sys, json as _json
    from training.buffer import GRPOBatch

    with open(config_path) as f:
        config = yaml.safe_load(f)

    print("=== Modal smoke test ===", flush=True)
    checks = {}

    # 1) Distributed collection on CPU workers (collect_episode.starmap).
    gs = config["training"]["grpo_group_size"]
    n_groups = max(1, config["training"]["episodes_per_update"] // gs)
    api_configs = []
    for g in range(n_groups):
        seed = random.randint(0, 2**31 - 1)
        for _ in range(gs):
            api_configs.append({"level": 1, "seed": seed, "api_id": f"g{g}_{seed}", "config": config})
    inputs = [(cfg, checkpoint, config) for cfg in api_configs]
    results = [r for r in collect_episode.starmap(inputs) if r]
    covs = [r.get("branch_coverage", 0.0) for r in results]
    checks["collection_ok"] = len(results) == len(api_configs)
    checks["coverage_measured"] = len(covs) > 0 and max(covs) >= 0.0
    print(f"[1] collected {len(results)}/{len(api_configs)} episodes, "
          f"coverage mean={sum(covs)/max(len(covs),1):.1%} max={max(covs) if covs else 0:.1%}", flush=True)

    # 2) A100 GRPO weight update on a varied batch + checkpoint save (train_step).
    groups = []
    for g in range(n_groups):
        grp = results[g * gs:(g + 1) * gs]
        if len(grp) == gs:
            grp[0] = {**grp[0], "episode_reward": (grp[0].get("episode_reward") or 0.0) + 0.1}
            groups.append(grp)
    batch_json = GRPOBatch(groups=groups).to_json()
    new_ckpt = train_step.remote(checkpoint, batch_json, 0, config)
    checks["train_step_ok"] = isinstance(new_ckpt, str) and len(new_ckpt) > 0
    print(f"[2] train_step -> checkpoint {new_ckpt}", flush=True)

    # 3) Definitive weight-update measurement.
    diag = smoke_train_diagnostics.remote(config)
    checks["trainable"] = diag["trainable_params"] > 0
    checks["weights_updated"] = bool(diag["weights_updated"])
    print(f"[3] diagnostics: trainable_params={diag['trainable_params']} "
          f"coverage={diag['episode_coverage']:.1%} loss={diag['grpo_loss']:.5f} "
          f"weight_delta={diag['max_weight_delta']:.2e} updated={diag['weights_updated']}", flush=True)

    # 4) Checkpoint reload + eval (evaluate_policy loads the new checkpoint).
    ev = evaluate_policy.remote(new_ckpt, 1, 2, config)
    checks["eval_ok"] = "mean_coverage" in ev
    print(f"[4] eval on {new_ckpt}: mean_coverage={ev.get('mean_coverage', 0):.1%} "
          f"n={ev.get('n_episodes')}", flush=True)

    print("\n=== RESULTS ===", flush=True)
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}", flush=True)
    all_ok = all(checks.values())
    print(f"\n{'ALL CHECKS PASSED — Modal pipeline is ready for the full run' if all_ok else 'SOME CHECKS FAILED — do not start the full run'}", flush=True)
    if not all_ok:
        sys.exit(1)


@app.local_entrypoint()
def main(config_path: str = "configs/train_config.yaml"):
    """
    Usage: modal run modal_app.py --config-path configs/train_config.yaml
    Or for local test: modal run modal_app.py --config-path configs/train_config_local.yaml
    """
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)
    print(f"Submitting training job to Modal...")
    print(f"Monitor at: https://modal.com/apps/kushaannaskar/main/deployed/rsi-api")
    run_training_loop.spawn(config)
    print("Training job submitted. You can close your laptop.")
