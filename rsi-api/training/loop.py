"""
Main RSI training orchestrator. Runs locally, dispatches to Modal.
No modal import here — Modal is accessed via modal_app.py imports.
"""
import json
import random
import modal
from curriculum.factory import generate_api
from curriculum.scheduler import DifficultyScheduler
from curriculum.failure_analyzer import analyze
from verifier.sandbox import run_parallel_episodes
from training.buffer import RolloutBuffer, GRPOBatch
from agent.policy import Policy


class TrainingLoop:
    def __init__(self, config: dict):
        self.config = config
        self.scheduler = DifficultyScheduler(config["curriculum"])
        self.buffer = RolloutBuffer(config)
        self.episode_count = 0
        self.current_checkpoint = None    # Modal Volume path to latest checkpoint

        # Auto-resume from latest checkpoint on Modal Volume if it exists
        import os
        latest_ckpt = "/checkpoints/latest"
        if os.path.exists(latest_ckpt) and os.path.exists(
            os.path.join(latest_ckpt, "adapter_model.safetensors")
        ):
            self.current_checkpoint = latest_ckpt
            print(f"[Resume] Auto-resuming from {latest_ckpt}")
        else:
            print("[Resume] No checkpoint found, starting from scratch")

        # Try to read last episode count from metrics log
        metrics_log = "/checkpoints/logs/metrics.jsonl"
        if os.path.exists(metrics_log):
            try:
                with open(metrics_log) as f:
                    lines = [l for l in f.readlines() if l.strip()]
                if lines:
                    last = json.loads(lines[-1])
                    self.episode_count = last.get("episode", 0)
                    print(f"[Resume] Resuming from episode {self.episode_count}")
            except Exception:
                pass

    def run(self):
        """
        Main loop until config["training"]["total_episodes"] reached.
        """
        current_factory_weights = None
        total_episodes = self.config["training"]["total_episodes"]
        n_workers = self.config["modal"]["n_episode_workers"]
        checkpoint_every = self.config["training"]["checkpoint_every"]

        while self.episode_count < total_episodes:
            difficulty = self.scheduler.current_difficulty()

            group_size = self.config["training"]["grpo_group_size"]
            n_groups = max(1, n_workers // group_size)

            api_configs = []
            for group_idx in range(n_groups):
                # Same seed for all episodes in a group = same API structure
                group_seed = random.randint(0, 2**32 - 1)
                group_api_id = f"group_{group_idx}_{group_seed}"
                for _ in range(group_size):
                    api_configs.append({
                        "level": difficulty.level,
                        "factory_weights": current_factory_weights,
                        "seed": group_seed,          # SAME seed = same API structure
                        "api_id": group_api_id,      # SAME api_id for all in group
                        "config": self.config,
                    })

            results = run_parallel_episodes(
                api_configs,
                self.current_checkpoint,
                self.config,
            )

            for result in results:
                print(f"[Buffer] Adding episode api_id={result.get('api_id', 'MISSING')}")
                self.buffer.add_episode(result)
                print(f"[Buffer] Total collected: {self.buffer.total_collected}, Ready: {self.buffer.is_ready()}")
                self.scheduler.record_episode(result.get("success", False))
            self.scheduler.step()

            print(f"[Buffer] Checking readiness: {self.buffer.total_collected} episodes, by_api keys: {len(self.buffer.by_api)}")
            if self.buffer.is_ready():
                batch = self.buffer.get_batch()
                batch_json = batch.to_json()
                print(f"[Training] Buffer ready. Calling train_step on Modal A100...")
                train_step_fn = modal.Function.from_name("rsi-api", "train_step")
                call = train_step_fn.spawn(
                    self.current_checkpoint or "",
                    batch_json,
                    self.episode_count,
                    self.config,
                )
                new_ckpt = call.get()
                self.current_checkpoint = new_ckpt
                print(f"[Training] train_step complete. New checkpoint: {new_ckpt}")

            self.episode_count += len(results)

            if self.episode_count % checkpoint_every < n_workers:
                mean_cov = sum(r.get("branch_coverage", 0) for r in results) / max(len(results), 1)
                sr = sum(1 for r in results if r.get("success")) / max(len(results), 1)
                self._log_metrics({
                    "mean_coverage": mean_cov,
                    "success_rate": sr,
                    "curriculum_level": difficulty.level,
                    "checkpoint": self.current_checkpoint,
                })

            failure_dist = analyze(results)
            current_factory_weights = failure_dist.to_factory_weights()

    def load_checkpoint(self, path: str):
        """Set self.current_checkpoint = path (Modal Volume path)."""
        self.current_checkpoint = path

    def _log_metrics(self, metrics: dict):
        """Print metrics to stdout. loop.py does not write to disk — train_step owns logging."""
        print(json.dumps({"episode": self.episode_count, **metrics}))
