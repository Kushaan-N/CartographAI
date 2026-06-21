"""
Thin wrapper around modal_app.run_parallel_episodes.
Guards the import so local testing works without Modal credentials.
"""
try:
    from modal_app import run_parallel_episodes
except ImportError:
    def run_parallel_episodes(api_configs, checkpoint_path, config):
        """
        Local fallback when Modal is not available.
        Runs episodes sequentially using ThreadPoolExecutor.
        Used for local testing only — use Modal for real training.
        """
        import sys
        sys.path.insert(0, '.')
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from curriculum.factory import generate_api
        from training.episode import run_episode
        from agent.actions import ActionSpace

        class RandomPolicy:
            def __init__(self): self.action_space = ActionSpace()
            def sample_action(self, obs): return self.action_space.sample_random()
            def log_prob(self, obs, action):
                import torch; return torch.tensor(-1.0, requires_grad=True)

        results = []
        max_workers = min(4, len(api_configs))

        def run_one(cfg):
            api = generate_api(level=cfg.get("level", 1), config=cfg.get("config", config))
            try:
                result = run_episode(RandomPolicy(), api, config)
                result["api_id"] = api.api_id
                return result
            except Exception as e:
                return {
                    "trajectory": [], "episode_reward": 0.0,
                    "branch_coverage": 0.0, "success": False,
                    "failure_mode": "exception", "api_id": "unknown",
                    "metadata": {"error": str(e), "total_steps_taken": 0,
                                "bonus_steps_granted": 0, "bonus_log": [],
                                "auth_tokens_acquired": 0, "endpoints_discovered": 0,
                                "api_locked": False, "consecutive_4xx_peak": 0}
                }
            finally:
                api.shutdown()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_one, cfg) for cfg in api_configs]
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"Worker failed: {e}")

        return results

__all__ = ["run_parallel_episodes"]
