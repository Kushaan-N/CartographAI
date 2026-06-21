"""
Clusters failed episodes by failure mode.
Feeds failure distribution back to factory as sampling weights.

Failure modes:
- auth_fail: agent never acquired any auth token
- wrong_order: 403 received after 2xx on a dependency endpoint
- missing_header: repeated 4xx with same endpoint suggests missed header
- infinite_loop: same (method, endpoint) tuple appears >3 times in trajectory
- red_herring_trap: >50% of steps spent on endpoints with zero coverage contribution
- coverage_plateau: coverage > 0 but stopped improving after step 10
"""
from dataclasses import dataclass, field


@dataclass
class FailureDistribution:
    auth_fail_rate: float = 0.0
    wrong_order_rate: float = 0.0
    missing_header_rate: float = 0.0
    infinite_loop_rate: float = 0.0
    red_herring_trap_rate: float = 0.0
    coverage_plateau_rate: float = 0.0
    api_locked_rate: float = 0.0
    n_episodes: int = 0

    def to_factory_weights(self) -> dict:
        """
        Convert failure rates to factory pattern sampling weights.
        Higher failure rate on a pattern -> higher weight in next batch.
        Normalize so weights sum to 1.0.
        Returns dict with same keys as failure fields.
        """
        rates = {
            "auth_fail": self.auth_fail_rate,
            "wrong_order": self.wrong_order_rate,
            "missing_header": self.missing_header_rate,
            "infinite_loop": self.infinite_loop_rate,
            "red_herring_trap": self.red_herring_trap_rate,
            "coverage_plateau": self.coverage_plateau_rate,
            "api_locked": self.api_locked_rate * 2.0,  # 2x weight: catastrophic reckless behavior
        }
        total = sum(rates.values())
        if total == 0.0:
            n = len(rates)
            return {k: 1.0 / n for k in rates}
        return {k: v / total for k, v in rates.items()}


def analyze(trajectories: list[dict]) -> FailureDistribution:
    """
    Analyze a batch of EpisodeResult dicts and return FailureDistribution.
    """
    n = len(trajectories)
    if n == 0:
        return FailureDistribution(n_episodes=0)

    auth_fail = 0
    infinite_loop = 0
    wrong_order = 0
    missing_header = 0
    red_herring_trap = 0
    coverage_plateau = 0
    api_locked = 0

    for result in trajectories:
        meta = result.get("metadata", {})

        if meta.get("auth_tokens_acquired", 0) == 0 and not result["success"]:
            auth_fail += 1

        if result["failure_mode"] == "infinite_loop":
            infinite_loop += 1

        if result["failure_mode"] == "wrong_order":
            wrong_order += 1

        if result["failure_mode"] == "missing_header":
            missing_header += 1

        total_steps = max(meta.get("total_steps", 1), 1)
        rh_steps = meta.get("red_herring_steps", 0)
        if rh_steps / total_steps > 0.5:
            red_herring_trap += 1

        if result["branch_coverage"] > 0 and result["branch_coverage"] < 0.3 and not result["success"]:
            coverage_plateau += 1

        if meta.get("api_locked", False):
            api_locked += 1

    return FailureDistribution(
        auth_fail_rate=auth_fail / n,
        wrong_order_rate=wrong_order / n,
        missing_header_rate=missing_header / n,
        infinite_loop_rate=infinite_loop / n,
        red_herring_trap_rate=red_herring_trap / n,
        coverage_plateau_rate=coverage_plateau / n,
        api_locked_rate=api_locked / n,
        n_episodes=n,
    )
