"""
Zone of Proximal Development difficulty gate.

Advances curriculum level when rolling success rate > 0.70.
Retreats when rolling success rate < 0.40.
Rolling window default: 30 episodes.
"""
from collections import deque
from dataclasses import dataclass


@dataclass
class DifficultyConfig:
    level: int
    endpoints: int
    auth_schemes: int
    dependencies: int
    red_herrings: int


class DifficultyScheduler:
    def __init__(self, config: dict):
        self.config = config
        self.level = 1
        self.window = deque(maxlen=config["rolling_window"])
        self.advance_threshold = config["zpd_advance_threshold"]
        self.retreat_threshold = config["zpd_retreat_threshold"]

    def record_episode(self, success: bool):
        self.window.append(bool(success))

    def current_success_rate(self) -> float:
        if not self.window:
            return 0.0
        return sum(self.window) / len(self.window)

    def current_difficulty(self) -> DifficultyConfig:
        levels = self.config["levels"]
        max_level = max(int(k) for k in levels)
        level = min(self.level, max_level)
        cfg = levels[str(level)] if str(level) in levels else levels[level]
        return DifficultyConfig(
            level=level,
            endpoints=cfg["endpoints"],
            auth_schemes=cfg["auth_schemes"],
            dependencies=cfg["dependencies"],
            red_herrings=cfg["red_herrings"],
        )

    def step(self) -> bool:
        levels = self.config["levels"]
        max_level = max(int(k) for k in levels)
        rate = self.current_success_rate()
        changed = False
        if rate > self.advance_threshold and self.level < max_level:
            self.level += 1
            self.window.clear()
            changed = True
        elif rate < self.retreat_threshold and self.level > 1:
            self.level -= 1
            self.window.clear()
            changed = True
        return changed

    def state_dict(self) -> dict:
        return {"level": self.level, "window": list(self.window)}

    def load_state_dict(self, state: dict):
        self.level = state["level"]
        self.window = deque(state["window"], maxlen=self.config["rolling_window"])
