"""
Reward composition.

Episode reward: R = syntax_valid * (branch_coverage - destructive_penalty)
syntax_valid is a HARD GATE — invalid Python returns exactly 0.0.
Step reward: shaped intermediates applied at each step.
"""
import ast
from dataclasses import dataclass
from verifier.coverage_runner import CoverageResult


@dataclass
class StepInfo:
    new_endpoints_discovered: int = 0
    auth_token_acquired: bool = False
    request_was_repeated: bool = False
    hints_extracted: int = 0
    destructive_call_out_of_order: bool = False


def is_syntax_valid(client_code: str) -> bool:
    try:
        ast.parse(client_code)
        return True
    except SyntaxError:
        return False


def compute_step_reward(step_info: StepInfo, config: dict) -> float:
    reward_cfg = config["reward"] if "reward" in config else config
    r = 0.0
    r += reward_cfg["new_endpoint_bonus"] * step_info.new_endpoints_discovered
    if step_info.auth_token_acquired:
        r += reward_cfg["auth_token_bonus"]
    if step_info.request_was_repeated:
        r -= reward_cfg["repeated_request_penalty"]
    r += 0.1 * step_info.hints_extracted
    return r


def compute_episode_reward(
    coverage_result: CoverageResult,
    client_code: str,
    trajectory: list,
    config: dict,
) -> float:
    if not is_syntax_valid(client_code):
        return 0.0
    reward_cfg = config["reward"] if "reward" in config else config
    destructive_penalty = reward_cfg["destructive_penalty"] if any(
        step.get("step_info") and step["step_info"].get("destructive_call_out_of_order", False)
        for step in trajectory
        if isinstance(step, dict)
    ) else 0.0
    return coverage_result.branch_coverage - destructive_penalty
