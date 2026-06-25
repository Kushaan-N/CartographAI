"""
Single episode rollout logic.
Called inside collect_episode() Modal function AND locally during demo.
No Modal imports here.
"""
import requests
import time
import re
import json as json_lib
from agent.memory import WorkingMemory
from agent.actions import ActionSpace, Action
from agent.error_forensics import extract_hints, apply_hints
from verifier.reward import compute_step_reward, compute_episode_reward, StepInfo
from verifier.coverage_runner import instrument


def run_episode(policy, api, config: dict) -> dict:
    memory = WorkingMemory()
    action_space = ActionSpace()
    trajectory = []
    fingerprints = set()
    last_response = {"status_code": 0, "headers": {}, "body": ""}

    # Seed initial hypotheses
    action_space.expand_endpoints(["/", "/health", "/api", "/status", "/login", "/auth"])
    for ep in action_space.known_endpoints:
        memory.add_hypothesis(ep)

    # Dynamic budget tracking
    base_budget = config["training"]["max_episode_steps"]
    bonus_steps_per_auth = config["training"].get("bonus_steps_per_auth", 2)
    bonus_steps_per_dep = config["training"].get("bonus_steps_per_dependency", 2)
    max_total_steps = config["training"].get("max_total_steps", 35)

    budget_remaining = base_budget
    total_steps_taken = 0
    auth_tokens_at_last_check = 0
    dependencies_at_last_check = 0
    bonus_log = []

    # Fragility tracking
    consecutive_4xx = 0
    fragility_threshold = config["training"].get("fragility_threshold", 5)
    fragility_warning_threshold = config["training"].get("fragility_warning_threshold", 3)
    fragility_penalty = config["training"].get("fragility_penalty", -1.0)
    api_locked = False

    while budget_remaining > 0 and total_steps_taken < max_total_steps:
        step = total_steps_taken

        obs = build_observation(memory, last_response, step)
        action = policy.sample_action(obs)

        fingerprint = action.fingerprint()
        repeated = fingerprint in fingerprints
        fingerprints.add(fingerprint)

        response = execute_request(action, api.url)
        last_response = response

        status = response["status_code"]
        if 200 <= status < 300:
            memory.update_node(
                action.endpoint, "discovered",
                last_status_code=status,
                last_method=action.method,
            )
            body_str = json_lib.dumps(response["body"]) if isinstance(response["body"], dict) else str(response["body"])
            new_paths = re.findall(r'["\'](/[a-zA-Z0-9_/.-]+)["\']', body_str)
            for path in new_paths:
                if path not in list(memory.G.nodes):
                    memory.add_hypothesis(path)
                    action_space.expand_endpoints([path])

            token_keys = ["token", "access_token", "api_key", "key", "auth_token", "bearer"]
            body_dict = response["body"] if isinstance(response["body"], dict) else {}
            for k, v in body_dict.items():
                if any(tk in k.lower() for tk in token_keys) and isinstance(v, str):
                    memory.store_auth_token(action.endpoint, v)
        else:
            memory.update_node(
                action.endpoint, "failed",
                last_status_code=status,
                last_method=action.method,
            )

        # Fragility mechanic
        if 400 <= status < 500:
            consecutive_4xx += 1
        else:
            consecutive_4xx = 0

        if consecutive_4xx >= fragility_warning_threshold:
            last_response["_fragility_warning"] = True
            last_response["_consecutive_4xx"] = consecutive_4xx

        if consecutive_4xx >= fragility_threshold:
            api_locked = True
            trajectory.append({
                "obs": obs,
                "action": {"method": action.method, "endpoint": action.endpoint,
                           "headers": action.headers, "body": action.body},
                "step_reward": fragility_penalty,
                "response": {"status_code": response["status_code"], "body_preview": str(response["body"])[:100]},
                "step_info": {"api_locked": True, "consecutive_4xx": consecutive_4xx},
                "budget_remaining": 0,
                "bonus_granted": False,
            })
            total_steps_taken += 1
            break

        hints = extract_hints(response)
        hint_bonus = apply_hints(hints, memory, action_space)

        current_auth_count = len(memory.auth_tokens)
        current_dep_count = len(memory.G.edges())

        new_auth_acquired = current_auth_count > auth_tokens_at_last_check
        new_dep_discovered = current_dep_count > dependencies_at_last_check

        n_discovered = len([n for n in memory.G.nodes if memory.G.nodes[n].get("status") == "discovered"])
        step_info = StepInfo(
            new_endpoints_discovered=max(0, n_discovered - step),
            auth_token_acquired=new_auth_acquired,
            request_was_repeated=repeated,
            hints_extracted=len(hints),
            destructive_call_out_of_order=False,
        )

        step_reward = compute_step_reward(step_info, config) + hint_bonus

        if new_auth_acquired:
            budget_remaining += bonus_steps_per_auth
            bonus_log.append(f"step {step}: +{bonus_steps_per_auth} steps (auth token acquired from {action.endpoint})")
            auth_tokens_at_last_check = current_auth_count

        if new_dep_discovered:
            budget_remaining += bonus_steps_per_dep
            bonus_log.append(f"step {step}: +{bonus_steps_per_dep} steps (new dependency discovered)")
            dependencies_at_last_check = current_dep_count

        trajectory.append({
            "obs": obs,
            "action": {"method": action.method, "endpoint": action.endpoint,
                       "headers": action.headers, "body": action.body},
            "step_reward": step_reward,
            "response": {
                "status_code": response["status_code"],
                "body_preview": str(response["body"])[:100],
            },
            "step_info": {
                "new_endpoints_discovered": step_info.new_endpoints_discovered,
                "auth_token_acquired": step_info.auth_token_acquired,
                "request_was_repeated": step_info.request_was_repeated,
                "hints_extracted": step_info.hints_extracted,
            },
            "budget_remaining": budget_remaining,
            "bonus_granted": new_auth_acquired or new_dep_discovered,
        })

        budget_remaining -= 1
        total_steps_taken += 1
        memory.step_count += 1

    client_code = generate_client(memory, api.url)
    instr = instrument(api.source_code, api.port)
    cov_result = instr.run_client(client_code)
    instr.shutdown()

    episode_reward = compute_episode_reward(cov_result, client_code, trajectory, config)
    success = cov_result.branch_coverage > 0.5
    failure_mode = classify_failure(trajectory, memory)

    if api_locked:
        episode_reward = fragility_penalty
        failure_mode = "api_locked"
        success = False

    return {
        "trajectory": trajectory,
        "episode_reward": episode_reward,
        "branch_coverage": cov_result.branch_coverage,
        "curriculum_level": config.get("_current_level", 1),
        "success": success,
        "failure_mode": failure_mode,
        "api_id": api.api_id,
        "metadata": {
            "total_steps_taken": total_steps_taken,
            "base_budget": base_budget,
            "bonus_steps_granted": total_steps_taken - base_budget if total_steps_taken > base_budget else 0,
            "bonus_log": bonus_log,
            "auth_tokens_acquired": len(memory.auth_tokens),
            "endpoints_discovered": len([n for n in memory.G.nodes if memory.G.nodes[n].get("status") == "discovered"]),
            "red_herring_steps": 0,
            "total_steps": total_steps_taken,
            "api_locked": api_locked,
            "consecutive_4xx_peak": consecutive_4xx,
            "fragility_triggered": api_locked,
        },
    }


def build_observation(memory: WorkingMemory, last_response: dict, step: int) -> dict:
    obs = memory.to_observation()
    obs["last_response"] = last_response or {"status_code": 0, "headers": {}, "body": ""}
    obs["episode_step"] = step
    obs["fragility_warning"] = last_response.get("_fragility_warning", False)
    obs["consecutive_4xx"] = last_response.get("_consecutive_4xx", 0)
    return obs


def execute_request(action: Action, base_url: str) -> dict:
    try:
        kwargs = action.to_request_kwargs(base_url)
        resp = requests.request(timeout=3, **kwargs)
        try:
            body = resp.json()
        except Exception:
            body = resp.text
        return {
            "status_code": resp.status_code,
            "headers": dict(resp.headers),
            "body": body,
        }
    except Exception as e:
        return {"status_code": 0, "headers": {}, "body": str(e)}


def _safe_fn_name(ep: str) -> str:
    """Turn an endpoint path into a valid Python identifier suffix. Endpoints can
    contain '/', '-', '.', etc. (and arrive via body-parsing); without sanitizing,
    names like call_v1.2_data would be a SyntaxError, invalidating the WHOLE client
    and gating the episode reward to 0.0 even when discovery succeeded."""
    return re.sub(r"\W", "_", ep.strip("/")) or "root"


def generate_client(memory: WorkingMemory, base_url: str) -> str:
    lines = [
        "import os, requests",
        f'BASE_URL = os.environ.get("API_BASE_URL", "{base_url}")',
        "",
    ]

    # Topological sort of discovered endpoints
    import networkx as nx
    discovered = [
        n for n, d in memory.G.nodes(data=True)
        if d.get("status") in ("discovered", "mapped")
    ]

    # Try topological order respecting dependency edges
    try:
        order = list(nx.topological_sort(memory.G))
        ordered_discovered = [n for n in order if n in discovered]
    except Exception:
        ordered_discovered = discovered

    all_tokens = memory.get_all_tokens()

    for ep in ordered_discovered:
        ep_tokens = {}
        # Include any stored token as Authorization header
        for src_ep, tok in all_tokens.items():
            ep_tokens["Authorization"] = f"Bearer {tok}"
            break  # use first available token

        safe_name = _safe_fn_name(ep)
        headers_repr = repr(ep_tokens) if ep_tokens else "{}"
        lines.append(f"def call_{safe_name}():")
        lines.append(f"    r = requests.get(BASE_URL + {repr(ep)}, headers={headers_repr}, timeout=3)")
        lines.append(f"    return r.status_code, r.text")
        lines.append("")

    lines.append("if __name__ == '__main__':")
    for ep in ordered_discovered:
        safe_name = _safe_fn_name(ep)
        lines.append(f"    call_{safe_name}()")

    return "\n".join(lines)


def classify_failure(trajectory: list, memory: WorkingMemory) -> str:
    if not trajectory:
        return "coverage_plateau"

    statuses = [s["response"]["status_code"] for s in trajectory]
    if statuses.count(0) > len(statuses) * 0.5:
        return "auth_fail"

    auth_failures = sum(1 for s in statuses if s == 401 or s == 403)
    if auth_failures > len(statuses) * 0.5:
        return "auth_fail"

    endpoints_hit = set(s["action"]["endpoint"] if isinstance(s["action"], dict) else s["action"].endpoint for s in trajectory)
    if len(endpoints_hit) == 1:
        return "infinite_loop"

    discovered = sum(
        1 for _, d in memory.G.nodes(data=True)
        if d.get("status") in ("discovered", "mapped")
    )
    if discovered == 0:
        return "coverage_plateau"

    return "success"


def _extract_endpoints_from_body(body) -> list:
    if not body:
        return []
    text = body if isinstance(body, str) else str(body)
    return re.findall(r'(?<!["\w])(/[a-zA-Z][a-zA-Z0-9_/-]{0,30})', text)
