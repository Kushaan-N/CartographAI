"""
Tests the full demo visualization pipeline without needing
a trained model. Uses random policy so no GPU required.
Run this to confirm the visualization works before demo day.

Usage: python demo/test_demo.py
Then open: http://localhost:8080 in your browser
"""
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_agent_loop(server, config):
    """
    Runs agent episodes in a background thread.
    Broadcasts graph updates to the demo server on every step.
    Uses random policy so no GPU needed for testing.
    """
    from curriculum.factory import generate_api
    from training.episode import build_observation, execute_request
    from agent.memory import WorkingMemory
    from agent.actions import ActionSpace, Action
    from agent.error_forensics import extract_hints, apply_hints

    action_space = ActionSpace()

    class RandomPolicy:
        def __init__(self): self.action_space = ActionSpace()
        def sample_action(self, obs): return self.action_space.sample_random()

    policy = RandomPolicy()
    episode_num = 0

    while True:
        episode_num += 1
        print(f"\n[Agent] Starting episode {episode_num}...")

        api = generate_api(level=2, config=config)
        memory = WorkingMemory()

        # Register broadcast callback so every memory update
        # gets pushed to the websocket clients
        def make_broadcast_cb(srv, mem):
            def cb(graph_data):
                srv.broadcast_sync({"type": "graph", **graph_data})
            return cb

        memory.set_broadcast_callback(make_broadcast_cb(server, memory))

        # Seed initial hypotheses
        for ep in ["/", "/health", "/auth", "/login", "/api"]:
            memory.add_hypothesis(ep)
            action_space.expand_endpoints([ep])

        last_response = {"status_code": 0, "headers": {}, "body": ""}
        base_budget = config["training"]["max_episode_steps"]
        budget = base_budget
        total_steps = 0
        max_total = config["training"].get("max_total_steps", 35)
        consecutive_4xx = 0
        fragility_threshold = config["training"].get("fragility_threshold", 5)
        fingerprints = set()

        while budget > 0 and total_steps < max_total:
            obs = build_observation(memory, last_response, total_steps)
            action = policy.sample_action(obs)
            fingerprint = action.fingerprint()
            fingerprints.add(fingerprint)

            # Update node to "probing" before request
            memory.update_node(action.endpoint, "probing")

            server.broadcast_sync({
                "type": "terminal",
                "text": f"{action.method} {action.endpoint} {dict(action.headers)}",
                "line_type": "action"
            })

            response = execute_request(action, api.url)
            last_response = response
            status = response["status_code"]

            if 400 <= status < 500:
                consecutive_4xx += 1
            else:
                consecutive_4xx = 0

            if 200 <= status < 300:
                memory.update_node(
                    action.endpoint, "discovered",
                    last_status_code=status,
                    last_method=action.method
                )
                server.broadcast_sync({
                    "type": "terminal",
                    "text": f"  ← {status} OK {str(response['body'])[:80]}",
                    "line_type": "success"
                })

                body_dict = response["body"] if isinstance(response["body"], dict) else {}
                for k, v in body_dict.items():
                    if any(tk in k.lower() for tk in ["token", "access_token", "api_key", "key"]):
                        if isinstance(v, str):
                            memory.store_auth_token(action.endpoint, v)
                            budget += config["training"].get("bonus_steps_per_auth", 2)
                            server.broadcast_sync({
                                "type": "terminal",
                                "text": f"  [BONUS] Auth token acquired → +2 steps",
                                "line_type": "hint"
                            })
            else:
                memory.update_node(
                    action.endpoint, "failed",
                    last_status_code=status
                )
                server.broadcast_sync({
                    "type": "terminal",
                    "text": f"  ← {status} {str(response['body'])[:80]}",
                    "line_type": "error"
                })

            hints = extract_hints(response)
            if hints:
                apply_hints(hints, memory, action_space)
                for hint in hints:
                    server.broadcast_sync({
                        "type": "terminal",
                        "text": f"  [HINT] {hint.hint_type}: {hint.value} (+{hint.reward_bonus} reward)",
                        "line_type": "hint"
                    })

            if consecutive_4xx >= fragility_threshold:
                server.broadcast_sync({
                    "type": "terminal",
                    "text": f"  [LOCKED] API locked after {consecutive_4xx} consecutive failures",
                    "line_type": "error"
                })
                break

            budget -= 1
            total_steps += 1
            memory.step_count += 1

            # Small delay so visualization is readable
            time.sleep(0.3)

        from training.episode import generate_client
        from verifier.coverage_runner import instrument
        client_code = generate_client(memory, api.url)
        try:
            instr = instrument(api.source_code, api.port)
            cov_result = instr.run_client(client_code)
            instr.shutdown()
            coverage = cov_result.branch_coverage
        except Exception:
            coverage = 0.0

        server.broadcast_sync({
            "type": "coverage",
            "coverage": coverage,
            "level": 2
        })
        server.broadcast_sync({
            "type": "terminal",
            "text": f"[EPISODE {episode_num} COMPLETE] Coverage: {coverage:.1%} Steps: {total_steps}",
            "line_type": "coverage"
        })

        print(f"[Agent] Episode {episode_num} done. Coverage: {coverage:.1%}")
        api.shutdown()

        time.sleep(3)
        memory.reset()


def main():
    import yaml
    with open("configs/train_config.yaml") as f:
        config = yaml.safe_load(f)

    from demo.server import DemoServer

    server = DemoServer(checkpoint=None, port=8080, level=2)

    print("=" * 50)
    print("RSI-API DEMO VISUALIZATION TEST")
    print("=" * 50)
    print("Starting demo server on http://localhost:8080")
    print("Open that URL in your browser to see the visualization")
    print("Press Ctrl+C to stop")
    print("=" * 50)

    agent_thread = threading.Thread(
        target=run_agent_loop,
        args=(server, config),
        daemon=True
    )
    agent_thread.start()

    server.run()


if __name__ == "__main__":
    main()
