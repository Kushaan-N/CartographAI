"""
Judge live-injection handler.
Accepts Flask route code, injects into running demo API, notifies agent memory.
"""
import re
from agent.memory import WorkingMemory


def extract_endpoint_path(route_code: str) -> str:
    match = re.search(r'@app\.route\(["\']([^"\']+)["\']', route_code)
    if not match:
        raise ValueError(f"No @app.route() decorator found in: {route_code[:100]}")
    return match.group(1)


def handle_inject(route_code: str, demo_api, memory: WorkingMemory) -> dict:
    try:
        path = extract_endpoint_path(route_code)
        import requests
        r = requests.post(
            f"{demo_api.url}/_inject",
            json={"code": route_code},
            timeout=2,
        )
        if r.status_code == 200:
            if memory is not None:
                memory.add_hypothesis(path)
            return {
                "success": True,
                "endpoint": path,
                "message": f"Injected {path}. Agent will probe on next step.",
            }
        else:
            return {"success": False, "error": f"API rejected inject: {r.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}
