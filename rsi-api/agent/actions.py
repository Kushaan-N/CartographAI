"""
Constrained action space.
Action = (method, endpoint, headers, body) tuple.
Policy outputs structured JSON, parsed here into Action objects.
CRITICAL: from_model_output() must strip markdown fences — Gemma-3
sometimes wraps JSON in ```json ... ``` blocks.
"""
from dataclasses import dataclass, asdict
from typing import Optional
import json
import random

VALID_METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]


@dataclass
class Action:
    method: str
    endpoint: str
    headers: dict
    body: Optional[dict]

    def to_request_kwargs(self, base_url: str) -> dict:
        """
        Return kwargs for requests.request().
        {method, url: base_url+endpoint, headers, json: body}
        """
        return {
            "method": self.method,
            "url": base_url.rstrip("/") + self.endpoint,
            "headers": self.headers,
            "json": self.body,
        }

    def is_valid(self) -> bool:
        """method in VALID_METHODS and endpoint starts with /"""
        return self.method in VALID_METHODS and self.endpoint.startswith("/")

    def to_prompt_repr(self) -> str:
        """JSON string for LLM context and log_prob computation."""
        return json.dumps(asdict(self))

    def fingerprint(self) -> str:
        """Unique string for repeated-request detection: method+endpoint+sorted(headers)"""
        return f"{self.method}:{self.endpoint}:{json.dumps(self.headers, sort_keys=True)}"


class ActionSpace:
    def __init__(self):
        self.known_endpoints = ["/", "/api", "/health", "/status", "/login", "/auth"]
        self.header_templates = [
            {},
            {"Content-Type": "application/json"},
            {"Authorization": "Bearer {token}"},
            {"X-API-Key": "{key}"},
            {"X-Service-Token": "{token}"},
        ]

    def expand_endpoints(self, new_endpoints: list[str]):
        """Add to known_endpoints, dedup, preserve order."""
        for ep in new_endpoints:
            if ep not in self.known_endpoints:
                self.known_endpoints.append(ep)

    def sample_random(self) -> Action:
        """
        Random valid action from current vocabulary.
        Used for smoke tests and fallback on parse failure.
        """
        method = random.choice(["GET", "POST"])
        endpoint = random.choice(self.known_endpoints)
        headers = random.choice(self.header_templates).copy()
        return Action(method=method, endpoint=endpoint, headers=headers, body=None)

    def from_model_output(self, text: str) -> Optional[Action]:
        """
        Parse model output into Action.
        Steps:
        1. Strip markdown fences: remove ```json, ```, leading/trailing whitespace
        2. json.loads()
        3. Validate required keys: method, endpoint, headers, body
        4. Validate method in VALID_METHODS
        5. Validate endpoint starts with /
        6. Return Action or None if any step fails
        """
        try:
            import re
            text = re.sub(r'```json|```', '', text).strip()
            data = json.loads(text)
            if not all(k in data for k in ("method", "endpoint", "headers", "body")):
                return None
            action = Action(
                method=data["method"],
                endpoint=data["endpoint"],
                headers=data.get("headers", {}),
                body=data.get("body"),
            )
            return action if action.is_valid() else None
        except Exception:
            return None
