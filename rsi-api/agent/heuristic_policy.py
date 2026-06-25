"""
Scripted "cheating" policy used to generate SFT warm-up demonstrations.

It is NOT trainable and is NOT used at inference. Its only job is to produce
high-coverage episodes whose (observation -> action) pairs teach the real Gemma
policy the *behaviour* of API exploration: probe auth endpoints, send auth
headers, and hit endpoints drawn from the factory's fixed domain vocabulary.

It cheats by reading api.source_code:
  - extracts real endpoint paths + their HTTP methods,
  - extracts dependency-token headers (X-Token-*) and their secret values,
  - probes auth endpoints with admin/admin to harvest tokens.

This is acceptable because the *completions* it emits (auth-probing, domain-word
endpoints, Bearer/X-API-Key headers) are the generalizable behaviour we want the
policy to imitate; the API-specific secrets it discovers are incidental.

Interface (matches agent/policy.Policy):
  - sample_action(observation) -> Action
  - log_prob(observation, action) -> torch.Tensor   (dummy; not trained)
"""
import re
import torch

from agent.actions import Action, ActionSpace

_ROUTE_RE = re.compile(
    r'@app\.route\(\s*["\']([^"\']+)["\']\s*(?:,\s*methods\s*=\s*\[([^\]]*)\])?\s*\)'
)
_DEP_TOKEN_RE = re.compile(
    r'request\.headers\.get\(\s*["\'](X-Token-[^"\']+)["\']\s*\)\s*!=\s*["\']([^"\']+)["\']'
)

# Endpoints we should never probe as data routes.
_SKIP = {"/_inject"}
# Paths that look like auth/login routes worth POSTing credentials to.
_AUTH_HINTS = ("login", "auth", "token", "oauth", "session", "signin", "credential")


class HeuristicPolicy:
    def __init__(self, api, config: dict = None):
        self.action_space = ActionSpace()
        self.source = getattr(api, "source_code", "") or ""

        # (path, method) for every real route, in source order.
        self.routes: list[tuple[str, str]] = []
        for path, methods in _ROUTE_RE.findall(self.source):
            if path in _SKIP:
                continue
            method = "GET"
            if methods:
                first = methods.split(",")[0].strip().strip("'\"").upper()
                if first:
                    method = first
            self.routes.append((path, method))

        # Dependency token headers extracted straight from the source (cheat):
        # {"X-Token-foo": "<secret>"} — satisfies dependency gates.
        self.dep_headers: dict[str, str] = {
            name: val for name, val in _DEP_TOKEN_RE.findall(self.source)
        }

        self._plan: list[Action] = self._build_plan()
        self._idx = 0

    # ── header helpers ────────────────────────────────────────────────────────
    def _auth_headers(self) -> dict:
        """Generic auth headers that satisfy level-1 schemes (bearer/api_key/
        misleading accept ANY value) plus all dependency-token headers."""
        h = {
            "Authorization": "Bearer test123",
            "X-API-Key": "test123",
            "X-Service-Token": "test123",
        }
        h.update(self.dep_headers)
        return h

    # ── plan construction ─────────────────────────────────────────────────────
    def _build_plan(self) -> list[Action]:
        plan: list[Action] = []

        # 1. Probe the root index FIRST. With a discoverable index, this surfaces
        # the real endpoint paths into the observation (episode.py parses them),
        # so the resulting demonstrations teach the generalizable behaviour
        # "probe /, then probe the endpoints the observation now lists" rather
        # than memorizing API-specific names.
        plan.append(Action("GET", "/", {}, None))

        # 2. Warm up on /health (always present, no auth).
        plan.append(Action("GET", "/health", {}, None))

        # 3. Harvest tokens from auth-looking endpoints (admin/admin + oauth creds).
        cred_bodies = [
            {"username": "admin", "password": "admin"},
            {"client_id": "admin", "client_secret": "admin"},
        ]
        for path, _method in self.routes:
            if any(hint in path.lower() for hint in _AUTH_HINTS):
                for body in cred_bodies:
                    plan.append(Action("POST", path, {"Content-Type": "application/json"}, body))

        # 4. Probe every real endpoint with correct method + full auth headers.
        for path, method in self.routes:
            plan.append(Action(method, path, self._auth_headers(), None))

        return plan

    # ── policy interface ──────────────────────────────────────────────────────
    def sample_action(self, observation: dict) -> Action:
        if self._idx < len(self._plan):
            action = self._plan[self._idx]
            self._idx += 1
            # Return a fresh copy so downstream mutation can't corrupt the plan.
            return Action(action.method, action.endpoint, dict(action.headers), action.body)

        # Plan exhausted: keep re-probing discovered/known endpoints with auth so
        # remaining budget is spent productively rather than on garbage.
        obs = observation or {}
        candidates = (
            list(obs.get("discovered_endpoints", []))
            or [p for p, _ in self.routes]
            or ["/health"]
        )
        path = candidates[self._idx % len(candidates)]
        method = next((m for p, m in self.routes if p == path), "GET")
        self._idx += 1
        return Action(method, path, self._auth_headers(), None)

    def log_prob(self, observation: dict, action) -> torch.Tensor:
        # Scripted policy — never trained. Return a differentiable constant so any
        # accidental use in a loss doesn't crash.
        return torch.tensor(-1.0, requires_grad=True)
