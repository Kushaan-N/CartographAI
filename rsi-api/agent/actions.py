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

# Endpoint names in factory-generated APIs are drawn from a FIXED vocabulary
# (curriculum/diversity.py DOMAIN_POOLS) plus version suffixes. Sampling guesses
# from this same vocabulary gives the policy a real (non-zero) hit rate on unseen
# APIs — random alphanumeric slugs are unguessable, but a large fraction of
# endpoints use these domain words. Mirrors diversity.DOMAIN_POOLS so exploration
# and the actual name distribution line up.
COMMON_PATH_SEGMENTS = [
    # enterprise
    "records", "accounts", "ledger", "invoices", "contracts", "personnel",
    "assets", "compliance", "audit", "transactions", "policies", "workflows",
    "submissions", "approvals", "registry",
    # technical
    "nodes", "clusters", "pipelines", "artifacts", "manifests", "schemas",
    "indices", "streams", "queues", "subscribers", "consumers", "producers",
    "namespaces", "contexts", "handlers",
    # civic
    "permits", "licenses", "filings", "allocations", "disbursements",
    "applicants", "beneficiaries", "facilities", "zones", "incidents",
    "complaints", "services", "programs", "agencies", "resources",
    # generic
    "items", "entries", "objects", "entities", "elements", "instances",
    "references", "documents", "data", "users", "list", "api",
]

# Version/qualifier suffixes the factory appends to domain words (diversity.py).
PATH_SUFFIXES = ["", "", "", "_v2", "_v3", "_2024", "_prod", "_beta", "_ext", "_legacy", "_new"]

# Auth-flavored endpoint names the factory uses for auth routes.
AUTH_PATH_SEGMENTS = [
    "authenticate", "authorize", "token", "credentials", "session", "access",
    "identity", "verify", "validate", "signin", "auth", "login", "oauth/token",
]


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

    def _sample_guess_endpoint(self) -> str:
        """
        Generate a plausible endpoint guess from the factory's fixed vocabulary.
        Forms: /records, /records_v2, /api/items, /v1/contracts, /auth, etc.
        """
        roll = random.random()
        if roll < 0.20:
            return random.choice(self.known_endpoints)
        if roll < 0.35:
            return "/" + random.choice(AUTH_PATH_SEGMENTS)
        seg = random.choice(COMMON_PATH_SEGMENTS) + random.choice(PATH_SUFFIXES)
        prefix = random.choice(["", "", "", "/api", "/v1", "/v2"])
        return f"{prefix}/{seg}"

    def sample_random(self) -> Action:
        """
        Random valid action. Guesses endpoints from the factory's domain
        vocabulary (not just 6 hardcoded paths) so exploration has a real hit
        rate on unseen APIs. Used for smoke tests and parse-failure fallback.
        """
        method = random.choice(["GET", "POST"])
        endpoint = self._sample_guess_endpoint()
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


# ── Constrained action selection ──────────────────────────────────────────────
# The agent SELECTS an action from a discrete, valid candidate set instead of
# generating a free-form endpoint string. This honors the "constrained action
# space" invariant: the policy can never emit an invalid/hallucinated endpoint,
# can't over-emit one path, and the RL problem becomes "rank the candidates"
# (tractable) rather than "generate an exact random string" (hard for a 1B model).

# Generic auth headers that satisfy the level-1 schemes (bearer / api_key /
# misleading all accept ANY value).
AUTH_HEADERS = {
    "Authorization": "Bearer t",
    "X-API-Key": "t",
    "X-Service-Token": "t",
}
SEED_CANDIDATES = ["/", "/health", "/api", "/status", "/login", "/auth"]


def build_candidates(observation: dict) -> list:
    """Ordered, deduped endpoints the policy may probe this step: the discovery
    index '/' and seeded guesses first, then everything the observation has
    surfaced (hypothesized = not-yet-probed, then already-discovered). Order is
    deterministic so the same observation always yields the same menu — required
    for log_prob to reconstruct the choice."""
    hyp = observation.get("hypothesized_endpoints", []) or []
    disc = observation.get("discovered_endpoints", []) or []
    out = []
    for ep in SEED_CANDIDATES + list(hyp) + list(disc):
        if ep not in out:
            out.append(ep)
    return out


def selection_to_action(choice: int, method: str, auth: bool, candidates: list) -> Action:
    """Resolve a (choice, method, auth) selection into a concrete Action."""
    if not candidates:
        candidates = SEED_CANDIDATES
    if not isinstance(choice, int) or not (0 <= choice < len(candidates)):
        choice = 0
    return Action(
        method=method if method in ("GET", "POST") else "GET",
        endpoint=candidates[choice],
        headers=dict(AUTH_HEADERS) if auth else {},
        body=None,
    )


def action_to_selection(action, candidates: list) -> dict:
    """Inverse of selection_to_action — map a concrete Action back to a selection
    (used to format SFT demos and to score actions in log_prob)."""
    ep = action.endpoint if not isinstance(action, dict) else action.get("endpoint")
    method = action.method if not isinstance(action, dict) else action.get("method", "GET")
    headers = action.headers if not isinstance(action, dict) else action.get("headers", {})
    try:
        choice = candidates.index(ep)
    except (ValueError, AttributeError):
        choice = 0
    return {
        "choice": choice,
        "method": method if method in ("GET", "POST") else "GET",
        "auth": bool(headers),
    }


def selection_json(selection: dict) -> str:
    """Canonical JSON string for a selection (stable key order)."""
    return json.dumps({
        "choice": selection["choice"],
        "method": selection["method"],
        "auth": selection["auth"],
    })


def parse_selection(text: str, candidates: list) -> Optional[Action]:
    """Parse a model's selection JSON into an Action, constrained to candidates."""
    try:
        import re
        text = re.sub(r"```json|```", "", text).strip()
        data = json.loads(text)
        choice = int(data["choice"])
        if not (0 <= choice < len(candidates)):
            return None
        return selection_to_action(choice, data.get("method", "GET"),
                                   bool(data.get("auth", False)), candidates)
    except Exception:
        return None
