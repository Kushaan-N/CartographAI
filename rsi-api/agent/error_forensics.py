"""
Extracts structural hints from HTTP error response bodies.
Gives +0.1 reward bonus per hint found and applied.

Hint patterns to detect:
- Missing header: X-Something
- Authenticate at /endpoint
- Token expired, refresh at /endpoint
- Requires role: admin
- Use endpoint /endpoint
"""
from dataclasses import dataclass
import re
import json


@dataclass
class Hint:
    hint_type: str      # missing_header | auth_endpoint | refresh_endpoint | role_required
    value: str          # the extracted value (header name, path, role)
    confidence: float   # 0.6 for string match, 0.9 for JSON match
    reward_bonus: float = 0.1


HINT_PATTERNS = [
    (r"[Mm]issing header[:\s]+([A-Za-z0-9_-]+)", "missing_header", 1),
    (r"[Tt]oken expired.*?refresh at ([/\w]+)", "refresh_endpoint", 1),
    (r"[Aa]uthenticate at ([/\w]+)", "auth_endpoint", 1),
    (r"[Rr]equires? role[:\s]+(\w+)", "role_required", 1),
    (r"[Uu]se endpoint ([/\w]+)", "auth_endpoint", 1),
    (r"[Pp]rovide ([A-Za-z0-9_-]+) header", "missing_header", 1),
]


def extract_hints(response: dict) -> list[Hint]:
    """
    Parse response dict for structural hints.
    response: {status_code: int, headers: dict, body: str|dict}

    Steps:
    1. Only run on 4xx responses (status_code >= 400 and < 500)
    2. Convert body to string: json.dumps if dict, str() otherwise
    3. Apply each HINT_PATTERN with re.search()
    4. confidence=0.9 if body was valid JSON, 0.6 if plain string
    5. Return list of all Hint instances found
    """
    if response["status_code"] < 400 or response["status_code"] >= 500:
        return []
    is_json = isinstance(response["body"], dict)
    body_str = json.dumps(response["body"]) if is_json else str(response["body"])
    confidence = 0.9 if is_json else 0.6
    hints = []
    for pattern, hint_type, _ in HINT_PATTERNS:
        match = re.search(pattern, body_str)
        if match:
            hints.append(Hint(hint_type=hint_type, value=match.group(1), confidence=confidence))
    return hints


def apply_hints(hints: list[Hint], memory, action_space) -> float:
    """
    Apply hints to memory and action_space. Return total reward bonus.

    For each hint:
    - missing_header: add {hint.value: "{value}"} to action_space.header_templates
    - auth_endpoint: action_space.expand_endpoints([hint.value])
                     + memory.add_hypothesis(hint.value)
    - refresh_endpoint: same as auth_endpoint
    - role_required: log only, no direct action (future work)

    Return sum of hint.reward_bonus for all applied hints.
    """
    total_bonus = 0.0
    for hint in hints:
        if hint.hint_type == "missing_header":
            action_space.header_templates.append({hint.value: "placeholder"})
        elif hint.hint_type in ("auth_endpoint", "refresh_endpoint"):
            action_space.expand_endpoints([hint.value])
            memory.add_hypothesis(hint.value)
        total_bonus += hint.reward_bonus
    return total_bonus
