"""
Authentication pattern library for procedural API generation.

Five auth scheme types, each with:
- flask_code: string of Python/Flask code enforcing the auth
- error_hint: JSON string returned on auth failure (mined by error_forensics.py)
- difficulty: 1-3, used by sample_schemes() to match curriculum level
"""
import random
from dataclasses import dataclass


@dataclass
class AuthScheme:
    name: str
    flask_code: str      # Flask route handler code as string
    error_hint: str      # JSON error body returned on auth failure
    difficulty: int      # 1=easy, 2=medium, 3=hard


BEARER_TOKEN = AuthScheme(
    name="bearer_token",
    flask_code='\n'.join([
        'def check_bearer_auth():',
        '    auth = request.headers.get("Authorization", "")',
        '    if not auth.startswith("Bearer "):',
        '        return jsonify({"error": "Missing header: Authorization", "hint": "Use Bearer token"}), 401',
        '    return None',
    ]),
    error_hint='{"error": "Missing header: Authorization", "hint": "Use Bearer token"}',
    difficulty=1,
)

API_KEY_HEADER = AuthScheme(
    name="api_key_header",
    flask_code='\n'.join([
        'def check_api_key_auth():',
        '    if not request.headers.get("X-API-Key"):',
        '        return jsonify({"error": "Missing header: X-API-Key"}), 401',
        '    return None',
    ]),
    error_hint='{"error": "Missing header: X-API-Key"}',
    difficulty=1,
)

SESSION_COOKIE = AuthScheme(
    name="session_cookie",
    flask_code='\n'.join([
        '@app.route("/login", methods=["POST"])',
        'def login():',
        '    session["authenticated"] = True',
        '    return jsonify({"status": "logged in"})',
        '',
        'def check_session_auth():',
        '    if not session.get("authenticated"):',
        '        return jsonify({"error": "Authenticate at /login", "hint": "POST to /login first"}), 401',
        '    return None',
    ]),
    error_hint='{"error": "Authenticate at /login", "hint": "POST to /login first"}',
    difficulty=2,
)

OAUTH2_MOCK = AuthScheme(
    name="oauth2_mock",
    flask_code='\n'.join([
        '_oauth_tokens = set()',
        '',
        '@app.route("/oauth/token", methods=["POST"])',
        'def oauth_token():',
        '    data = request.json or {}',
        '    if not data.get("client_id") or not data.get("client_secret"):',
        '        return jsonify({"error": "Missing client_id or client_secret"}), 400',
        '    token = str(uuid.uuid4())',
        '    _oauth_tokens.add(token)',
        '    return jsonify({"access_token": token, "token_type": "bearer"})',
        '',
        'def check_oauth_auth():',
        '    auth = request.headers.get("Authorization", "")',
        '    if not auth.startswith("Bearer "):',
        '        return jsonify({"error": "Use endpoint /oauth/token", "hint": "POST client_id and client_secret"}), 401',
        '    token = auth[7:]',
        '    if token not in _oauth_tokens:',
        '        return jsonify({"error": "Use endpoint /oauth/token", "hint": "POST client_id and client_secret"}), 401',
        '    return None',
    ]),
    error_hint='{"error": "Use endpoint /oauth/token", "hint": "POST client_id and client_secret"}',
    difficulty=3,
)

MISLEADING_ERROR = AuthScheme(
    name="misleading_error",
    flask_code='\n'.join([
        'def check_misleading_auth():',
        '    if not request.headers.get("X-Service-Token"):',
        '        return jsonify({"status": "teapot", "hint": "Try adding X-Service-Token header"}), 418',
        '    return None',
    ]),
    error_hint='{"status": "teapot", "hint": "Try adding X-Service-Token header"}',
    difficulty=3,
)

ALL_SCHEMES = [BEARER_TOKEN, API_KEY_HEADER, SESSION_COOKIE, OAUTH2_MOCK, MISLEADING_ERROR]

SCHEME_CHECK_FN = {
    "bearer_token": "check_bearer_auth",
    "api_key_header": "check_api_key_auth",
    "session_cookie": "check_session_auth",
    "oauth2_mock": "check_oauth_auth",
    "misleading_error": "check_misleading_auth",
}


def sample_schemes(level: int, n: int) -> list[AuthScheme]:
    """
    Sample n auth schemes appropriate for the given difficulty level.
    Level 1: only difficulty<=1 schemes
    Level 2: difficulty<=2 schemes
    Level 3+: all schemes
    Use random.choices with equal weights within the eligible set.
    """
    if level <= 1:
        eligible = [s for s in ALL_SCHEMES if s.difficulty <= 1]
    elif level == 2:
        eligible = [s for s in ALL_SCHEMES if s.difficulty <= 2]
    else:
        eligible = ALL_SCHEMES
    return random.choices(eligible, k=n)
