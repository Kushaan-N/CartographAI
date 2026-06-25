"""
Procedural Flask API generator.

generate_api(level, config, factory_weights) -> RunningAPI

Each call produces a unique Flask app. Runs as subprocess on Modal CPU
workers during episode collection, and locally during demo.

INVARIANT: every call produces a Flask app the model has never seen.
No caching. No reuse across episodes. Use uuid for api_id.
"""
import subprocess
import socket
import sys
import uuid
import time
import tempfile
import os
import random
import json as _json
import requests as req
from dataclasses import dataclass

from curriculum.auth_schemes import AuthScheme, sample_schemes, SCHEME_CHECK_FN
from curriculum.diversity import DiversityConfig
from curriculum.scheduler import DifficultyConfig


ENDPOINT_POOL = [
    "/users", "/products", "/orders", "/data", "/export",
    "/config", "/admin", "/metrics", "/events", "/reports",
]


@dataclass
class RunningAPI:
    process: subprocess.Popen
    url: str
    source_code: str
    port: int
    api_id: str
    tmp_path: str           # temp file path, for cleanup in shutdown()

    def is_alive(self) -> bool:
        """
        Check subprocess is running AND responding to GET /health.
        Returns False if process died or health check fails after 1s timeout.
        """
        try:
            r = req.get(f"{self.url}/health", timeout=1.0)
            return r.status_code == 200 and self.process.poll() is None
        except Exception:
            return False

    def shutdown(self):
        """
        Terminate subprocess, wait up to 3s, kill if still alive.
        Delete tmp_path temp file.
        """
        self.process.terminate()
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.kill()
        if os.path.exists(self.tmp_path):
            os.unlink(self.tmp_path)


def find_free_port() -> int:
    """Bind to port 0, get assigned port, close socket, return port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _build_source(secret: str, token: str, api_key: str, auth_code: str, endpoint_code: str) -> str:
    header = (
        "import os\n"
        "import uuid\n"
        "from flask import Flask, request, jsonify, session\n"
        "app = Flask(__name__)\n"
        f'app.secret_key = "{secret}"\n'
        "\n"
        f'VALID_TOKEN = "{token}"\n'
        f'API_KEY = "{api_key}"\n'
    )
    health = (
        '\n@app.route("/health")\n'
        'def health():\n'
        '    return jsonify(status="ok")\n'
    )
    inject = (
        '\nimport re as _re\n'
        'from werkzeug.routing import Rule as _Rule\n'
        '\n@app.route("/_inject", methods=["POST"])\n'
        'def _inject():\n'
        '    data = request.get_json(force=True) or {}\n'
        '    code = data.get("code", "")\n'
        '    try:\n'
        '        m = _re.search(r\'@app\\.route\\(["\\\']([^"\\\']+)["\\\']\', code)\n'
        '        if not m:\n'
        '            return jsonify(success=False, error="No @app.route found"), 400\n'
        '        path = m.group(1)\n'
        '        code_no_dec = _re.sub(r\'@[^\\n]+\\n\', \'\', code)\n'
        '        ns = {"request": request, "jsonify": jsonify}\n'
        '        exec(code_no_dec, ns)\n'
        '        fn = next((v for k,v in ns.items() if callable(v) and k not in ("request","jsonify")), None)\n'
        '        if fn:\n'
        '            ep = "injected_" + path.replace("/","_")\n'
        '            app.view_functions[ep] = fn\n'
        '            app.url_map.add(_Rule(path, endpoint=ep, methods=["GET","POST","PUT","DELETE"]))\n'
        '            return jsonify(success=True)\n'
        '        return jsonify(success=False, error="No callable found"), 400\n'
        '    except Exception as e:\n'
        '        return jsonify(success=False, error=str(e)), 400\n'
    )
    footer = (
        '\nif __name__ == "__main__":\n'
        '    app.run(port=int(os.environ.get("PORT", 5000)), debug=False)\n'
    )
    return "\n".join([header, auth_code, health, inject, endpoint_code, footer])


_SCHEME_ERROR_TYPE = {
    "bearer_token": "missing_bearer",
    "api_key_header": "missing_api_key",
    "session_cookie": "missing_session",
    "oauth2_mock": "missing_oauth",
    "misleading_error": "misleading",
}


def _make_diverse_auth_code(scheme: AuthScheme, diversity: DiversityConfig) -> str:
    """Rebuild a scheme's flask_code with a diversity-generated error body."""
    error_type = _SCHEME_ERROR_TYPE.get(scheme.name)
    if not error_type:
        return scheme.flask_code

    err = _json.dumps(diversity.get_error_body(error_type))

    if scheme.name == "bearer_token":
        return '\n'.join([
            'def check_bearer_auth():',
            '    auth = request.headers.get("Authorization", "")',
            '    if not auth.startswith("Bearer "):',
            f'        return jsonify({err}), 401',
            '    return None',
        ])
    if scheme.name == "api_key_header":
        return '\n'.join([
            'def check_api_key_auth():',
            '    if not request.headers.get("X-API-Key"):',
            f'        return jsonify({err}), 401',
            '    return None',
        ])
    if scheme.name == "session_cookie":
        return '\n'.join([
            '@app.route("/login", methods=["POST"])',
            'def login():',
            '    session["authenticated"] = True',
            '    return jsonify({"status": "logged in"})',
            '',
            'def check_session_auth():',
            '    if not session.get("authenticated"):',
            f'        return jsonify({err}), 401',
            '    return None',
        ])
    if scheme.name == "oauth2_mock":
        return '\n'.join([
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
            f'        return jsonify({err}), 401',
            '    token = auth[7:]',
            '    if token not in _oauth_tokens:',
            f'        return jsonify({err}), 401',
            '    return None',
        ])
    if scheme.name == "misleading_error":
        return '\n'.join([
            'def check_misleading_auth():',
            '    if not request.headers.get("X-Service-Token"):',
            f'        return jsonify({err}), 418',
            '    return None',
        ])
    return scheme.flask_code


def generate_api(level: int, config: dict, factory_weights: dict = None) -> RunningAPI:
    """
    Generate and spawn a new Flask API subprocess.
    """
    import random as _random
    seed = config.get("seed", None) if isinstance(config, dict) else None
    forced_api_id = config.get("api_id", None) if isinstance(config, dict) else None

    if seed is not None:
        _random.seed(seed)

    curriculum = config["curriculum"]
    levels = curriculum["levels"]
    max_level = max(int(k) for k in levels)
    clamped = min(level, max_level)
    d = levels.get(str(clamped)) or levels.get(clamped)

    n_endpoints = d["endpoints"]
    n_auth_schemes = d["auth_schemes"]
    n_dependencies = d["dependencies"]
    n_red_herrings = d["red_herrings"]

    diversity = DiversityConfig()

    # Sample auth schemes and deduplicate for code generation
    schemes = sample_schemes(level, n_auth_schemes)
    seen_names: set = set()
    unique_schemes: list = []
    for s in schemes:
        if s.name not in seen_names:
            unique_schemes.append(s)
            seen_names.add(s.name)

    auth_code = "\n\n".join(_make_diverse_auth_code(s, diversity) for s in unique_schemes)

    # Generate diverse endpoint names, excluding auth-scheme-reserved paths
    _reserved = {"/health", "/_inject", "/login", "/oauth/token"}
    _raw = diversity.get_endpoint_names(n=n_endpoints * 2 + 4)
    endpoints = [ep for ep in _raw if ep not in _reserved][:n_endpoints]
    if len(endpoints) < n_endpoints:
        endpoints += [f"/ep{i}" for i in range(n_endpoints - len(endpoints))]

    # Mark red herring endpoints
    n_rh = min(n_red_herrings, len(endpoints))
    red_herring_set = set(random.sample(endpoints, n_rh)) if n_rh > 0 else set()

    # Build linear dependency chain: ep[i+1] depends on ep[i]
    dep_order = list(endpoints)
    random.shuffle(dep_order)
    n_dep = min(n_dependencies, len(dep_order) - 1)

    dependencies: dict = {}   # B -> A  (B requires token from A)
    dep_tokens: dict = {}     # A -> token  (A provides this token)
    for i in range(n_dep):
        a = dep_order[i]
        b = dep_order[i + 1]
        dependencies[b] = a
        dep_tokens[a] = uuid.uuid4().hex[:16]

    # Generate endpoint handlers
    endpoint_parts = []
    for i, ep in enumerate(endpoints):
        if not unique_schemes:
            scheme = None
            check_fn = None
        else:
            scheme = unique_schemes[i % len(unique_schemes)]
            check_fn = SCHEME_CHECK_FN[scheme.name]
        method = random.choice(["GET", "POST"])
        fn_name = ep.lstrip("/").replace("-", "_")
        is_rh = ep in red_herring_set
        dep_on = dependencies.get(ep)
        provides_token = ep in dep_tokens

        lines = [
            f'@app.route("{ep}", methods=["{method}"])',
            f"def {fn_name}():",
        ]
        if check_fn:
            lines += [
                f"    result = {check_fn}()",
                "    if result:",
                "        return result",
            ]

        if dep_on:
            dep_name = dep_on.lstrip("/")
            tok = dep_tokens[dep_on]
            lines += [
                f'    if request.headers.get("X-Token-{dep_name}") != "{tok}":',
                f'        return jsonify({{"error": "Missing required header X-Token-{dep_name}", "hint": "Call {dep_on} first"}}), 403',
            ]

        resp_schema = diversity.get_response_schema()
        if provides_token:
            resp_schema[f"X-Token-{fn_name}"] = dep_tokens[ep]
        resp_json = _json.dumps(resp_schema)

        if is_rh:
            lines += [
                "    # RED_HERRING",
                f'    return jsonify({resp_json})',
            ]
        else:
            lines += [
                f'    return jsonify({resp_json})',
            ]

        endpoint_parts.append("\n".join(lines))

    endpoint_code = "\n\n".join(endpoint_parts)

    # Discovery surface: an open root index that lists available routes (as real
    # APIs commonly do). This makes endpoints discoverable by probing rather than
    # by guessing unguessable random-slug names. The agent must still infer auth
    # schemes and dependency order to actually COVER each handler's branches —
    # listing a route != covering it. Controlled by config["factory"]["discoverable_index"]
    # (default True). episode.py already parses paths from 200 bodies into memory.
    factory_cfg = config.get("factory", {}) if isinstance(config, dict) else {}
    if factory_cfg.get("discoverable_index", True):
        index_paths = list(endpoints)
        index_code = "\n".join([
            '@app.route("/")',
            "def _api_index():",
            f"    return jsonify({{'endpoints': {index_paths!r}, 'service': 'api'}})",
        ])
        endpoint_code = index_code + "\n\n" + endpoint_code

    secret = uuid.uuid4().hex
    token = uuid.uuid4().hex
    api_key = uuid.uuid4().hex

    source_code = _build_source(secret, token, api_key, auth_code, endpoint_code)

    # Write to temp file
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
        f.write(source_code)
        tmp_path = f.name

    port = find_free_port()
    process = subprocess.Popen(
        [sys.executable, tmp_path],
        env={**os.environ, "PORT": str(port)},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    api_id = forced_api_id if forced_api_id else str(uuid.uuid4())
    url = f"http://127.0.0.1:{port}"

    running_api = RunningAPI(
        process=process,
        url=url,
        source_code=source_code,
        port=port,
        api_id=api_id,
        tmp_path=tmp_path,
    )

    # Wait up to 5s for the server to be ready
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if running_api.is_alive():
            break
        time.sleep(0.1)

    return running_api
