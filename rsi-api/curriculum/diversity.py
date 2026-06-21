"""
Diversity utilities for anti-template curriculum generation.

Prevents factory.py from producing APIs with predictable naming patterns
that the agent could memorize rather than truly learn from.

Three diversity axes:
1. Endpoint naming: random alphanumeric, realistic-but-varied, domain-shifted
2. Error message phrasing: large pool of semantically equivalent but
   lexically diverse error messages for each auth failure type
3. Data schema fields: random field names in response bodies

Usage:
    from curriculum.diversity import DiversityConfig, generate_endpoint_names
    names = generate_endpoint_names(n=6, mode="mixed")
"""

import random
import string
import hashlib

# ── Endpoint name generation ──────────────────────────────────────────────────

DOMAIN_POOLS = {
    "enterprise": [
        "records", "accounts", "ledger", "invoices", "contracts",
        "personnel", "assets", "compliance", "audit", "transactions",
        "policies", "workflows", "submissions", "approvals", "registry"
    ],
    "technical": [
        "nodes", "clusters", "pipelines", "artifacts", "manifests",
        "schemas", "indices", "streams", "queues", "subscribers",
        "consumers", "producers", "namespaces", "contexts", "handlers"
    ],
    "civic": [
        "permits", "licenses", "filings", "allocations", "disbursements",
        "applicants", "beneficiaries", "facilities", "zones", "incidents",
        "complaints", "services", "programs", "agencies", "resources"
    ],
    "generic": [
        "items", "entries", "objects", "resources", "entities",
        "elements", "instances", "references", "documents", "data"
    ]
}

AUTH_ENDPOINT_NAMES = [
    "authenticate", "authorize", "token", "credentials", "session",
    "access", "identity", "verify", "validate", "handshake",
    "connect", "signin", "auth", "whoami", "introspect"
]


def generate_endpoint_names(n: int, mode: str = "mixed") -> list[str]:
    """
    Generate n unique endpoint path names.

    Modes:
    - "mixed": combination of all strategies (recommended for training)
    - "random": short random alphanumeric strings (/xk3p, /auth_v2)
    - "domain": words from a randomly chosen domain pool
    - "versioned": domain words with version suffixes (/records_v2, /data_2024)

    Always includes one auth-flavored endpoint name.
    """
    names = set()
    auth_name = "/" + random.choice(AUTH_ENDPOINT_NAMES)
    names.add(auth_name)

    strategies = {
        "random": lambda: "/" + _random_slug(),
        "domain": lambda: "/" + _domain_word(),
        "versioned": lambda: "/" + _versioned_word(),
    }

    while len(names) < n:
        if mode == "mixed":
            strategy = random.choice(list(strategies.values()))
        else:
            strategy = strategies.get(mode, strategies["domain"])
        names.add(strategy())

    return list(names)[:n]


def _random_slug(length: int = None) -> str:
    """
    Generate a random alphanumeric slug.
    Length between 4-8 chars if not specified.
    Mix of letters and underscores, no purely numeric strings.
    Examples: xk3p, auth_v2, data_prod, svc_01
    """
    if length is None:
        length = random.randint(4, 8)
    chars = string.ascii_lowercase + string.digits
    slug = random.choice(string.ascii_lowercase)
    slug += "".join(random.choices(chars + "_", k=length - 1))
    slug = slug.replace("__", "_").rstrip("_")
    return slug


def _domain_word() -> str:
    """Pick a random word from a random domain pool."""
    pool = random.choice(list(DOMAIN_POOLS.values()))
    return random.choice(pool)


def _versioned_word() -> str:
    """Domain word with random version suffix: _v2, _2024, _prod, _beta, _new"""
    suffixes = ["_v2", "_v3", "_2024", "_prod", "_beta", "_new", "_legacy", "_ext"]
    return _domain_word() + random.choice(suffixes)


# ── Error message diversity ───────────────────────────────────────────────────

ERROR_PHRASES = {
    "missing_bearer": [
        "Missing header: Authorization",
        "Authorization header required",
        "Bearer token not provided",
        "No authorization credentials found",
        "Authentication header absent",
        "Provide Authorization: Bearer token",
        "Request missing required auth header",
        "Header Authorization is mandatory",
    ],
    "missing_api_key": [
        "Missing header: X-API-Key",
        "API key header not found",
        "X-API-Key is required",
        "No API key provided in headers",
        "Authentication failed: missing key",
        "Provide X-API-Key header",
        "API access requires X-API-Key header",
        "Key header absent from request",
    ],
    "missing_session": [
        "Authenticate at /login",
        "Session not established, POST to /login first",
        "No active session found",
        "Login required before accessing this resource",
        "Session cookie missing, authenticate first",
        "Use /login endpoint to establish session",
        "Authentication session required",
        "Please login before continuing",
    ],
    "missing_oauth": [
        "Use endpoint /oauth/token",
        "OAuth token required, POST to /oauth/token",
        "Access token not found",
        "Obtain access token from /oauth/token",
        "OAuth2 authentication required",
        "Request access token first",
        "Token endpoint: /oauth/token",
        "Bearer token must be obtained via OAuth flow",
    ],
    "misleading": [
        "Service temporarily unavailable",
        "Request could not be processed",
        "Invalid request format",
        "Unexpected error occurred",
        "Resource access denied",
        "Operation not permitted",
    ],
    "dependency": [
        "Prerequisite not satisfied",
        "Required prior step incomplete",
        "Dependency not met",
        "Complete initialization first",
        "Prior operation required",
        "Setup step missing",
    ]
}


def get_error_phrase(error_type: str) -> str:
    """
    Get a random error phrase for the given error type.
    error_type must be a key in ERROR_PHRASES.
    Returns a random phrase from the pool.
    Raises KeyError if error_type not found.
    """
    phrases = ERROR_PHRASES.get(error_type)
    if not phrases:
        raise KeyError(f"Unknown error type: {error_type}. Valid: {list(ERROR_PHRASES.keys())}")
    return random.choice(phrases)


def get_error_body(error_type: str, extra: dict = None) -> dict:
    """
    Generate a diverse error response body dict.

    Not always {"error": "..."} — varies the structure:
    - Sometimes: {"error": "..."}
    - Sometimes: {"message": "...", "code": 401}
    - Sometimes: {"detail": "...", "status": "unauthorized"}
    - Sometimes: {"errors": ["..."], "request_id": "abc123"}

    Always contains the hint phrase from get_error_phrase(error_type).
    extra dict is merged in if provided.
    """
    phrase = get_error_phrase(error_type)
    style = random.choice(["simple", "detailed", "nested", "list"])
    request_id = hashlib.md5(phrase.encode()).hexdigest()[:8]

    if style == "simple":
        body = {"error": phrase}
    elif style == "detailed":
        body = {"message": phrase, "code": 401, "request_id": request_id}
    elif style == "nested":
        body = {"detail": phrase, "status": "unauthorized", "meta": {"id": request_id}}
    else:  # list
        body = {"errors": [phrase], "request_id": request_id}

    if extra:
        body.update(extra)
    return body


# ── Response body diversity ───────────────────────────────────────────────────

FIELD_NAME_POOLS = {
    "id_fields": ["id", "uid", "uuid", "record_id", "ref", "identifier", "key"],
    "name_fields": ["name", "title", "label", "description", "display_name", "caption"],
    "date_fields": ["created_at", "updated_at", "timestamp", "date", "modified", "issued"],
    "status_fields": ["status", "state", "active", "enabled", "flag", "condition"],
    "value_fields": ["value", "amount", "quantity", "count", "total", "sum", "balance"],
    "meta_fields": ["version", "source", "origin", "category", "type", "kind", "class"]
}


def generate_response_schema(n_fields: int = None) -> dict:
    """
    Generate a random response body schema with varied field names.
    n_fields between 3-8 if not specified.
    Pick field names from different pools to ensure diversity.
    Returns dict suitable for jsonify() in Flask handler.
    """
    if n_fields is None:
        n_fields = random.randint(3, 8)

    schema = {}
    pools_used = set()

    def add_from_pool(pool_key):
        if pool_key not in pools_used:
            field = random.choice(FIELD_NAME_POOLS[pool_key])
            pools_used.add(pool_key)
            return field, pool_key
        return None, None

    # Always include an id field
    field, pool = add_from_pool("id_fields")
    schema[field] = "".join(random.choices(string.hexdigits.lower(), k=12))

    remaining_pools = list(FIELD_NAME_POOLS.keys())
    random.shuffle(remaining_pools)

    for pool_key in remaining_pools:
        if len(schema) >= n_fields:
            break
        field = random.choice(FIELD_NAME_POOLS[pool_key])
        if field not in schema:
            if pool_key == "date_fields":
                schema[field] = "2024-01-15T10:30:00Z"
            elif pool_key == "status_fields":
                schema[field] = random.choice(["active", "pending", "complete", "inactive"])
            elif pool_key == "value_fields":
                schema[field] = random.randint(1, 10000)
            else:
                schema[field] = "".join(random.choices(string.ascii_lowercase, k=8))

    return schema


# ── Diversity config ──────────────────────────────────────────────────────────

class DiversityConfig:
    """
    Controls diversity parameters for a single API generation call.
    Instantiated by factory.py with randomized settings each call.
    """
    def __init__(self):
        self.naming_mode = random.choice(["mixed", "random", "domain", "versioned"])
        self.domain = random.choice(list(DOMAIN_POOLS.keys()))
        self.error_body_style = random.choice(["simple", "detailed", "nested"])
        self.response_complexity = random.randint(3, 8)

    def get_endpoint_names(self, n: int) -> list[str]:
        return generate_endpoint_names(n, mode=self.naming_mode)

    def get_error_body(self, error_type: str) -> dict:
        return get_error_body(error_type)

    def get_response_schema(self) -> dict:
        return generate_response_schema(self.response_complexity)
