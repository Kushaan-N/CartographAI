"""
NetworkX working memory graph. Persists across steps within one episode.

Node statuses: hypothesized | probing | discovered | failed | mapped
INVARIANT: to_json() must always succeed — no non-serializable objects.
"""
import networkx as nx
from typing import Optional, Callable


class WorkingMemory:
    def __init__(self):
        self.G = nx.DiGraph()
        self.auth_tokens: dict[str, str] = {}
        self.step_count: int = 0
        self.red_herring_steps: int = 0
        self.total_steps: int = 0
        self._broadcast_cb: Optional[Callable] = None

    def set_broadcast_callback(self, fn: Callable):
        """Called by demo/server.py to register websocket broadcast."""
        self._broadcast_cb = fn

    def _broadcast(self):
        """Call broadcast callback if registered. Catches all exceptions."""
        if self._broadcast_cb:
            try:
                self._broadcast_cb(self.to_json())
            except Exception:
                pass

    def add_hypothesis(self, endpoint: str):
        """
        Add endpoint as hypothesized node if not already in graph.
        Attrs: status="hypothesized", times_probed=0, last_status_code=None
        Call _broadcast() after.
        """
        if endpoint not in self.G:
            self.G.add_node(endpoint, status="hypothesized", times_probed=0, last_status_code=None)
            self._broadcast()

    def update_node(self, endpoint: str, status: str, **attrs):
        """
        Update node status and any extra attrs.
        Increment times_probed by 1.
        Call _broadcast() after.
        """
        if endpoint not in self.G:
            self.add_hypothesis(endpoint)
        self.G.nodes[endpoint]["status"] = status
        self.G.nodes[endpoint]["times_probed"] = self.G.nodes[endpoint].get("times_probed", 0) + 1
        for k, v in attrs.items():
            self.G.nodes[endpoint][k] = v
        self._broadcast()

    def add_dependency(self, from_ep: str, to_ep: str, label: str = ""):
        """Add directed edge. Both nodes must exist (add_hypothesis if not)."""
        self.add_hypothesis(from_ep)
        self.add_hypothesis(to_ep)
        self.G.add_edge(from_ep, to_ep, label=label, confirmed=False)
        self._broadcast()

    def store_auth_token(self, endpoint: str, token: str):
        """Store token string keyed by the endpoint that provided it."""
        self.auth_tokens[endpoint] = token

    def get_auth_token(self, endpoint: str) -> Optional[str]:
        """Return stored token or None."""
        return self.auth_tokens.get(endpoint)

    def get_all_tokens(self) -> dict:
        """Return copy of auth_tokens dict."""
        return dict(self.auth_tokens)

    def to_json(self) -> dict:
        """
        Serialize for websocket broadcast.
        Must always succeed — catch serialization errors.
        Format:
        {
          "nodes": [{"id": ep, "status": s, "times_probed": n, ...}],
          "edges": [{"source": a, "target": b, "label": l, "confirmed": bool}],
          "auth_tokens_acquired": n,
          "step_count": n
        }
        """
        nodes = [{"id": n, **self.G.nodes[n]} for n in self.G.nodes]
        edges = [{"source": u, "target": v, **self.G.edges[u, v]} for u, v in self.G.edges]
        return {
            "nodes": nodes,
            "edges": edges,
            "auth_tokens_acquired": len(self.auth_tokens),
            "step_count": self.step_count,
        }

    def to_observation(self) -> dict:
        """
        Compact representation for policy prompt.
        Returns:
        {
          "discovered_endpoints": [...],
          "hypothesized_endpoints": [...],
          "failed_endpoints": [...],
          "dependency_edges": [{"from": a, "to": b, "label": l}],
          "auth_tokens_acquired": n,
          "episode_step": n
        }
        """
        nodes = self.G.nodes
        return {
            "discovered_endpoints": [n for n in nodes if nodes[n].get("status") in ("discovered", "mapped")],
            "hypothesized_endpoints": [n for n in nodes if nodes[n].get("status") == "hypothesized"],
            "failed_endpoints": [n for n in nodes if nodes[n].get("status") == "failed"],
            "dependency_edges": [
                {"from": u, "to": v, "label": self.G.edges[u, v].get("label", "")}
                for u, v in self.G.edges
            ],
            "auth_tokens_acquired": len(self.auth_tokens),
            "episode_step": self.step_count,
        }

    def reset(self):
        """Clear all state for new episode."""
        self.G.clear()
        self.auth_tokens = {}
        self.step_count = 0
        self.red_herring_steps = 0
        self.total_steps = 0
