"""
FastAPI + websocket demo server. Runs locally after training.
Loads checkpoint, runs agent against live mock API, broadcasts graph updates.
"""
import asyncio
import json
import os
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import uvicorn


class DemoServer:
    def __init__(self, checkpoint: Optional[str], port: int, level: int):
        self.checkpoint = checkpoint
        self.port = port
        self.level = level
        self.app = FastAPI()
        self.connections: list[WebSocket] = []
        self.demo_api = None
        self.memory = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._setup_routes()

    def _setup_routes(self):
        static_dir = os.path.join(os.path.dirname(__file__), "static")
        self.app.mount("/static", StaticFiles(directory=static_dir), name="static")

        @self.app.on_event("startup")
        async def _capture_loop():
            self._loop = asyncio.get_event_loop()

        @self.app.get("/")
        async def index():
            return FileResponse(os.path.join(static_dir, "index.html"))

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.connections.append(websocket)
            try:
                while True:
                    await websocket.receive_text()
            except WebSocketDisconnect:
                if websocket in self.connections:
                    self.connections.remove(websocket)

        @self.app.post("/inject")
        async def inject(request: Request):
            from demo.inject import handle_inject
            body = await request.json()
            result = handle_inject(body["route_code"], self.demo_api, self.memory)
            return JSONResponse(result)

        @self.app.get("/metrics")
        async def metrics():
            log_path = "logs/metrics.jsonl"
            if not os.path.exists(log_path):
                return JSONResponse([])
            with open(log_path) as f:
                lines = f.readlines()[-200:]
            return JSONResponse([json.loads(l) for l in lines if l.strip()])

    async def broadcast(self, data: dict):
        if not self.connections:
            return
        msg = json.dumps(data)
        dead = []
        for ws in self.connections:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            if ws in self.connections:
                self.connections.remove(ws)

    def broadcast_sync(self, data: dict):
        """Thread-safe broadcast — safe to call from background threads."""
        if self._loop and self._loop.is_running():
            future = asyncio.run_coroutine_threadsafe(self.broadcast(data), self._loop)
            try:
                future.result(timeout=1.0)
            except Exception:
                pass

    def run(self):
        if self.checkpoint is None:
            # Test mode — no model, agent loop managed externally (e.g. test_demo.py)
            uvicorn.run(self.app, host="0.0.0.0", port=self.port)
            return

        import threading
        import yaml
        from agent.memory import WorkingMemory
        from curriculum.factory import generate_api
        from training.episode import run_episode

        with open("configs/train_config.yaml") as f:
            config = yaml.safe_load(f)

        # Generate demo API
        self.demo_api = generate_api(level=self.level, config=config)

        # Setup memory with broadcast callback
        self.memory = WorkingMemory()
        loop = asyncio.new_event_loop()

        def sync_broadcast(data):
            try:
                asyncio.run_coroutine_threadsafe(
                    self.broadcast({**data, "type": "graph"}), loop
                )
            except Exception:
                pass

        self.memory.set_broadcast_callback(sync_broadcast)

        # Load policy (stub — real policy requires GPU checkpoint)
        try:
            from agent.policy import Policy
            policy = Policy(config)
            policy.load(self.checkpoint)
        except Exception:
            from agent.actions import ActionSpace

            class _FallbackPolicy:
                def __init__(self):
                    self.action_space = ActionSpace()

                def sample_action(self, obs):
                    return self.action_space.sample_random()

            policy = _FallbackPolicy()

        def agent_loop():
            while True:
                try:
                    result = run_episode(policy, self.demo_api, config)
                    asyncio.run_coroutine_threadsafe(
                        self.broadcast({
                            "type": "coverage",
                            "coverage": result["branch_coverage"],
                            "level": self.level,
                        }),
                        loop,
                    )
                    self.memory.reset()
                except Exception as e:
                    print(f"Demo episode error: {e}")
                import time
                time.sleep(2)

        t = threading.Thread(target=agent_loop, daemon=True)
        t.start()

        uvicorn.run(self.app, host="0.0.0.0", port=self.port, loop="asyncio")
