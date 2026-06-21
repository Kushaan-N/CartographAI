# RSI-API — Claude Code Guide

## What this project is
An RL system that trains Gemma-3-4B-IT + LoRA to reverse-engineer undocumented APIs through active probing. Given only an IP:port, the agent must output a working Python client that maximizes branch coverage of the target API's route handlers. The training environment recursively hardens itself based on the agent's failure modes — this is the RSI loop. All GPU training and episode collection runs on Modal. Model weights pulled from HuggingFace at container build time.

## Architecture in one paragraph
The curriculum engine (curriculum/) procedurally generates Flask APIs with randomized auth schemes and endpoint dependencies. The agent (agent/) maintains a NetworkX working memory graph of discovered endpoints and uses a constrained action space to probe them. The verifier (verifier/) instruments target APIs with coverage.py and returns deterministic branch coverage scores in under 200ms. The GRPO trainer (training/) uses group contrastive reward signal across 4 parallel probe trajectories per API. All Modal functions are defined in modal_app.py — this is the only file that imports modal. The demo layer (demo/) broadcasts real-time graph updates over websocket to a d3.js visualization. RSI loop: failure_analyzer clusters failed episodes → scheduler adjusts difficulty → factory generates harder APIs → repeat.

## Model details
- Base model: google/gemma-3-1b-it
- Source: HuggingFace (official Google release)
- LoRA rank: 16, alpha: 32, dropout: 0.05
- Target modules: q_proj, v_proj, k_proj, o_proj
- Training dtype: bfloat16
- Fits on A100 40GB with LoRA + batch_size=32 comfortably
- CRITICAL: Always use tokenizer.apply_chat_template() for ALL prompt construction. Never manually format prompt strings. Gemma-3 uses specific special tokens that must come from the tokenizer.
- Do NOT use gemma-3-4b (base). Must be the -it (instruction-tuned) variant.
- Gemma-3 requires transformers>=4.50.0

## Modal architecture
- modal_app.py is the ONLY file that imports modal. No modal imports anywhere else.
- GPU function: train_step() — A100 40GB, performs one GRPO weight update
- CPU functions: collect_episode() — 2 CPUs per worker, runs one agent episode
- Parallel collection: collect_episode.map() runs 50 episodes simultaneously
- Volume: modal.Volume named "rsi-api-checkpoints" for checkpoint persistence
- Secret: modal.Secret named "huggingface-secret" stores HF_TOKEN
- Image: built once with all deps including transformers>=4.50.0, cached by Modal
- Cost estimate: A100 40GB ~$3/hr x 12h = ~$36. CPU workers ~$10. Total ~$46 of $250 budget.

## File ownership
- Terminal 1: curriculum/ — factory.py, auth_schemes.py, scheduler.py, failure_analyzer.py
- Terminal 2: agent/ — policy.py, memory.py, actions.py, error_forensics.py
- Terminal 3: verifier/ + training/ — coverage_runner.py, reward.py, episode.py, buffer.py, grpo.py, loop.py
- Terminal 4: modal_app.py + verifier/sandbox.py
- Terminal 5: demo/ + tests/ + integration

## One-time setup sequence (run before anything else)
```bash
# 1. Install modal locally
pip install modal

# 2. Authenticate modal
modal token new

# 3. Create HuggingFace secret in Modal
modal secret create huggingface-secret HF_TOKEN=your_token_here

# 4. Create checkpoint volume
modal volume create rsi-api-checkpoints

# 5. Deploy modal app (after modal_app.py is implemented)
modal deploy modal_app.py

# 6. Install local deps (for demo + tests only)
pip install flask flask-cors coverage networkx fastapi uvicorn websockets "transformers>=4.50.0" peft torch trl accelerate bitsandbytes pyyaml pytest requests python-multipart
```

## Running the project
```bash
# Start training (orchestrates Modal remotely from local machine)
python run_training.py --config configs/train_config.yaml

# Download checkpoint locally for demo
modal volume get rsi-api-checkpoints /checkpoints/latest ./checkpoints/latest

# Run demo server locally
python run_demo.py --checkpoint checkpoints/latest --port 8080

# Run tests locally (no GPU needed)
pytest tests/ -v
```

## Critical invariants — never break these
1. coverage_runner.py must return in under 200ms. Episode throughput depends on this.
2. Action space must stay constrained — (method, endpoint, headers, body) tuples only. No free-form generation.
3. Reward syntax_valid is a HARD GATE — invalid client returns exactly 0.0, not a soft multiplier.
4. WorkingMemory.to_json() must always succeed — no non-serializable objects ever stored in the graph.
5. factory.py must generate a fresh unique API on every call — no reuse across episodes.
6. ALWAYS use tokenizer.apply_chat_template() in policy.py — never manual prompt strings.
7. modal_app.py is the ONLY modal import. All other files are pure Python.
8. Checkpoints write to Modal Volume. loop.py never writes to local disk during training.
9. Never introduce new `raise NotImplementedError` stubs — all functions must be fully implemented.

## Checkpointing
- LoRA adapter saves every 500 episodes to Modal Volume: /checkpoints/episode_{n}/
- /checkpoints/latest/ always points to most recent
- Metrics log: /checkpoints/logs/metrics.jsonl
- Download command: modal volume get rsi-api-checkpoints /checkpoints/latest ./checkpoints/latest

## Key design decisions

### Rate-limit fragility mechanic
If the agent sends fragility_threshold (default 5) consecutive 4xx-producing requests, the API locks up and the episode terminates immediately with fragility_penalty (-1.0) reward. A warning signal appears in the observation at fragility_warning_threshold (default 3) consecutive failures, giving the policy one last chance to change strategy. This forces deliberate probing over brute-force guessing. In real-world APIs, aggressive probing triggers IP bans and rate limits — this mechanic ensures the trained agent develops genuinely cautious exploration behavior. api_locked episodes are tracked by failure_analyzer and up-weighted 2x in the next curriculum batch.

### Dynamic episode budgets
Episodes start with a base budget of 20 steps. Each auth token acquisition grants +2 bonus steps. Each new dependency discovery grants +2 bonus steps. Hard ceiling of 35 total steps regardless of bonuses. This directly addresses the Level 4 bottleneck where 15 endpoints and 5 dependency chains exceed a fixed 20-step budget. The bonus_log in episode metadata tracks exactly what triggered each bonus for debugging.

## Common failure modes and fixes
- "Gemma not following JSON format": manual prompt formatting. Fix: use apply_chat_template().
- "transformers version error with Gemma-3": upgrade to transformers>=4.50.0.
- "Modal GPU OOM": reduce batch_size to 16 in train_config.yaml.
- "Modal function timeout": reduce max_episode_steps to 15.
- "coverage.py returns 0.0 on valid client": Flask subprocess died. Check factory.py process health check.
- "GRPO loss NaN": learning rate too high. Set to 2e-5 in train_config.yaml.
- "Modal auth error": run modal token new again.
- "HuggingFace 401 on Gemma-3": HF_TOKEN wrong or license not accepted at hf.co/google/gemma-3-1b-it.
- "Reward always 0 for first 200 episodes": normal for sparse reward. Check shaped intermediates firing in metrics.jsonl.
