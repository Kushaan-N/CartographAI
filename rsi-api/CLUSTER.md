# Running the training loop on a GPU box (no Modal)

The whole RSI-API loop is plain Python — only `modal_app.py` touches Modal, and
nothing here imports it. On a CUDA GPU you load the model **once** and run the
warmup + GRPO loop in-process: fast, free, observable. This is the recommended
way to actually answer "does GRPO improve the agent?".

## 0. Prereqs
- A CUDA GPU (anything with ~16GB+ is plenty for Gemma-3-1B + LoRA).
- Python 3.10–3.11.
- A HuggingFace token with access to `google/gemma-3-1b-it` (accept the license at
  https://huggingface.co/google/gemma-3-1b-it).

## 1. Setup
```bash
git clone <your-repo-url> && cd CartographAI/rsi-api
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN=hf_xxx            # or: huggingface-cli login
# (transformers reads HF_TOKEN / HUGGING_FACE_HUB_TOKEN automatically)
```

Quick sanity that the env is good (fast, no GPU needed):
```bash
pytest tests/ -q                 # expect ~40 passed
```

## 2. Warm up the policy (SFT)
Teaches the LLM the constrained action-selection behaviour. On a real GPU bump
the batch size for a much faster run than the Mac MPS default.
```bash
python run_warmup.py 200 --batch 16
# -> writes checkpoints/warmup/  (this is what the gate loads)
```
Expect token accuracy ~0.83 and, when validated at epsilon=0, ~20% pure-policy
coverage (the LLM exploring on its own).

## 3. Run the GRPO gate (the real experiment)
Runs the full loop — same-seed contrastive groups, GRPO updates, epsilon decay —
and prints `pure_policy_coverage` (exploration OFF) every few updates. THIS is the
go/no-go signal.
```bash
python run_local_smoke.py 800 --checkpoint checkpoints/warmup --eval-every-updates 3
# metrics also written to local_metrics.jsonl
```

### Reading the result — the one number that matters
Watch **`pure_policy_cov`** in the output (and `pure_policy_mean_coverage` in
`local_metrics.jsonl`):
- **Climbs above ~20%** over training → GRPO is improving the agent. RL works → scale up.
- **Stays flat ~20%** → the warmup is the ceiling; GRPO isn't adding (yet).
- **Drops toward 7.1%** → GRPO is *degrading* the policy → stop and debug (this is
  what the contrastive-group bug caused; that bug is now fixed in collect via seeds,
  and run_local_smoke already seeds groups correctly).

`collect_mean_coverage` is *with* the exploration scaffold and will be higher —
it's not the verdict. `pure_policy_cov` is.

## 4. Scaling up (only after the gate is green)
- More episodes: `python run_local_smoke.py 5000 ...`
- The config knobs live in `configs/train_config_local.yaml` (lora_rank 16,
  learning_rate, grpo_group_size, exploration_epsilon_start/_end/_decay_episodes).
- If GRPO looks unstable, lower `learning_rate` (5e-5 -> 1e-5) first.

---

# Getting Claude Code onto the cluster

Claude Code is a terminal CLI; it runs fine over SSH.

```bash
# on the cluster, after SSH:
node --version                       # need >= 18 ; if missing, install via nvm
npm install -g @anthropic-ai/claude-code
export ANTHROPIC_API_KEY=sk-ant-...  # simplest headless auth
cd CartographAI/rsi-api
tmux new -s claude                   # so an SSH drop doesn't kill the session
claude
```
Notes:
- If you'd rather not use an API key, just run `claude` and use the login flow —
  it prints a URL/code you open in *your local* browser (works headless).
- Needs outbound HTTPS to api.anthropic.com (set HTTPS_PROXY if the cluster is
  locked down).
- Run it inside `tmux`/`screen` so long sessions survive disconnects.
