"""
LLM policy: Gemma-3-4B-IT + LoRA.

CRITICAL RULES:
1. Always use tokenizer.apply_chat_template() — never manual prompt strings.
2. Gemma-3 requires transformers>=4.50.0.
3. Load with torch_dtype=torch.bfloat16, device_map="auto".
4. Save LoRA adapter only — not the full base model.

Interface:
- sample_action(observation) -> Action
- log_prob(observation, action) -> torch.Tensor
- save(path): LoRA adapter only
- load(path): from saved LoRA adapter
- update() is a stub — owned by training/grpo.py
"""
import torch
import json
import random
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from agent.actions import Action, ActionSpace

MODEL_NAME = "google/gemma-3-1b-it"


def decayed_epsilon(config: dict, episode_count: int) -> float:
    """Linearly anneal the exploration epsilon over training so the LLM policy
    takes over from the structured-exploration scaffold as it learns.

    Reads training.exploration_epsilon_start / _end / _decay_episodes. Falls back
    to the constant training.exploration_epsilon when start/end aren't configured,
    so existing configs keep their fixed behaviour.
    """
    t = config.get("training", {}) if isinstance(config, dict) else {}
    const = t.get("exploration_epsilon", 0.0)
    start = t.get("exploration_epsilon_start", const)
    end = t.get("exploration_epsilon_end", start)
    horizon = t.get("exploration_decay_episodes") or t.get("total_episodes") or 1
    frac = min(1.0, max(0.0, episode_count / max(horizon, 1)))
    return start + (end - start) * frac


def build_agent_prompt(observation: dict, tokenizer, fragility_threshold: int = 5) -> str:
    """
    Build the agent prompt from an observation using apply_chat_template.

    Single source of truth for prompt formatting, shared by Policy.sample_action
    (inference) and the SFT warm-up data formatter (training) so the two can
    never drift. NEVER construct the prompt string manually elsewhere.
    """
    from agent.actions import build_candidates

    fragility_line = ""
    if observation.get("fragility_warning"):
        consec = observation.get("consecutive_4xx", 0)
        fragility_line = (
            f"WARNING: {consec} consecutive 4xx responses. API locks at "
            f"{fragility_threshold}. Be deliberate.\n"
        )

    candidates = build_candidates(observation)
    discovered = set(observation.get("discovered_endpoints", []) or [])
    menu = "\n".join(
        f"{i}: {ep}{'  (probed OK)' if ep in discovered else ''}"
        for i, ep in enumerate(candidates)
    )

    messages = [{
        "role": "user",
        "content": (
            "You are an API exploration agent. Probe endpoints to maximize branch "
            "coverage of an undocumented API. Probing '/' returns the list of real "
            "endpoints; probe endpoints you have NOT yet discovered, using auth.\n\n"
            f"Step: {observation.get('episode_step', 0)}\n"
            f"Auth tokens acquired: {observation.get('auth_tokens_acquired', 0)}\n"
            f"Last response: status={observation.get('last_response', {}).get('status_code', 'none')} "
            f"body={str(observation.get('last_response', {}).get('body', ''))[:300]}\n"
            f"{fragility_line}"
            "\nCandidate endpoints to probe:\n"
            f"{menu}\n\n"
            "Pick the single best candidate to probe next. Prefer ones not yet "
            "discovered. Output JSON only, no explanation:\n"
            '{"choice": <number>, "method": "GET", "auth": true}'
        ),
    }]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


class Policy:
    def __init__(self, config: dict, checkpoint_path: str = None):
        """
        Load Gemma-3-1B-IT from HuggingFace and attach LoRA.

        If checkpoint_path is given, load that saved LoRA adapter (e.g. SFT
        warm-up weights) instead of a freshly-initialized one. Otherwise attach
        a new LoRA from config.
        """
        load_from = checkpoint_path or MODEL_NAME
        self.tokenizer = AutoTokenizer.from_pretrained(load_from)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        # Frozen reference policy for the GRPO KL anchor. Without a reference to
        # pull toward, the bare policy-gradient update collapses the policy's
        # entropy to a single deterministic action (the observed 7.1%/1-branch
        # floor). The reference is a SECOND, non-trainable copy of the same
        # warmup adapter, so KL(pi_theta || pi_ref) keeps the trained policy near
        # the competent warmup init instead of letting it drift to a degenerate
        # one. If no checkpoint is given (fresh LoRA) we fall back to the base
        # model (adapter disabled) as the reference.
        self._has_ref_adapter = False
        if checkpoint_path:
            self.model = PeftModel.from_pretrained(
                base_model, checkpoint_path, is_trainable=True
            )
            try:
                self.model.load_adapter(
                    checkpoint_path, adapter_name="reference", is_trainable=False
                )
                self.model.set_adapter("default")
                self._has_ref_adapter = True
            except Exception as e:  # pragma: no cover - environment dependent
                print(f"[policy] reference adapter load failed ({e}); "
                      f"using base-model KL reference instead")
        else:
            lora_config = LoraConfig(
                r=config["model"]["lora_rank"],
                lora_alpha=config["model"]["lora_alpha"],
                lora_dropout=config["model"]["lora_dropout"],
                target_modules=config["model"]["target_modules"],
                task_type=TaskType.CAUSAL_LM,
                bias="none",
            )
            self.model = get_peft_model(base_model, lora_config)
        self.model.print_trainable_parameters()
        self.action_space = ActionSpace()
        self.device = next(self.model.parameters()).device
        self.fragility_threshold = config["training"].get("fragility_threshold", 5)
        self.fragility_warning_threshold = config["training"].get("fragility_warning_threshold", 3)
        # Epsilon-greedy structured exploration. SFT teaches the policy to emit
        # plausible actions but not to *copy* endpoints from the observation
        # (hard for a 1B model via LoRA SFT). With prob exploration_epsilon we
        # instead probe the discovery index "/" or an un-probed endpoint the
        # observation already surfaced, with auth headers. This bootstraps the
        # reward variance GRPO needs; GRPO then trains the LLM toward these
        # actions (their log_prob is computed under the model as usual).
        self.exploration_epsilon = config["training"].get("exploration_epsilon", 0.0)
        # When True, sample_action decodes greedily (do_sample=False). Used by the
        # exploration-off pure-policy eval so the measurement reflects the policy's
        # MODE, not a temperature-0.7 sample — temperature noise was masking
        # whether the learned policy actually improved. Off during collection
        # (we want stochastic rollouts there for advantage variance).
        self.eval_greedy = False

    def _build_prompt(self, observation: dict) -> str:
        """
        Build prompt using tokenizer.apply_chat_template().
        NEVER construct the prompt string manually.

        Message structure:
        messages = [{
            "role": "user",
            "content": (
                f"You are an API exploration agent. Your goal is to discover "
                f"all endpoints of an undocumented API and produce a working client.\\n\\n"
                f"Step: {observation['episode_step']}\\n"
                f"Discovered endpoints: {observation['discovered_endpoints']}\\n"
                f"Hypothesized endpoints: {observation['hypothesized_endpoints']}\\n"
                f"Auth tokens acquired: {observation['auth_tokens_acquired']}\\n"
                f"Last response: status={observation['last_response']['status_code']} "
                f"body={str(observation['last_response']['body'])[:200]}\\n\\n"
                f"Output a JSON object with exactly these keys:\\n"
                f'{{\"method\": \"GET|POST|PUT|DELETE|PATCH\", '
                f'\"endpoint\": \"/path\", '
                f'\"headers\": {{}}, '
                f'\"body\": null}}'
            )
        }]

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        """
        return build_agent_prompt(observation, self.tokenizer, self.fragility_threshold)

    def sample_action(self, observation: dict) -> Action:
        """
        Build prompt -> tokenize -> generate -> parse action.

        Generation params:
        - max_new_tokens=128
        - temperature=0.7
        - do_sample=True
        - pad_token_id=self.tokenizer.eos_token_id

        Parse output with self.action_space.from_model_output().
        If parsing fails, return self.action_space.sample_random() as fallback.
        """
        if self.exploration_epsilon > 0 and random.random() < self.exploration_epsilon:
            explore = self._explore_action(observation)
            if explore is not None:
                return explore

        prompt = self._build_prompt(observation)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            if self.eval_greedy:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        from agent.actions import build_candidates, parse_selection, selection_to_action
        candidates = build_candidates(observation)
        action = parse_selection(text, candidates)
        if action is not None:
            return action
        # Constrained fallback: pick a random candidate (never a hallucinated
        # endpoint) — prefer an un-probed one to keep discovery moving.
        discovered = set(observation.get("discovered_endpoints", []) or [])
        unprobed = [i for i, ep in enumerate(candidates) if ep not in discovered]
        idx = random.choice(unprobed) if unprobed else 0
        return selection_to_action(idx, "GET", True, candidates)

    _EXPLORE_HEADERS = {
        "Authorization": "Bearer test123",
        "X-API-Key": "test123",
        "X-Service-Token": "test123",
    }

    def _explore_action(self, observation: dict) -> Optional[Action]:
        """Structured exploration: probe the discovery index, then un-probed
        endpoints the observation has surfaced. Returns None if nothing useful
        to probe (caller falls back to LLM generation)."""
        hyp = observation.get("hypothesized_endpoints", []) or []
        discovered = set(observation.get("discovered_endpoints", []) or [])
        # Probe "/" first — it returns the endpoint index, populating hypotheses.
        if "/" in hyp and "/" not in discovered:
            return Action("GET", "/", {}, None)
        unprobed = [e for e in hyp if e not in discovered and e != "/"]
        if not unprobed:
            return None
        ep = random.choice(unprobed)
        return Action("GET", ep, dict(self._EXPLORE_HEADERS), None)

    def _activate_policy(self):
        """Make the trainable ('default') adapter the active one. No-op when there
        is no separate reference adapter."""
        if self._has_ref_adapter:
            self.model.set_adapter("default")

    def _score_tokens(self, observation: dict, action):
        """Score the SELECTION (choice/method/auth) the action corresponds to,
        under whichever adapter is currently active. Returns
        (sum_token_log_prob, mean_token_entropy) over the completion tokens."""
        from agent.actions import (Action as ActionClass, build_candidates,
                                    action_to_selection, selection_json)
        if isinstance(action, dict):
            action = ActionClass(
                method=action["method"],
                endpoint=action["endpoint"],
                headers=action.get("headers", {}),
                body=action.get("body"),
            )
        prompt = self._build_prompt(observation)
        # Same candidate menu the prompt presents.
        candidates = build_candidates(observation)
        completion = selection_json(action_to_selection(action, candidates))
        full_text = prompt + completion
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(self.device)
        prompt_ids = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)["input_ids"]
        prompt_len = prompt_ids.shape[1]
        outputs = self.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        # float32 for numerically-stable log_softmax/entropy (logits are bf16).
        shift_logits = outputs.logits[:, prompt_len - 1:-1, :].float()
        shift_labels = inputs["input_ids"][:, prompt_len:]
        if shift_labels.shape[1] == 0:
            z = torch.tensor(0.0, requires_grad=True, device=self.device)
            return z, z
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(2)).squeeze(2)
        # Predictive entropy at the completion positions (encourages the policy
        # to stay spread out instead of collapsing onto a single action).
        entropy = -(log_probs.exp() * log_probs).sum(-1).mean()
        return token_log_probs.sum(), entropy

    def log_prob(self, observation: dict, action) -> torch.Tensor:
        """
        Compute log probability of action given observation (trainable policy).
        Used by GRPOTrainer.compute_loss() in training/grpo.py.
        """
        self._activate_policy()
        lp, _ = self._score_tokens(observation, action)
        return lp

    def score_policy(self, observation: dict, action):
        """(sum_log_prob, mean_entropy) under the trainable policy — both carry
        gradient. Used for the GRPO policy-gradient term and entropy bonus."""
        self._activate_policy()
        return self._score_tokens(observation, action)

    def ref_log_prob(self, observation: dict, action) -> torch.Tensor:
        """Detached log_prob of the action under the FROZEN reference policy
        (the warmup adapter, or the base model if no checkpoint). Used for the
        GRPO KL anchor."""
        with torch.no_grad():
            if self._has_ref_adapter:
                self.model.set_adapter("reference")
                lp, _ = self._score_tokens(observation, action)
                self.model.set_adapter("default")
            else:
                with self.model.disable_adapter():
                    lp, _ = self._score_tokens(observation, action)
        return lp.detach()

    def save(self, path: str):
        """
        Save LoRA adapter weights only.
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        Do NOT save full base model weights — too large.
        """
        import os
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path: str):
        """
        Load LoRA adapter from path.
        base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
        self.model = PeftModel.from_pretrained(base, path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        """
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        # is_trainable=True is REQUIRED: PeftModel.from_pretrained loads adapters
        # FROZEN by default (requires_grad=False). Without this, train_step()
        # reloads the checkpoint and GRPO updates NOTHING (grad_norm 0) — training
        # silently becomes a no-op after the first checkpoint.
        self.model = PeftModel.from_pretrained(base, path, is_trainable=True)
        self.tokenizer = AutoTokenizer.from_pretrained(path)

    def update(self, batch, advantages) -> float:
        """Stub — weight updates owned by training/grpo.py via optimizer."""
        return 0.0
