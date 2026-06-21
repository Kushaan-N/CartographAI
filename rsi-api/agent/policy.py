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
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from agent.actions import Action, ActionSpace

MODEL_NAME = "google/gemma-3-1b-it"


class Policy:
    def __init__(self, config: dict):
        """
        Load Gemma-3-4B-IT from HuggingFace and attach LoRA.

        Steps:
        1. AutoTokenizer.from_pretrained(MODEL_NAME)
        2. AutoModelForCausalLM.from_pretrained(
               MODEL_NAME,
               torch_dtype=torch.bfloat16,
               device_map="auto"
           )
        3. lora_config = LoraConfig(
               r=config["model"]["lora_rank"],
               lora_alpha=config["model"]["lora_alpha"],
               lora_dropout=config["model"]["lora_dropout"],
               target_modules=config["model"]["target_modules"],
               task_type=TaskType.CAUSAL_LM,
               bias="none"
           )
        4. self.model = get_peft_model(base_model, lora_config)
        5. self.model.print_trainable_parameters()
        6. self.action_space = ActionSpace()
        """
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

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
        fragility_line = ""
        if observation.get("fragility_warning"):
            consec = observation.get("consecutive_4xx", 0)
            fragility_line = f"WARNING: {consec} consecutive 4xx responses. API locks at {self.fragility_threshold}. Be deliberate.\n"

        messages = [{
            "role": "user",
            "content": (
                f"You are an API exploration agent. Discover all endpoints of an undocumented API.\n\n"
                f"Step: {observation.get('episode_step', 0)}\n"
                f"Discovered: {observation.get('discovered_endpoints', [])}\n"
                f"Hypothesized: {observation.get('hypothesized_endpoints', [])}\n"
                f"Auth tokens acquired: {observation.get('auth_tokens_acquired', 0)}\n"
                f"Last response: status={observation.get('last_response', {}).get('status_code', 'none')} "
                f"body={str(observation.get('last_response', {}).get('body', ''))[:300]}\n"
                f"{fragility_line}"
                f"\nOutput JSON only, no explanation:\n"
                f'{{\"method\": \"GET\", \"endpoint\": \"/path\", \"headers\": {{}}, \"body\": null}}'
            ),
        }]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

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
        prompt = self._build_prompt(observation)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        action = self.action_space.from_model_output(text)
        return action if action else self.action_space.sample_random()

    def log_prob(self, observation: dict, action) -> torch.Tensor:
        """
        Compute log probability of action given observation.
        Used by GRPOTrainer.compute_loss() in training/grpo.py.
        """
        from agent.actions import Action as ActionClass
        if isinstance(action, dict):
            action = ActionClass(
                method=action["method"],
                endpoint=action["endpoint"],
                headers=action.get("headers", {}),
                body=action.get("body"),
            )
        prompt = self._build_prompt(observation)
        completion = action.to_prompt_repr()
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
        logits = outputs.logits
        shift_logits = logits[:, prompt_len - 1:-1, :]
        shift_labels = inputs["input_ids"][:, prompt_len:]
        if shift_labels.shape[1] == 0:
            return torch.tensor(0.0, requires_grad=True, device=self.device)
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(2)).squeeze(2)
        return token_log_probs.sum()

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
        self.model = PeftModel.from_pretrained(base, path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)

    def update(self, batch, advantages) -> float:
        """Stub — weight updates owned by training/grpo.py via optimizer."""
        return 0.0
