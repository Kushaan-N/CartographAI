"""
SFT warm-up on ToolBench dataset.
Runs before GRPO to give model API interaction priors.
"""
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

MODEL_NAME = "google/gemma-3-1b-it"


class SFTWarmup:
    def __init__(self, config: dict):
        self.config = config
        self.warmup_config = config["warmup"]
        self.model_config = config["model"]
        self.tokenizer = None
        self.model = None

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        lora_config = LoraConfig(
            r=self.model_config["lora_rank"],
            lora_alpha=self.model_config["lora_alpha"],
            lora_dropout=self.model_config["lora_dropout"],
            target_modules=self.model_config["target_modules"],
            task_type=TaskType.CAUSAL_LM,
            bias="none"
        )
        self.model = get_peft_model(base_model, lora_config)
        self.model.print_trainable_parameters()

    def _format_sample(self, sample: dict):
        try:
            text = json.dumps(sample) if isinstance(sample, dict) else str(sample)
            import re
            method = "GET"
            for m in ["POST", "PUT", "DELETE", "PATCH", "GET"]:
                if m in text:
                    method = m
                    break
            paths = re.findall(r'["\'](/[a-zA-Z0-9_/.-]+)["\']', text)
            endpoint = paths[0] if paths else "/api/data"
            headers = {}
            if "Authorization" in text or "Bearer" in text:
                headers["Authorization"] = "Bearer {token}"
            if "api_key" in text.lower():
                headers["X-API-Key"] = "{key}"
            if "Content-Type" in text:
                headers["Content-Type"] = "application/json"
            completion = json.dumps({
                "method": method,
                "endpoint": endpoint,
                "headers": headers,
                "body": None
            })
            messages = [{
                "role": "user",
                "content": (
                    "You are an API exploration agent. Discover all endpoints of an undocumented API.\n\n"
                    "Step: 1\n"
                    "Discovered: []\n"
                    "Hypothesized: []\n"
                    "Auth tokens acquired: 0\n"
                    "Last response: status=none body=\n\n"
                    "Output JSON only, no explanation:\n"
                    "{\"method\": \"GET\", \"endpoint\": \"/path\", \"headers\": {}, \"body\": null}"
                )
            }]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return prompt + completion + self.tokenizer.eos_token
        except Exception:
            return None

    def load_and_filter_dataset(self):
        try:
            dataset = load_dataset("ToolBench/ToolBench", split="train")
        except Exception:
            dataset = load_dataset("gorilla-llm/APIBench", split="train")

        keywords = ["GET", "POST", "Authorization", "Bearer", "api_key", "endpoint", "url"]

        def has_api_content(sample):
            text = json.dumps(sample) if isinstance(sample, dict) else str(sample)
            return any(kw in text for kw in keywords)

        dataset = dataset.filter(has_api_content)
        max_samples = self.warmup_config["max_samples"]
        if len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))

        formatted = []
        for sample in dataset:
            text = self._format_sample(sample)
            if text:
                formatted.append({"text": text})

        from datasets import Dataset
        return Dataset.from_list(formatted)

    def run(self) -> str:
        import os
        self.load_model()
        dataset = self.load_and_filter_dataset()
        print(f"Training on {len(dataset)} samples")

        output_path = self.warmup_config["output_path"]
        os.makedirs(output_path, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=output_path,
            num_train_epochs=self.warmup_config["num_epochs"],
            per_device_train_batch_size=self.warmup_config["batch_size"],
            gradient_accumulation_steps=self.warmup_config["gradient_accumulation_steps"],
            learning_rate=self.warmup_config["learning_rate"],
            bf16=True,
            logging_steps=self.warmup_config["logging_steps"],
            save_strategy="epoch",
            report_to="none",
        )

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=self.warmup_config["max_seq_length"],
            args=training_args,
        )

        trainer.train()
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        print(f"Warmup complete. Checkpoint saved to {output_path}")
        return output_path
