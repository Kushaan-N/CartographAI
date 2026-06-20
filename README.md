# CartographAI
train a small LLM (Qwen2.5-3B + LoRA) to reverse-engineer completely undocumented APIs from scratch. Given only an IP address, it must produce a working Python client by actively probing endpoints, inferring auth schemes from error bodies, and building a dependency map of the API's structure.
