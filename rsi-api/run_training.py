"""
Entry point for RSI training loop.
Orchestrates Modal GPU training from local machine.
Usage: python run_training.py --config configs/train_config.yaml
       python run_training.py --config configs/train_config.yaml --warmup
"""
import argparse
import yaml
from training.loop import TrainingLoop


def main():
    parser = argparse.ArgumentParser(description="RSI-API Training")
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--resume", default=None,
                        help="Modal Volume path to resume from")
    parser.add_argument("--warmup", action="store_true",
                        help="Run SFT warmup on ToolBench before GRPO training")
    parser.add_argument("--warmup-config", default="configs/warmup_config.yaml",
                        help="Path to warmup config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.warmup:
        with open(args.warmup_config) as f:
            import yaml as _yaml
            warmup_cfg = _yaml.safe_load(f)
        config["warmup"] = warmup_cfg["warmup"]

        print("=== Starting SFT Warmup on ToolBench ===")
        from training.warmup import SFTWarmup
        warmup = SFTWarmup(config)
        warmup_checkpoint = warmup.run()
        print(f"=== Warmup complete. Starting GRPO from {warmup_checkpoint} ===")

        loop = TrainingLoop(config)
        loop.load_checkpoint(warmup_checkpoint)
    else:
        loop = TrainingLoop(config)

    if args.resume:
        loop.load_checkpoint(args.resume)

    loop.run()


if __name__ == "__main__":
    main()
