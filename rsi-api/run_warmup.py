"""
SFT warm-up runner (constrained-selection format). Runs locally / on a GPU box.

Collects HeuristicPolicy demonstration episodes, formats them as constrained
action-selection (choice/method/auth) pairs, and SFT-trains the LoRA adapter.
Saves to checkpoints/warmup (the path run_local_smoke.py loads from).

Usage:
  python run_warmup.py [n_episodes] [--batch N] [--config configs/train_config_local.yaml]

On a real CUDA GPU you can raise --batch (8-16) for much faster SFT than the
MPS default of 4.
"""
import sys, argparse, yaml
from training.warmup import SFTWarmup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n_episodes", nargs="?", type=int, default=200)
    ap.add_argument("--config", default="configs/train_config_local.yaml")
    ap.add_argument("--warmup-config", default="configs/warmup_config.yaml")
    ap.add_argument("--batch", type=int, default=8, help="per-device train batch (raise on a real GPU)")
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--levels", default="1,2", help="comma-separated curriculum levels to collect")
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open(args.warmup_config) as f:
        config["warmup"] = yaml.safe_load(f)["warmup"]

    config["warmup"]["local_batch_size"] = args.batch
    config["warmup"]["logging_steps"] = 25
    config["warmup"]["max_seq_length"] = args.max_seq_length

    levels = tuple(int(x) for x in args.levels.split(","))
    warmup = SFTWarmup(config)
    path = warmup.run_local(n_episodes=args.n_episodes, levels=levels)
    print(f"SFT checkpoint: {path}")


if __name__ == "__main__":
    main()
