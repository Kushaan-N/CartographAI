"""
Entry point for live demo server. Runs entirely locally.
Download checkpoint first:
    modal volume get rsi-api-checkpoints /checkpoints/latest ./checkpoints/latest
Usage: python run_demo.py --checkpoint checkpoints/latest --port 8080
"""
import argparse
from demo.server import DemoServer


def main():
    parser = argparse.ArgumentParser(description="RSI-API Live Demo")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Local path to downloaded LoRA checkpoint",
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--level",
        type=int,
        default=3,
        help="Curriculum level for demo API (1-4)",
    )
    args = parser.parse_args()

    server = DemoServer(
        checkpoint=args.checkpoint,
        port=args.port,
        level=args.level,
    )
    server.run()


if __name__ == "__main__":
    main()
