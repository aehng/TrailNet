import argparse
import _bootstrap  # noqa: F401
from trailnet.eval import evaluate, parse_threshold_list


def main():
    parser = argparse.ArgumentParser(description="Evaluate on held-out geographic blocks")
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--thresholds",
        help="Comma-separated decision thresholds, e.g. 0.60,0.65,0.90",
    )
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint, thresholds=parse_threshold_list(args.thresholds))


if __name__ == "__main__":
    main()
