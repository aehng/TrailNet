import argparse
import _bootstrap  # noqa: F401
from trailnet.eval import evaluate

parser = argparse.ArgumentParser(description="Evaluate on held-out geographic blocks")
parser.add_argument("--config")
parser.add_argument("--checkpoint")
args = parser.parse_args()
evaluate(args.config, args.checkpoint)
