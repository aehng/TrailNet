import argparse
import _bootstrap  # noqa: F401
from trailnet.config import load_config, select_areas
from trailnet.infer import predict_area

parser = argparse.ArgumentParser(description="Predict TrailNet probability rasters")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--area")
group.add_argument("--all-areas", action="store_true")
parser.add_argument("--config")
parser.add_argument("--checkpoint")
args = parser.parse_args()
config = load_config(args.config)
for selected in select_areas(config, args.area, args.all_areas):
    print(predict_area(config, selected, args.checkpoint))
