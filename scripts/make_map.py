import argparse
import _bootstrap  # noqa: F401
from trailnet.config import load_config, select_areas
from trailnet.viz import build_map_area

parser = argparse.ArgumentParser(description="Build a Leaflet map of predicted trails")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--area")
group.add_argument("--all-areas", action="store_true")
parser.add_argument("--config")
args = parser.parse_args()
config = load_config(args.config)
for selected in select_areas(config, args.area, args.all_areas):
    print(build_map_area(config, selected))
