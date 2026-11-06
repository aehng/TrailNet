import argparse
import _bootstrap  # noqa: F401
from trailnet.config import load_config, select_areas
from trailnet.data.tiles import build_area_tiles


def main():
    parser = argparse.ArgumentParser(description="Prepare TrailNet raster tiles")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--area")
    group.add_argument("--all-areas", action="store_true")
    parser.add_argument("--config", help="Config YAML path (defaults to repository config.yaml)")
    args = parser.parse_args()
    config = load_config(args.config)
    for selected in select_areas(config, args.area, args.all_areas):
        print(f"Preparing {selected['name']}...")
        print(build_area_tiles(config, selected))


if __name__ == "__main__":
    main()
