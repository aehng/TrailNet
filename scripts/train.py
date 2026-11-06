import argparse
import _bootstrap  # noqa: F401
from trailnet.config import load_config, select_areas
from trailnet.train import run_training


def main():
    parser = argparse.ArgumentParser(description="Train TrailNet from prepared spatial tiles")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--area")
    group.add_argument("--all-areas", action="store_true")
    parser.add_argument("--config", help="Config YAML path (defaults to repository config.yaml)")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.area or args.all_areas:
        selected = select_areas(config, args.area, args.all_areas)
        area_names = [area["name"] for area in selected]
    else:
        area_names = None  # default: pool all configured areas
    run_training(args.config, area_names=area_names)


if __name__ == "__main__":
    main()
