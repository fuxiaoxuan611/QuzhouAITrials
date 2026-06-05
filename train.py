import argparse
import sys

import yaml

from pcse_gym.config import load_config, train_from_config
from pcse_gym.config.summary import print_config_summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train an NUE-PCSE-Gym experiment from YAML.")
    parser.add_argument("--config", required=True, help="Path to an experiment YAML file.")
    parser.add_argument("--override", action="append", default=[], help="Override a config value with key=value.")
    parser.add_argument("--dry-run", action="store_true", help="Build env/model and exit without learning.")
    parser.add_argument("--print-config", action="store_true", help="Print the merged config and exit.")

    parser.add_argument("--seed", type=int, default=None, help="Legacy alias for experiment.seed and agent.seed.")
    parser.add_argument("--nsteps", type=int, default=None, help="Legacy alias for training.total_timesteps.")
    parser.add_argument("--no-comet", action="store_true", help="Legacy alias for logging.comet.enabled=false.")
    parser.add_argument("--device", default=None, help="Legacy alias for agent.device.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    overrides = list(args.override)
    if args.dry_run:
        overrides.append("run.dry_run=true")
    if args.seed is not None:
        overrides.append(f"experiment.seed={args.seed}")
        overrides.append(f"agent.seed={args.seed}")
    if args.nsteps is not None:
        overrides.append(f"training.total_timesteps={args.nsteps}")
    if args.no_comet:
        overrides.append("logging.comet.enabled=false")
    if args.device is not None:
        overrides.append(f"agent.device={args.device}")

    config = load_config(args.config, overrides)
    print_config_summary(config)
    if args.print_config:
        print(yaml.safe_dump(config, sort_keys=False))
        return 0
    train_from_config(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
