"""Utility for profiling EV2Gym environment initialization."""

from __future__ import annotations

import argparse
import cProfile
import pstats
from pathlib import Path
from typing import Optional

from ev2gym.models.ev2gym_env import EV2Gym


def profile_env_initialization(
    config_path: Path,
    repeats: int,
    sort_by: str,
    dump_path: Optional[Path],
) -> None:
    """Profile repeated EV2Gym initializations and report hotspots."""

    def run_initializations() -> None:
        for _ in range(repeats):
            EV2Gym(config_file=str(config_path), verbose=False)

    profiler = cProfile.Profile()
    profiler.enable()
    run_initializations()
    profiler.disable()

    if dump_path is not None:
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(str(dump_path))

    stats = pstats.Stats(profiler).sort_stats(sort_by)
    stats.print_stats(30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile EV2Gym environment initialization."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ev2gym/example_config_files/residential_v2g.yaml"),
        help="Path to the EV2Gym YAML configuration file.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of times to instantiate the environment for averaging.",
    )
    parser.add_argument(
        "--sort",
        type=str,
        default="cumtime",
        help=(
            "Sorting key for profiler output (e.g. 'cumtime', 'tottime', "
            "'ncalls')."
        ),
    )
    parser.add_argument(
        "--dump",
        type=Path,
        default=None,
        help="Optional path to write raw profiler stats (.prof).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile_env_initialization(
        config_path=args.config,
        repeats=args.repeats,
        sort_by=args.sort,
        dump_path=args.dump,
    )


if __name__ == "__main__":
    main()
