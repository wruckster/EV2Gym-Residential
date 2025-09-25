"""Utility script to profile `train_tianshou.py` and surface bottlenecks.

This wraps `train_tianshou.main()` with `cProfile` so you can capture the CPU
hotspots of a full (or truncated) training run. Use small configs or
override the training length so the profiling cycle finishes quickly.
"""

from __future__ import annotations

import argparse
import cProfile
import contextlib
import pstats
from pathlib import Path
from typing import Iterable, Optional

import train_tianshou


def _run_training(config_path: Path, repeats: int) -> None:
    """Invoke `train_tianshou.main()` `repeats` times for profiling."""
    for _ in range(repeats):
        train_tianshou.main(str(config_path))


def _print_filtered_stats(
    stats: pstats.Stats,
    patterns: Iterable[str],
    limit: Optional[int],
) -> None:
    """Print profiler statistics filtered by the provided substrings."""
    for pattern in patterns:
        print("\n" + "=" * 80)
        print(f"Top entries matching '{pattern}':")
        print("=" * 80)
        stats.print_stats(pattern, limit)


def profile_training(
    config_path: Path,
    repeats: int,
    sort_by: str,
    limit: Optional[int],
    dump_path: Optional[Path],
    filters: Iterable[str],
    show_callers: bool,
    strip_dirs: bool,
) -> None:
    """Run profiling and report the major hotspots."""
    profiler = cProfile.Profile()
    profiler.enable()
    _run_training(config_path=config_path, repeats=repeats)
    profiler.disable()

    if dump_path is not None:
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(str(dump_path))

    stats = pstats.Stats(profiler)
    if strip_dirs:
        stats.strip_dirs()
    stats.sort_stats(sort_by)

    if filters:
        _print_filtered_stats(stats, filters, limit)
    else:
        stats.print_stats(limit)

    if show_callers:
        print("\n" + "=" * 80)
        print("Top callers:")
        print("=" * 80)
        stats.print_callers(limit)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile train_tianshou.py to identify bottlenecks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("train_config.yaml"),
        help="Path to the YAML configuration consumed by train_tianshou.py.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of times to run the training loop under the profiler.",
    )
    parser.add_argument(
        "--sort",
        type=str,
        default="cumtime",
        help="Sort key for cProfile stats (e.g. cumtime, tottime, ncalls).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=30,
        help="Limit the number of rows displayed in profiler output (None shows all).",
    )
    parser.add_argument(
        "--dump",
        type=Path,
        default=None,
        help="Optional path to write raw profiler stats (.prof) for later inspection.",
    )
    parser.add_argument(
        "--filter",
        dest="filters",
        nargs="*",
        default=["train_tianshou", "ev2gym", "tianshou"],
        help=(
            "Restrict printed stats to entries whose path/function contains these substrings. "
            "Pass an empty string ("") to disable filtering."
        ),
    )
    parser.add_argument(
        "--no-callers",
        dest="show_callers",
        action="store_false",
        help="Do not print caller information (saves output).",
    )
    parser.add_argument(
        "--strip-dirs",
        action="store_true",
        help="Strip directory information from file paths in profiler output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    filters = [] if args.filters == [""] else args.filters
    limit = None if args.limit <= 0 else args.limit

    with contextlib.suppress(KeyboardInterrupt):
        profile_training(
            config_path=args.config,
            repeats=args.repeats,
            sort_by=args.sort,
            limit=limit,
            dump_path=args.dump,
            filters=filters,
            show_callers=args.show_callers,
            strip_dirs=args.strip_dirs,
        )


if __name__ == "__main__":
    main()
