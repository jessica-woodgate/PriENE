"""
Run a sweep: python -m src.sweep.run configs/sweeps/<sweep>.yaml [--workers N] [--dry-run]
Stop it at any time (Ctrl+C); running it again skips finished runs and continues unfinished ones.
"""
import argparse
from .spec import load
from .runner import SweepRunner

def main(argv=None):
    parser = argparse.ArgumentParser(description="Run a training parameter sweep")
    parser.add_argument("sweep", help="the sweep file (configs/sweeps/*.yaml)")
    parser.add_argument("--workers", type=int, default=1, help="runs at the same time (default 1)")
    parser.add_argument("--dry-run", action="store_true", help="list the runs still to do without running them")
    args = parser.parse_args(argv)
    SweepRunner(load(args.sweep), workers=args.workers).run_all(dry_run=args.dry_run)

if __name__ == "__main__":
    main()
