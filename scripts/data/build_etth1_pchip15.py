import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_provider.ett_pchip15 import build_etth1_pchip15_cache, default_etth1_pchip15_cache_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Build ETTh1 PCHIP 15min cache files.")
    parser.add_argument("--root_path", default="dataset/ETT-small", help="Directory containing ETTh1.csv")
    parser.add_argument("--data_path", default="ETTh1.csv", help="Hourly source CSV filename")
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="Output cache directory. Defaults to <root_path>/pchip15_cache/<csv_name>",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cache_dir = args.cache_dir or default_etth1_pchip15_cache_dir(args.root_path, args.data_path)
    metadata = build_etth1_pchip15_cache(
        root_path=args.root_path,
        data_path=args.data_path,
        cache_dir=cache_dir,
    )
    summary = {
        "cache_dir": os.path.abspath(cache_dir),
        "raw_file": metadata["raw_file"],
        "split_sample_counts": metadata["split_sample_counts"],
        "real_target_points_per_sample": metadata["real_target_points_per_sample"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
