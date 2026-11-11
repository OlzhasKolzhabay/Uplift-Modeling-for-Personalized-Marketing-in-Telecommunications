"""Download the MegaFon uplift dataset into data/ (about 270 MB)."""

from __future__ import annotations

import argparse
from pathlib import Path

from uplift_telecom.config import Paths, repo_root
from uplift_telecom.data import download_megafon


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="download again even if the file exists")
    parser.add_argument("--dest", type=Path, default=Paths(repo_root()).data_file)
    args = parser.parse_args()
    path = download_megafon(args.dest, force=args.force)
    print(f"Dataset ready: {path} ({path.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
