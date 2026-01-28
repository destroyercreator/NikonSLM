from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

from lpbf_tracker.config import load_config
from lpbf_tracker.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="LPBF company discovery tracker")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to configuration YAML",
    )
    args = parser.parse_args()
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config = load_config(args.config)
    run_pipeline(config)


if __name__ == "__main__":
    main()
