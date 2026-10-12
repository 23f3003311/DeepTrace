"""Download the original FUNSD zip (149 train forms, 50 test forms)."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import requests

from deeptrace.funsd import count_pages, resolve_funsd_root

_URL = "https://guillaumejaume.github.io/FUNSD/dataset.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download FUNSD")
    parser.add_argument("--dest", default="data/funsd")
    parser.add_argument("--url", default=_URL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    try:
        root = resolve_funsd_root(dest)
    except FileNotFoundError:
        root = None
    if root is not None:
        train_count = count_pages(root, "train")
        test_count = count_pages(root, "test")
        print(f"FUNSD already present at {root} ({train_count} train, {test_count} test)")
        return

    zip_path = dest / "dataset.zip"
    print(f"Downloading {_URL if args.url == _URL else args.url}")
    with requests.get(args.url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with zip_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    handle.write(chunk)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(dest)
    root = resolve_funsd_root(dest)
    train_count = count_pages(root, "train")
    test_count = count_pages(root, "test")
    if (train_count, test_count) != (149, 50):
        raise RuntimeError(f"Expected 149 train and 50 test forms, found {train_count} and {test_count}")
    print(f"FUNSD ready at {root} ({train_count} train, {test_count} test)")


if __name__ == "__main__":
    main()
