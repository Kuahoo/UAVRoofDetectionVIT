"""Download and extract the public RoofD archive from Yandex Disk."""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

PUBLIC_URL = "https://disk.yandex.ru/d/gJeFS6WIRlr0fg"
API_URL = "https://cloud-api.yandex.net/v1/disk/public/resources/download"
USER_AGENT = "roofd-prototype/1.0"
EXPECTED_SIZE = 6_150_929_420


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default="data/raw/RoofD")
    parser.add_argument("--archive", default="data/raw/roof_defects_dataset.zip")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-archive", action="store_true")
    return parser.parse_args()


def download_url(public_url: str, attempts: int = 6) -> str:
    query = urllib.parse.urlencode({"public_key": public_url})
    request = urllib.request.Request(f"{API_URL}?{query}", headers={"User-Agent": USER_AGENT})
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.load(response)
            if "href" in payload:
                return payload["href"]
            raise RuntimeError(payload)
        except Exception as exc:
            if attempt == attempts:
                raise
            print(f"Metadata request failed ({exc}); retrying in {10 * attempt}s")
            time.sleep(10 * attempt)
    raise RuntimeError("unreachable")


def download(href: str, destination: Path) -> None:
    request = urllib.request.Request(href, headers={"User-Agent": USER_AGENT})
    downloaded = 0
    with urllib.request.urlopen(request, timeout=120) as source, destination.open("wb") as target:
        while True:
            chunk = source.read(1 << 20)
            if not chunk:
                break
            target.write(chunk)
            downloaded += len(chunk)
            if downloaded // (512 << 20) != (downloaded - len(chunk)) // (512 << 20):
                print(f"  {downloaded / 1e9:.2f} GB downloaded")


def main() -> None:
    args = parse_args()
    archive = Path(args.archive)
    destination = Path(args.dest)
    archive.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=True, exist_ok=True)

    if args.force or not archive.exists() or archive.stat().st_size < EXPECTED_SIZE:
        print("Resolving Yandex Disk download link...")
        href = download_url(PUBLIC_URL)
        print(f"Downloading to {archive} (about 6.2 GB)...")
        download(href, archive)
    else:
        print(f"Using existing archive {archive}")

    print(f"Extracting to {destination}...")
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(destination)
        names = bundle.namelist()
        print("Archive top-level entries:", sorted({name.split("/")[0] for name in names})[:10])

    if not args.keep_archive:
        archive.unlink()
        print(f"Removed {archive} (use --keep-archive to keep it)")
    print("Done. Run: python -m scripts.prepare_data")


if __name__ == "__main__":
    main()
