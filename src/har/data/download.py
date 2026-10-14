"""Download and extract the official UCI HAR dataset without ever overwriting it.

The UCI archive is a zip that itself contains ``UCI HAR Dataset.zip``; both
levels are handled. A small JSON manifest (source URL, SHA-256, timestamp) is
written next to the data so the exact download can be recorded in SQL later.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests

from har.config import DATASET_URL, PATHS

ARCHIVE_NAME = "uci_har_archive.zip"
MANIFEST_NAME = "download_manifest.json"
INNER_ZIP_NAME = "UCI HAR Dataset.zip"


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 hex digest of a file, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def download_archive(url: str = DATASET_URL, raw_dir: Path = PATHS.raw_dir) -> Path:
    """Download the archive to ``raw_dir``; skip if it is already there."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / ARCHIVE_NAME
    if target.exists():
        print(f"[skip] archive already present: {target}")
        return target
    tmp = target.with_suffix(".part")
    print(f"[download] {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    tmp.rename(target)  # only a complete download gets the final name
    return target


def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    """Extract members, rejecting any path that would escape ``dest``."""
    dest = dest.resolve()
    for member in zf.namelist():
        out = (dest / member).resolve()
        if dest != out and dest not in out.parents:
            raise ValueError(f"Unsafe path in archive: {member}")
    zf.extractall(dest)


def extract_dataset(archive: Path, raw_dir: Path = PATHS.raw_dir) -> Path:
    """Extract the dataset folder; skip if it already exists (never overwrite)."""
    dataset_dir = raw_dir / "UCI HAR Dataset"
    if dataset_dir.exists():
        print(f"[skip] dataset folder already present: {dataset_dir}")
        return dataset_dir
    with zipfile.ZipFile(archive) as outer:
        names = outer.namelist()
        if INNER_ZIP_NAME in names:  # official archive: zip inside a zip
            with zipfile.ZipFile(io.BytesIO(outer.read(INNER_ZIP_NAME))) as inner:
                _safe_extract(inner, raw_dir)
        else:
            _safe_extract(outer, raw_dir)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Expected folder not found after extraction: {dataset_dir}")
    return dataset_dir


def write_manifest(archive: Path, url: str, raw_dir: Path = PATHS.raw_dir) -> dict:
    """Write the download manifest once; later runs read the existing one."""
    path = raw_dir / MANIFEST_NAME
    if path.exists():
        return json.loads(path.read_text())
    manifest = {
        "source_url": url,
        "archive": archive.name,
        "sha256": sha256_of(archive),
        "size_bytes": archive.stat().st_size,
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(manifest, indent=2))
    return manifest


def fetch_dataset(url: str = DATASET_URL, raw_dir: Path = PATHS.raw_dir) -> tuple[Path, dict]:
    """Download (if needed), extract (if needed) and return (dataset_dir, manifest)."""
    archive = download_archive(url, raw_dir)
    manifest = write_manifest(archive, url, raw_dir)
    if sha256_of(archive) != manifest["sha256"]:
        raise RuntimeError("Archive checksum does not match the recorded manifest.")
    return extract_dataset(archive, raw_dir), manifest
