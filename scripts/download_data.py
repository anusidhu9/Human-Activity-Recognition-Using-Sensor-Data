"""Stage 2a: download and extract the UCI HAR dataset into data/raw/ (idempotent)."""

from har.data.download import fetch_dataset


def main() -> None:
    dataset_dir, manifest = fetch_dataset()
    print(f"dataset folder : {dataset_dir}")
    print(f"archive sha256 : {manifest['sha256']}")
    print(f"archive size   : {manifest['size_bytes'] / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
