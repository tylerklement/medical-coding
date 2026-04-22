#!/usr/bin/env python3
"""
download_data.py — Download the CodiEsp dataset from Zenodo and organise it.

Usage:
    python scripts/download_data.py

The CodiEsp dataset is hosted at:
    https://zenodo.org/records/3837305

After running, the data/ directory will be structured as:
    data/
    ├── train/
    │   ├── text_files/          ← Spanish .txt files (one per clinical case)
    │   ├── text_files_en/       ← English machine-translated .txt files
    │   ├── codiesp_D_train.tsv  ← Diagnosis codes (CodiEsp-D)
    │   ├── codiesp_P_train.tsv  ← Procedure codes (CodiEsp-P)
    │   └── codiesp_X_train.tsv  ← Codes + text spans (CodiEsp-X) ← main file
    ├── dev/
    │   ├── text_files/
    │   ├── codiesp_D_dev.tsv
    │   ├── codiesp_P_dev.tsv
    │   └── codiesp_X_dev.tsv
    └── test/
        └── text_files/
"""

import shutil
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# ── Zenodo record ──────────────────────────────────────────────────────────────
# Direct download link confirmed from https://zenodo.org/records/3837305
ZENODO_URL = "https://zenodo.org/records/3837305/files/codiesp.zip?download=1"
DATA_DIR = Path(__file__).parent.parent / "data"
TMP_ZIP = DATA_DIR / "codiesp.zip"


def download_file(url: str, dest: Path) -> None:
    """Stream-download a file with a progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    print(f"  → {dest}")
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
            bar.update(len(chunk))
    print(f"  Downloaded {dest.stat().st_size / 1e6:.1f} MB")


def unzip_and_organise(zip_path: Path, data_dir: Path) -> None:
    """
    Extract the CodiEsp zip into data_dir.

    The archive unpacks as:
        codiesp/
            train/   dev/   test/   background/

    We move the contents of codiesp/ directly into data_dir so the result is:
        data/train/   data/dev/   data/test/   data/background/
    """
    extract_tmp = data_dir / "_extract_tmp"
    extract_tmp.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {zip_path.name} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        print(f"  Archive contains {len(members)} files")
        zf.extractall(extract_tmp)

    # Find the top-level extracted folder (usually 'codiesp/')
    top_dirs = [p for p in extract_tmp.iterdir() if p.is_dir()]
    if len(top_dirs) == 1:
        source_root = top_dirs[0]
        print(f"  Extracted root: {source_root.name}/")
    else:
        # Files extracted directly (no wrapper folder)
        source_root = extract_tmp

    # Move each split folder into data/
    for split in ["train", "dev", "test", "background"]:
        src = source_root / split
        dst = data_dir / split
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(src), str(dst))
            print(f"  Moved {split}/ → data/{split}/")
        else:
            print(f"  Warning: {split}/ not found in archive")

    # Clean up temp dir
    shutil.rmtree(extract_tmp, ignore_errors=True)


def print_structure(data_dir: Path) -> None:
    """Print a summary of the extracted data layout."""
    print("\n── Data directory layout ──────────────────────")
    for split in ["train", "dev", "test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        tsv_files = sorted(split_dir.glob("*.tsv"))
        txt_count = len(list((split_dir / "text_files").glob("*.txt"))) \
            if (split_dir / "text_files").exists() else 0
        print(f"  data/{split}/")
        print(f"    text_files/  ({txt_count} .txt files)")
        for tsv in tsv_files:
            print(f"    {tsv.name}")
    print("───────────────────────────────────────────────")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if TMP_ZIP.exists():
        print(f"Archive already present at {TMP_ZIP} ({TMP_ZIP.stat().st_size / 1e6:.0f} MB), "
              "skipping download.")
    else:
        download_file(ZENODO_URL, TMP_ZIP)

    unzip_and_organise(TMP_ZIP, DATA_DIR)
    print_structure(DATA_DIR)
    print("\n✅ Done!")
    print("   Next: python scripts/build_icd10_index.py")
    print("   Then: jupyter notebook notebooks/01_eda.ipynb")


if __name__ == "__main__":
    main()
