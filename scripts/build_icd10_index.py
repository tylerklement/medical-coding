#!/usr/bin/env python3
"""
scripts/build_icd10_index.py

Download the official ICD-10-CM and ICD-10-PCS description files from CMS
and save them as JSON: {code: description_string}

Sources (confirmed live from cms.gov/medicare/coding-billing/icd-10-codes):
  - ICD-10-CM FY2026: 2026-code-descriptions-tabular-order.zip
  - ICD-10-PCS FY2026: 2026-icd-10-pcs-codes-file.zip

Usage:
    python scripts/build_icd10_index.py
"""

import json
import re
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import requests
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / "data"

# ── CMS FY2026 ICD-10 Downloads ───────────────────────────────────────────────
# Confirmed from https://www.cms.gov/medicare/coding-billing/icd-10-codes
# ICD-10-CM: code descriptions in tabular order (fixed-width .txt inside zip)
ICD10_CM_URL = "https://www.cms.gov/files/zip/2026-code-descriptions-tabular-order.zip"
# ICD-10-PCS: codes file (7-char code + space + long description)
ICD10_PCS_URL = "https://www.cms.gov/files/zip/2026-icd-10-pcs-codes-file.zip"


def _download_bytes(url: str) -> bytes:
    print(f"Downloading {url} …")
    resp = requests.get(url, timeout=120, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    buf = BytesIO()
    with tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in resp.iter_content(chunk_size=65536):
            buf.write(chunk)
            bar.update(len(chunk))
    buf.seek(0)
    return buf.read()


def parse_cm_descriptions(raw_bytes: bytes) -> dict:
    """
    Parse ICD-10-CM tabular-order file into {code: description}.

    The CMS zip contains icd10cm_order_2026.txt with space-separated fields:
        col 1-5  : zero-padded row number  (e.g. '00002')
        col 7-13 : ICD-10-CM code, left-justified with trailing spaces
        col 15   : header/valid flag  '0' = header category, '1' = billable
        col 17+  : short description (padded to ~60 chars) then long description

    Example line:
        '00002 A000    1 Cholera due to ...           Cholera due to ...'

    We split each line on whitespace so:
        parts[0] = row_num
        parts[1] = code
        parts[2] = flag ('0' or '1')
        parts[3:] = description words

    The description field starting at character 16 (0-indexed) contains
    both a short and long version separated by many spaces; we take
    everything from position 16 onward and strip it.
    """
    descs = {}
    try:
        with zipfile.ZipFile(BytesIO(raw_bytes)) as zf:
            candidates = [
                n for n in zf.namelist()
                if n.lower().endswith(".txt") and "order" in n.lower()
            ]
            if not candidates:
                candidates = [n for n in zf.namelist() if n.lower().endswith(".txt")]
            if not candidates:
                print(f"  No .txt found in CM zip. Contents: {zf.namelist()[:10]}")
                return {}
            fname = candidates[0]
            print(f"  Parsing CM file: {fname}")
            text = zf.read(fname).decode("latin-1", errors="replace")
    except Exception as e:
        print(f"  Failed to open CM zip: {e}")
        return {}

    for line in text.splitlines():
        # Minimum: '00001 A00     0 ...' = at least 18 chars
        if len(line) < 18:
            continue
        parts = line.split()
        # parts[0]=row, parts[1]=code, parts[2]=flag, parts[3:]=description
        if len(parts) < 4:
            continue
        flag = parts[2]
        if flag != "1":
            continue   # '0' = header/non-billable category
        code = parts[1]
        # The long description starts after the flag digit at col ~16
        # We grab everything from position 16 and collapse internal whitespace
        raw_desc = line[16:]
        # The field contains short desc + padding spaces + long desc
        # Split on multiple spaces to get the long description
        desc_parts = re.split(r" {2,}", raw_desc.strip())
        desc = desc_parts[-1].strip() if desc_parts else raw_desc.strip()
        if code and desc:
            descs[code] = desc
    return descs


def parse_pcs_descriptions(raw_bytes: bytes) -> dict:
    """
    Parse ICD-10-PCS codes file into {code: description}.

    The CMS zip contains a file like icd10pcs_codes_2026.txt.
    Each line: 7-char PCS code + space + long title.
    """
    descs = {}
    try:
        with zipfile.ZipFile(BytesIO(raw_bytes)) as zf:
            all_txt = [n for n in zf.namelist() if n.lower().endswith(".txt")]
            # Exclude addenda / changelog files — they list diffs, not full codes
            main_candidates = [
                n for n in all_txt
                if "addenda" not in n.lower() and "addendum" not in n.lower()
            ]
            candidates = main_candidates if main_candidates else all_txt
            if not candidates:
                print(f"  No .txt found in PCS zip. Contents: {zf.namelist()[:10]}")
                return {}
            # Pick the largest file — the full codes file is always the biggest
            fname = max(candidates, key=lambda n: zf.getinfo(n).file_size)
            print(f"  Parsing PCS file: {fname} ({zf.getinfo(fname).file_size:,} bytes)")
            text = zf.read(fname).decode("latin-1", errors="replace")
    except Exception as e:
        print(f"  Failed to open PCS zip: {e}")
        return {}

    for line in text.splitlines():
        line = line.strip()
        if len(line) < 9:   # 7-char code + space + ≥1 char
            continue
        code = line[:7]
        # PCS codes are exactly 7 uppercase alphanumeric chars
        if re.match(r"^[A-Z0-9]{7}$", code) and line[7] == " ":
            desc = line[8:].strip()
            if desc:
                descs[code] = desc
    return descs


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cm_out  = DATA_DIR / "icd10cm_descriptions.json"
    pcs_out = DATA_DIR / "icd10pcs_descriptions.json"

    # ── ICD-10-CM ─────────────────────────────────────────────────────────────
    # Skip only if the file already has genuine content (> 10 KB)
    if cm_out.exists() and cm_out.stat().st_size > 10_000:
        print(f"ICD-10-CM already exists ({cm_out.stat().st_size / 1e3:.0f} KB), skipping.")
    else:
        raw = _download_bytes(ICD10_CM_URL)
        cm_descs = parse_cm_descriptions(raw)
        if not cm_descs:
            print("ERROR: CM parser returned 0 codes. The URL or file format may have changed.")
            sys.exit(1)
        print(f"Parsed {len(cm_descs):,} ICD-10-CM codes")
        with open(cm_out, "w", encoding="utf-8") as f:
            json.dump(cm_descs, f, ensure_ascii=False, indent=2)
        print(f"Saved → {cm_out}")

    # ── ICD-10-PCS ────────────────────────────────────────────────────────────
    if pcs_out.exists() and pcs_out.stat().st_size > 10_000:
        print(f"ICD-10-PCS already exists ({pcs_out.stat().st_size / 1e3:.0f} KB), skipping.")
    else:
        raw = _download_bytes(ICD10_PCS_URL)
        pcs_descs = parse_pcs_descriptions(raw)
        if not pcs_descs:
            print("ERROR: PCS parser returned 0 codes. The URL or file format may have changed.")
            sys.exit(1)
        print(f"Parsed {len(pcs_descs):,} ICD-10-PCS codes")
        with open(pcs_out, "w", encoding="utf-8") as f:
            json.dump(pcs_descs, f, ensure_ascii=False, indent=2)
        print(f"Saved → {pcs_out}")

    print("\n✅  ICD-10 index build complete.")
    print(f"   CM  codes : {len(json.load(open(cm_out))):,}")
    print(f"   PCS codes : {len(json.load(open(pcs_out))):,}")


if __name__ == "__main__":
    main()
