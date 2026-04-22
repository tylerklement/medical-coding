"""
src/data_loader.py

Loads CodiEsp annotations and clinical texts, then converts character-level
reference positions into token-level BIO labels for NER fine-tuning.

CodiEsp zip layout (after download_data.py):
    data/
    ├── train/
    │   ├── text_files/            ← Spanish .txt files
    │   ├── text_files_en/         ← English .txt files
    │   ├── codiesp_D_train.tsv    ← diagnosis codes only
    │   ├── codiesp_P_train.tsv    ← procedure codes only
    │   └── codiesp_X_train.tsv   ← codes + spans  ← used here
    ├── dev/
    │   ├── text_files/
    │   └── codiesp_X_dev.tsv
    └── test/
        └── text_files/

Key classes / functions:
    CodiEspAnnotation   — dataclass for one annotation row
    load_annotations()  — parse a CodiEsp-X TSV annotation file
    load_text()         — read a clinical .txt file from text_files/
    resolve_tsv()       — auto-locate the correct TSV for a split
    build_bio_examples()— produce HuggingFace-ready tokenized examples
                          with token-level labels
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from transformers import PreTrainedTokenizerFast

# ── Label vocabulary ───────────────────────────────────────────────────────────
LABEL2ID: Dict[str, int] = {
    "O": 0,
    "B-DIAG": 1,
    "I-DIAG": 2,
    "B-PROC": 3,
    "I-PROC": 4,
}
ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}

# Mapping from CodiEsp label strings → BIO prefix
_CODIESP_LABEL_MAP = {
    "DIAGNOSTICO": "DIAG",
    "PROCEDIMIENTO": "PROC",
}


@dataclass
class CodiEspAnnotation:
    """One row from a CodiEsp annotation TSV file."""

    article_id: str
    label: str          # "DIAGNOSTICO" | "PROCEDIMIENTO"
    icd10_code: str
    text_reference: str
    # reference_position may be a single int or a range "start end"
    ref_start: int
    ref_end: int

    @property
    def bio_type(self) -> str:
        return _CODIESP_LABEL_MAP.get(self.label.upper(), "DIAG")


def resolve_tsv(data_dir: str | Path, split: str, task: str = "X") -> Path:
    """
    Auto-locate the CodiEsp annotation TSV for a given split and sub-task.

    Tries the following naming conventions in order (the v4 archive uses the
    first pattern, older releases use the second):

        {split}{task}.tsv          e.g.  trainX.tsv  devX.tsv  testD.tsv
        codiesp_{task}_{split}.tsv e.g.  codiesp_X_train.tsv  (older)
        {split}_annotations.tsv   e.g.  train_annotations.tsv (legacy)

    Args:
        data_dir: Root data directory (e.g., "data/").
        split:    "train" | "dev" | "test".
        task:     "X" (codes+spans, default) | "D" | "P".

    Returns:
        Path to the TSV file.

    Raises:
        FileNotFoundError if not found.
    """
    candidates = [
        # v4 naming: trainX.tsv, devX.tsv, testP.tsv …
        Path(data_dir) / split / f"{split}{task}.tsv",
        Path(data_dir) / split / f"{split}{task.lower()}.tsv",
        # older naming: codiesp_X_train.tsv
        Path(data_dir) / split / f"codiesp_{task}_{split}.tsv",
        Path(data_dir) / split / f"codiesp_{task.lower()}_{split}.tsv",
        # legacy fallback
        Path(data_dir) / split / f"{split}_annotations.tsv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find CodiEsp-{task} TSV for split '{split}' in {data_dir}. "
        f"Tried: {[str(c) for c in candidates]}"
    )


def load_annotations(tsv_path: str | Path) -> List[CodiEspAnnotation]:
    """
    Parse a CodiEsp-X annotation TSV.

    Expected columns (tab-separated, no header):
        articleID  label  ICD10-code  text-reference  reference-position

    Handles both CodiEsp-X format (5 cols) and CodiEsp-D/P format (2 cols).

    reference-position can be:
        - "123" (single character offset → we treat as point span)
        - "123 456" (start end offsets, space-separated)
    """
    tsv_path = Path(tsv_path)
    # Detect number of columns to handle both CodiEsp-X (5) and D/P (2)
    with open(tsv_path, encoding="utf-8") as f:
        first_line = f.readline()
    n_cols = len(first_line.split("\t"))

    if n_cols >= 5:
        col_names = ["article_id", "label", "icd10_code", "text_reference", "ref_pos"]
    else:
        # CodiEsp-D or CodiEsp-P (no span info)
        col_names = ["article_id", "icd10_code"]

    df = pd.read_csv(
        tsv_path,
        sep="\t",
        header=None,
        names=col_names,
        dtype=str,
        keep_default_na=False,
        usecols=range(len(col_names)),
    )

    # For D/P format: add dummy label and empty span info
    if "label" not in df.columns:
        df["label"] = "DIAGNOSTICO"
        df["text_reference"] = ""
        df["ref_pos"] = ""

    annotations: List[CodiEspAnnotation] = []
    for _, row in df.iterrows():
        ref_pos = str(row["ref_pos"]).strip()
        parts = ref_pos.split()
        try:
            if len(parts) == 2:
                start, end = int(parts[0]), int(parts[1])
            elif len(parts) == 1:
                start = int(parts[0])
                text_ref = str(row.get("text_reference", ""))
                end = start + len(text_ref) if text_ref else start
            else:
                # No span info (D/P format) — set invalid to skip in BIO conversion
                start, end = -1, -1
        except ValueError:
            start, end = -1, -1

        annotations.append(
            CodiEspAnnotation(
                article_id=row["article_id"].strip(),
                label=row["label"].strip(),
                icd10_code=row["icd10_code"].strip(),
                text_reference=str(row.get("text_reference", "")).strip(),
                ref_start=start,
                ref_end=end,
            )
        )
    return annotations


def load_text(text_dir: str | Path, article_id: str) -> str:
    """
    Read the raw clinical text for a given article ID.

    Searches directly in text_dir (if it points at text_files/) or
    falls back to text_dir/text_files/ automatically.
    """
    text_dir = Path(text_dir)
    # Try direct path first
    direct = text_dir / f"{article_id}.txt"
    if direct.exists():
        return direct.read_text(encoding="utf-8")
    # Try text_files/ sub-folder (in case caller passed the split dir)
    subdir = text_dir / "text_files" / f"{article_id}.txt"
    if subdir.exists():
        return subdir.read_text(encoding="utf-8")
    raise FileNotFoundError(
        f"Clinical text not found for article '{article_id}'. "
        f"Tried: {direct} and {subdir}"
    )


def _char_spans_to_token_labels(
    text: str,
    char_spans: List[Tuple[int, int, str]],  # (start, end, bio_type)
    encoding,  # tokenizer output with offset_mapping
) -> List[int]:
    """
    Convert character-level spans to token-level BIO label IDs.

    For each token, we check whether its character span overlaps with any
    annotated span and assign the appropriate B-/I- label.
    """
    num_tokens = len(encoding["input_ids"])
    labels = [LABEL2ID["O"]] * num_tokens

    for token_idx, (tok_start, tok_end) in enumerate(encoding["offset_mapping"]):
        if tok_start == tok_end:
            # Special token (CLS, SEP, PAD) → keep O but mark as -100 later
            continue
        for ann_start, ann_end, bio_type in char_spans:
            if tok_start >= ann_start and tok_end <= ann_end:
                # Token is inside this span
                if tok_start == ann_start:
                    labels[token_idx] = LABEL2ID[f"B-{bio_type}"]
                else:
                    labels[token_idx] = LABEL2ID[f"I-{bio_type}"]
                break  # first matching span wins

    return labels


def build_bio_examples(
    text_dir: str | Path,
    annotations: List[CodiEspAnnotation],
    tokenizer: PreTrainedTokenizerFast,
    max_length: int = 512,
    stride: int = 128,
) -> List[Dict]:
    """
    Produce a list of tokenized examples ready for HuggingFace Dataset creation.

    Uses a sliding window to handle documents longer than max_length tokens.

    Returns a list of dicts with keys:
        input_ids, attention_mask, labels
        (and optionally: token_type_ids, article_id, window_idx)
    """
    # Group annotations by article_id
    ann_by_article: Dict[str, List[CodiEspAnnotation]] = {}
    for ann in annotations:
        ann_by_article.setdefault(ann.article_id, []).append(ann)

    examples = []
    text_dir = Path(text_dir)

    for article_id, anns in ann_by_article.items():
        # Resolve text file: support both text_files/ subfolder and direct
        txt_path = text_dir / f"{article_id}.txt"
        txt_path_sub = text_dir / "text_files" / f"{article_id}.txt"
        if txt_path_sub.exists():
            txt_path = txt_path_sub
        elif not txt_path.exists():
            continue

        text = txt_path.read_text(encoding="utf-8")

        # Build char-span list: (start, end, bio_type)
        char_spans = [
            (a.ref_start, a.ref_end, a.bio_type)
            for a in anns
            if a.ref_start >= 0 and a.ref_end > a.ref_start
        ]

        # Tokenize with offset mapping + sliding window
        encoding = tokenizer(
            text,
            max_length=max_length,
            stride=stride,
            truncation=True,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        for window_idx in range(len(encoding["input_ids"])):
            window_enc = {k: encoding[k][window_idx] for k in encoding}
            labels_row = _char_spans_to_token_labels(
                text,
                char_spans,
                {
                    "input_ids": window_enc["input_ids"],
                    "offset_mapping": window_enc["offset_mapping"],
                },
            )
            # Mask special tokens with -100 (ignored by CrossEntropyLoss)
            for tok_idx, (tok_s, tok_e) in enumerate(window_enc["offset_mapping"]):
                if tok_s == tok_e:
                    labels_row[tok_idx] = -100

            examples.append(
                {
                    "input_ids": window_enc["input_ids"],
                    "attention_mask": window_enc["attention_mask"],
                    "labels": labels_row,
                    "article_id": article_id,
                    "window_idx": window_idx,
                }
            )

    return examples


# ── Standalone annotation loader (code-only, no NER labels) ───────────────────

def load_code_labels(
    tsv_path: str | Path,
    label_filter: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load CodiEsp annotations as a DataFrame for code-mapping experiments.

    Args:
        tsv_path: Path to annotation TSV.
        label_filter: "DIAGNOSTICO" | "PROCEDIMIENTO" | None (both).

    Returns DataFrame with columns:
        article_id, label, icd10_code, text_reference, ref_start, ref_end
    """
    anns = load_annotations(tsv_path)
    rows = [
        {
            "article_id": a.article_id,
            "label": a.label,
            "icd10_code": a.icd10_code,
            "text_reference": a.text_reference,
            "ref_start": a.ref_start,
            "ref_end": a.ref_end,
        }
        for a in anns
    ]
    df = pd.DataFrame(rows)
    if label_filter:
        df = df[df["label"].str.upper() == label_filter.upper()].reset_index(drop=True)
    return df


if __name__ == "__main__":
    # Quick smoke-test: print first 5 annotations
    import sys

    if len(sys.argv) > 1:
        tsv = sys.argv[1]
    else:
        # Auto-resolve from data/
        tsv = str(resolve_tsv("data", "train", task="X"))
    anns = load_annotations(tsv)
    print(f"Loaded {len(anns)} annotations from {tsv}")
    for a in anns[:5]:
        print(f"  [{a.label}] {a.article_id}: {a.icd10_code!r} "
              f"@ {a.ref_start}-{a.ref_end} → {a.text_reference!r}")
