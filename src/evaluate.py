"""
src/evaluate.py

Evaluation utilities for the CodiEsp medical coding pipeline.

Metrics:
  - Stage 1 (NER): seqeval entity-level precision, recall, F1
  - Stage 2 (Coding): exact-match accuracy, top-k accuracy, MAP@K
  - End-to-end: CodiEsp official-style MAP evaluation
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# seqeval for NER metrics
try:
    from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
    SEQEVAL_AVAILABLE = True
except ImportError:
    SEQEVAL_AVAILABLE = False


# ── NER Evaluation ─────────────────────────────────────────────────────────────

def evaluate_ner(
    true_labels: List[List[str]],   # list of token-label-sequence lists
    pred_labels: List[List[str]],
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Compute seqeval entity-level NER metrics.

    Args:
        true_labels: List of BIO label sequences (strings, e.g., ["O","B-DIAG","I-DIAG"])
        pred_labels: Corresponding predictions.

    Returns dict with keys: precision, recall, f1.
    """
    if not SEQEVAL_AVAILABLE:
        raise ImportError("seqeval is not installed: pip install seqeval")

    p = precision_score(true_labels, pred_labels)
    r = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    if verbose:
        print(classification_report(true_labels, pred_labels))
    return {"precision": p, "recall": r, "f1": f1}


# ── Code Mapping Evaluation ────────────────────────────────────────────────────

def top_k_accuracy(
    true_codes: List[str],
    pred_codes_ranked: List[List[str]],  # each inner list is ranked top-K predictions
    k: int = 5,
) -> float:
    """Fraction of samples where the true code appears in the top-K predictions."""
    hits = sum(
        1 for true, preds in zip(true_codes, pred_codes_ranked)
        if true in preds[:k]
    )
    return hits / len(true_codes) if true_codes else 0.0


def mean_average_precision(
    true_codes: List[str],
    pred_codes_ranked: List[List[str]],
) -> float:
    """
    Compute Mean Average Precision (MAP) in the CodiEsp style.

    For each sample, compute the average precision at each rank where a
    correct code appears, then average over all samples.
    """
    aps = []
    for true, preds in zip(true_codes, pred_codes_ranked):
        if not preds:
            aps.append(0.0)
            continue
        hits = 0
        precisions = []
        for rank, pred in enumerate(preds, start=1):
            if pred == true:
                hits += 1
                precisions.append(hits / rank)
        aps.append(np.mean(precisions) if precisions else 0.0)
    return float(np.mean(aps)) if aps else 0.0


# ── End-to-End CodiEsp Evaluation ─────────────────────────────────────────────

def _load_gold(tsv_path: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Load gold-standard CodiEsp annotations.

    Returns:
        {article_id: {"DIAG": [code, ...], "PROC": [code, ...]}}
    """
    df = pd.read_csv(
        tsv_path,
        sep="\t",
        header=None,
        names=["article_id", "label", "icd10_code", "text_ref", "ref_pos"],
        dtype=str,
        keep_default_na=False,
    )
    gold: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: {"DIAG": [], "PROC": []})
    for _, row in df.iterrows():
        label = "DIAG" if "DIAG" in row["label"].upper() else "PROC"
        gold[row["article_id"].strip()][label].append(row["icd10_code"].strip())
    return dict(gold)


def evaluate_end_to_end(
    predictions: Dict[str, List[Dict]],  # {article_id: pipeline.process_text() output}
    gold_tsv: str,
    label_filter: Optional[str] = None,  # "DIAG" | "PROC" | None
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Full end-to-end CodiEsp-style evaluation.

    Args:
        predictions: {article_id: list of result dicts from MedicalCodingPipeline}
        gold_tsv: Path to gold annotation TSV.
        label_filter: Restrict evaluation to DIAG or PROC.

    Returns dict with keys: precision, recall, f1, map.
    """
    gold = _load_gold(gold_tsv)

    all_tp = all_fp = all_fn = 0
    article_aps: List[float] = []

    for article_id, gold_codes_by_type in gold.items():
        pred_results = predictions.get(article_id, [])

        for label in (["DIAG", "PROC"] if not label_filter else [label_filter]):
            gold_set = set(gold_codes_by_type.get(label, []))
            pred_ranked = [
                r["icd10_code"]
                for r in sorted(pred_results, key=lambda x: -x["code_score"])
                if r["entity_type"] == label
            ]
            pred_set = set(pred_ranked)

            tp = len(gold_set & pred_set)
            fp = len(pred_set - gold_set)
            fn = len(gold_set - pred_set)
            all_tp += tp
            all_fp += fp
            all_fn += fn

            # Per-article AP
            hits = 0
            precisions = []
            for rank, code in enumerate(pred_ranked, 1):
                if code in gold_set:
                    hits += 1
                    precisions.append(hits / rank)
            article_aps.append(np.mean(precisions) if precisions else 0.0)

    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    map_score = float(np.mean(article_aps)) if article_aps else 0.0

    if verbose:
        print(f"{'Metric':<15} {'Value':>8}")
        print("-" * 25)
        for name, val in [("Precision", precision), ("Recall", recall),
                           ("F1", f1), ("MAP", map_score)]:
            print(f"{name:<15} {val:>8.4f}")

    return {"precision": precision, "recall": recall, "f1": f1, "map": map_score}
