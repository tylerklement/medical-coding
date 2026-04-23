#!/usr/bin/env python3
"""
train_reranker.py — Fine-tune a cross-encoder for ICD-10 code reranking.

The cross-encoder learns to score (span_text, icd10_description) pairs.
Training data is constructed from CodiEsp annotations:
  - Positive pairs: (span_text, gold_code_description) -> label 1
  - Negative pairs: (span_text, hard_negative_description) -> label 0
    Hard negatives come from the bi-encoder's top-K candidates that are NOT the gold code.

Usage:
    python train_reranker.py --no_wandb          # all paths auto-detected
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from torch.utils.data import DataLoader

from src.data_loader import load_annotations, load_text, resolve_tsv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train_tsv", default=None,
                   help="CodiEsp-X train TSV; auto-detected if omitted")
    p.add_argument("--train_text", default="data/train/")
    p.add_argument("--dev_tsv", default=None,
                   help="CodiEsp-X dev TSV; auto-detected if omitted")
    p.add_argument("--dev_text", default="data/dev/")
    p.add_argument("--icd10cm_json",  default="data/icd10cm_descriptions.json")
    p.add_argument("--icd10pcs_json", default="data/icd10pcs_descriptions.json")
    p.add_argument("--bi_encoder",    default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    p.add_argument("--cross_encoder", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    p.add_argument("--output_dir",    default="models/reranker/")
    p.add_argument("--epochs",           type=int,   default=3)
    p.add_argument("--batch_size",       type=int,   default=16)
    p.add_argument("--negatives_per_pos",type=int,   default=4)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--no_wandb",   action="store_true")
    return p.parse_args()


def load_icd10_descs(json_path: str) -> Dict[str, str]:
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def build_training_pairs(
    tsv_path: str,
    text_dir: str,
    cm_descs: Dict[str, str],
    pcs_descs: Dict[str, str],
    bi_encoder: SentenceTransformer,
    negatives_per_pos: int = 4,
    top_k_negatives: int = 30,
) -> List[InputExample]:
    """
    Build (span, code_description, label) training pairs.

    Strategy:
    1. For each annotation, use the gold code description as the positive.
    2. Mine hard negatives using the bi-encoder's top-K retrievals
       (excluding the gold code).
    """
    anns = load_annotations(tsv_path)
    all_examples: List[InputExample] = []

    for ann in anns:
        descs = cm_descs if ann.bio_type == "DIAG" else pcs_descs
        gold_desc = descs.get(ann.icd10_code)
        if not gold_desc:
            continue

        span_text = ann.text_reference

        # Positive example
        all_examples.append(InputExample(texts=[span_text, gold_desc], label=1.0))

        # Hard-negative mining via bi-encoder
        all_desc_items = list(descs.items())
        all_descs_text = [d for _, d in all_desc_items]
        all_codes = [c for c, _ in all_desc_items]

        span_emb = bi_encoder.encode([span_text], normalize_embeddings=True)
        desc_embs = bi_encoder.encode(all_descs_text, normalize_embeddings=True, batch_size=256)
        sims = (span_emb @ desc_embs.T)[0]
        top_k_idx = np.argsort(-sims)[:top_k_negatives]

        negatives = [
            all_descs_text[i]
            for i in top_k_idx
            if all_codes[i] != ann.icd10_code
        ][:negatives_per_pos]

        for neg_desc in negatives:
            all_examples.append(InputExample(texts=[span_text, neg_desc], label=0.0))

    random.shuffle(all_examples)
    return all_examples


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Auto-resolve TSV paths
    train_tsv = args.train_tsv or str(resolve_tsv("data", "train", task="X"))
    dev_tsv   = args.dev_tsv   or str(resolve_tsv("data", "dev",   task="X"))
    print(f"Train TSV : {train_tsv}")
    print(f"Dev   TSV : {dev_tsv}")

    print("Loading ICD-10 descriptions ...")
    cm_descs  = load_icd10_descs(args.icd10cm_json)
    pcs_descs = load_icd10_descs(args.icd10pcs_json)

    print("Loading bi-encoder for hard-negative mining …")
    bi_encoder = SentenceTransformer(args.bi_encoder)

    print("Building training pairs ...")
    train_examples = build_training_pairs(
        train_tsv, args.train_text,
        cm_descs, pcs_descs, bi_encoder,
        negatives_per_pos=args.negatives_per_pos,
    )
    print(f"  {len(train_examples)} training pairs")

    print("Building dev pairs ...")
    dev_examples = build_training_pairs(
        dev_tsv, args.dev_text,
        cm_descs, pcs_descs, bi_encoder,
        negatives_per_pos=args.negatives_per_pos,
    )
    print(f"  {len(dev_examples)} dev pairs")

    print(f"Loading cross-encoder: {args.cross_encoder}")
    model = CrossEncoder(args.cross_encoder, num_labels=1)

    train_dl = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(dev_examples)

    print("Training cross-encoder …")
    model.fit(
        train_dataloader=train_dl,
        evaluator=evaluator,
        epochs=args.epochs,
        output_path=args.output_dir,
        evaluation_steps=500,
        save_best_model=True,
    )

    print(f"Saved cross-encoder to {args.output_dir}")


if __name__ == "__main__":
    main()
