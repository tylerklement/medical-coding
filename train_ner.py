#!/usr/bin/env python3
"""
train_ner.py — Fine-tune the Spanish Biomedical BERT NER model on CodiEsp.

EDA-informed defaults (see notebooks/01_eda.ipynb):
  - Class-weighted loss:  O=1.0 | B/I-DIAG=2.0 | B/I-PROC=4.0
    Rationale: 78.5% DIAG vs 21.5% PROC; O tokens dominate the sequence.
  - Stride 256 (not 128):  median span = 2 words, so larger stride is safe
    and halves the number of training windows.
  - Threshold 0.40:  favors recall on the short-span, long-tail code set.

Usage:
    python train_ner.py --no_wandb                              # full run
    python train_ner.py --no_wandb --epochs 1 --max_train_samples 50 --max_dev_samples 20  # smoke-test

Outputs:
    models/ner/               ← Best model + tokenizer
    outputs/ner_results.json  ← Dev-set evaluation metrics
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from src.data_loader import (
    LABEL2ID,
    ID2LABEL,
    load_annotations,
    build_bio_examples,
    resolve_tsv,
)
from src.evaluate import evaluate_ner


def _best_device() -> str:
    """Return the best available device string: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# ── EDA-derived class weights ──────────────────────────────────────────────────
# Label order must match LABEL2ID: O=0, B-DIAG=1, I-DIAG=2, B-PROC=3, I-PROC=4
# Rationale:
#   O tokens are ~97% of all tokens → weight 1.0 (baseline)
#   DIAG spans are 78.5% of annotations → weight 2.0 (moderate boost)
#   PROC spans are only 21.5% of annotations → weight 4.0 (strong boost)
DEFAULT_CLASS_WEIGHTS = [1.0, 2.0, 2.0, 4.0, 4.0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune CodiEsp NER model")
    p.add_argument("--train_tsv", default=None,
                   help="Path to train CodiEsp-X TSV; auto-detected if omitted")
    p.add_argument("--train_text", default="data/train/",
                   help="Train split dir (text_files/ is resolved automatically)")
    p.add_argument("--dev_tsv", default=None,
                   help="Path to dev CodiEsp-X TSV; auto-detected if omitted")
    p.add_argument("--dev_text", default="data/dev/",
                   help="Dev split dir")
    p.add_argument("--model_name", default="PlanTL-GOB-ES/bsc-bio-ehr-es")
    p.add_argument("--output_dir", default="models/ner/")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=2,
                   help="Per-device batch size. Keep at 2 on M1 (8 GB RAM) to avoid OOM. "
                        "Use --grad_accum to maintain effective batch size.")
    p.add_argument("--grad_accum", type=int, default=4,
                   help="Gradient accumulation steps. Effective batch = batch_size × grad_accum. "
                        "Default 4 → effective batch 8.")
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--stride", type=int, default=256,
                   help="Sliding window stride. 256 is safe given short spans (median 2 words).")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--seed", type=int, default=42)
    # fp16 only makes sense on CUDA — MPS doesn't support it
    p.add_argument("--fp16", action="store_true", default=False,
                   help="Enable fp16 mixed precision (CUDA only — do NOT set on M1/MPS).")
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader workers. 0=safe on Mac/Windows. 2-4 fine on Linux/Colab.")
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--max_train_samples", type=int, default=None,
                   help="Cap training to this many articles (by unique ID). "
                        "None = use all. Example: --max_train_samples 50")
    p.add_argument("--max_dev_samples", type=int, default=None,
                   help="Cap dev to this many articles. None = use all.")
    p.add_argument(
        "--class_weights",
        nargs=5,
        type=float,
        default=DEFAULT_CLASS_WEIGHTS,
        metavar=("O", "B_DIAG", "I_DIAG", "B_PROC", "I_PROC"),
        help="Loss weights for [O, B-DIAG, I-DIAG, B-PROC, I-PROC]. "
             f"Default: {DEFAULT_CLASS_WEIGHTS} (EDA-derived)",
    )
    return p.parse_args()


def build_hf_dataset(
    tsv_path: str,
    text_dir: str,
    tokenizer,
    max_length: int,
    stride: int,
    max_articles: int | None = None,
) -> Dataset:
    """Build a HuggingFace Dataset from CodiEsp annotations.

    Args:
        max_articles: If set, subsample the first N unique article IDs.
                      Useful for smoke-testing without waiting for full training.
    """
    anns = load_annotations(tsv_path)
    if max_articles is not None:
        # Keep only the first max_articles unique IDs (preserves ordering)
        seen, kept = set(), []
        for a in anns:
            if a.article_id not in seen:
                seen.add(a.article_id)
                if len(seen) > max_articles:
                    break
            if a.article_id in seen:
                kept.append(a)
        anns = kept
        print(f"  Subsampled to {len(seen)} articles ({len(anns)} annotations)")
    examples = build_bio_examples(text_dir, anns, tokenizer, max_length, stride)
    if not examples:
        raise RuntimeError(
            f"build_bio_examples returned 0 examples for {tsv_path}. "
            "Check that text_files/ paths are correct and TSV files are non-empty."
        )
    ds = Dataset.from_list(examples)
    ds = ds.remove_columns(["article_id", "window_idx"])
    return ds


def compute_metrics_fn(eval_pred):
    """compute_metrics callback for Trainer — reports overall + per-type F1."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    true_seqs, pred_seqs = [], []
    for pred_row, label_row in zip(preds, labels):
        true_seq, pred_seq = [], []
        for p, l in zip(pred_row, label_row):
            if l == -100:
                continue
            true_seq.append(ID2LABEL[l])
            pred_seq.append(ID2LABEL[p])
        true_seqs.append(true_seq)
        pred_seqs.append(pred_seq)

    metrics = evaluate_ner(true_seqs, pred_seqs, verbose=False)
    return {
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
    }


class WeightedTrainer(Trainer):
    """
    HuggingFace Trainer subclass that applies per-class loss weights.

    This addresses two imbalances found in EDA:
      1. O tokens dominate (~97% of all tokens)
      2. PROC spans (21.5%) are far rarer than DIAG spans (78.5%)
    """

    def __init__(self, class_weights: list, **kwargs):
        super().__init__(**kwargs)
        device = next(self.model.parameters()).device
        self.loss_fct = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float).to(device),
            ignore_index=-100,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits                       # (B, seq_len, num_labels)
        loss = self.loss_fct(
            logits.view(-1, self.model.config.num_labels),
            labels.view(-1),
        )
        return (loss, outputs) if return_outputs else loss


def main():
    args = parse_args()
    device = _best_device()

    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"

    print(f"Device     : {device.upper()}")
    if device == "mps":
        print("           : Apple M1/M2 MPS backend active — fp16 disabled, "
              "grad_accum compensates for small batch size")
    print(f"Batch size : {args.batch_size} (effective: {args.batch_size * args.grad_accum} "
          f"with grad_accum={args.grad_accum})")
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Auto-resolve TSV paths if not provided
    train_tsv = args.train_tsv or str(resolve_tsv("data", "train", task="X"))
    dev_tsv   = args.dev_tsv   or str(resolve_tsv("data", "dev",   task="X"))
    print(f"Train TSV : {train_tsv}")
    print(f"Dev   TSV : {dev_tsv}")

    print("Building training dataset …")
    train_ds = build_hf_dataset(
        train_tsv, args.train_text, tokenizer, args.max_length, args.stride,
        max_articles=args.max_train_samples,
    )
    print(f"  Training examples (windows): {len(train_ds)}")

    print("Building dev dataset …")
    dev_ds = build_hf_dataset(
        dev_tsv, args.dev_text, tokenizer, args.max_length, args.stride,
        max_articles=args.max_dev_samples,
    )
    print(f"  Dev examples (windows): {len(dev_ds)}")

    print(f"Loading model: {args.model_name}")
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        # fp16 only on CUDA — MPS will silently ignore or error
        fp16=args.fp16 and device == "cuda",
        seed=args.seed,
        logging_steps=50,
        # Mac: fork-based multiprocessing causes PyTorch hangs → keep at 0
        # On Linux/Colab: pass --num_workers 2 for a small throughput boost
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=device == "cuda",  # pin_memory only helps CUDA
        report_to="none" if args.no_wandb else "wandb",
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    print(f"Class weights: O={args.class_weights[0]} | "
          f"B/I-DIAG={args.class_weights[1]} | "
          f"B/I-PROC={args.class_weights[3]}")

    trainer = WeightedTrainer(
        class_weights=args.class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("Starting training …")
    trainer.train()

    print(f"Saving best model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Final evaluation on dev set:")
    results = trainer.evaluate()
    print(results)

    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/ner_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to outputs/ner_results.json")


if __name__ == "__main__":
    main()
