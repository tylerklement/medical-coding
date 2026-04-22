#!/usr/bin/env python3
"""
predict.py — CLI for end-to-end medical coding inference.

Usage:
    # Single file:
    python predict.py --input data/test/text/case001.txt

    # Batch mode (entire directory):
    python predict.py --input data/test/text/ --output outputs/predictions/

    # Output as CodiEsp-X submission TSV:
    python predict.py --input data/test/text/ --format tsv

    # Diagnoses only:
    python predict.py --input data/test/text/ --types DIAG

    # Use fine-tuned models:
    python predict.py --input data/test/text/ \\
        --ner_model models/ner/ \\
        --cross_encoder models/reranker/

Options:
    --input            Path to a .txt file or directory of .txt files
    --output           Output directory for predictions (default: outputs/predictions/)
    --ner_model        NER model path (HuggingFace hub or local)
    --cross_encoder    Cross-encoder model path
    --types            Entity types to extract: DIAG, PROC, or both (default: both)
    --top_k            Number of code candidates to return per span (default: 1)
    --threshold        NER confidence threshold (default: 0.5)
    --format           Output format: json | tsv (default: json)
    --device           Device: cpu | cuda | mps (default: auto-detect)
"""

import argparse
import json
import sys
from pathlib import Path

from src.pipeline import MedicalCodingPipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Medical coding CLI — extracts ICD-10 codes from clinical texts"
    )
    p.add_argument("--input", required=True, help="Path to .txt file or directory")
    p.add_argument("--output", default="outputs/predictions/", help="Output directory")
    p.add_argument("--ner_model", default="PlanTL-GOB-ES/bsc-bio-ehr-es")
    p.add_argument("--cross_encoder", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    p.add_argument("--icd10cm_json", default="data/icd10cm_descriptions.json")
    p.add_argument("--icd10pcs_json", default="data/icd10pcs_descriptions.json")
    p.add_argument("--types", nargs="+", default=["DIAG", "PROC"],
                   choices=["DIAG", "PROC"])
    p.add_argument("--top_k", type=int, default=1)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--format", choices=["json", "tsv"], default="json")
    p.add_argument("--device", default=None)
    return p.parse_args()


def collect_txt_files(path: str) -> list:
    p = Path(path)
    if p.is_file():
        return [p]
    elif p.is_dir():
        files = sorted(p.glob("*.txt"))
        if not files:
            print(f"No .txt files found in {p}", file=sys.stderr)
            sys.exit(1)
        return files
    else:
        print(f"Input path does not exist: {p}", file=sys.stderr)
        sys.exit(1)


def main():
    args = parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load pipeline ─────────────────────────────────────────────────────────
    pipeline = MedicalCodingPipeline(
        ner_model_path=args.ner_model,
        cross_encoder_model=args.cross_encoder,
        icd10cm_json=args.icd10cm_json,
        icd10pcs_json=args.icd10pcs_json,
        ner_threshold=args.threshold,
        top_k_codes=args.top_k,
        device=args.device,
    ).load()

    # ── Process files ─────────────────────────────────────────────────────────
    txt_files = collect_txt_files(args.input)
    print(f"\nProcessing {len(txt_files)} file(s) …\n")

    all_results = {}
    tsv_lines = []

    for txt_path in txt_files:
        article_id = txt_path.stem
        print(f"  [{article_id}]", end=" ", flush=True)

        results = pipeline.process_file(txt_path, entity_types=args.types)
        all_results[article_id] = results

        n_diag = sum(1 for r in results if r["entity_type"] == "DIAG")
        n_proc = sum(1 for r in results if r["entity_type"] == "PROC")
        print(f"{n_diag} diagnoses, {n_proc} procedures")

        # Build TSV lines for CodiEsp-X submission format
        tsv_lines.append(pipeline.to_tsv(results, article_id=article_id))

    # ── Write output ──────────────────────────────────────────────────────────
    if args.format == "json":
        out_file = output_dir / "predictions.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Saved JSON predictions → {out_file}")
    else:
        out_file = output_dir / "predictions.tsv"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write("\n".join(tsv_lines))
        print(f"\n✅ Saved TSV predictions → {out_file}")

    # ── Print summary table ───────────────────────────────────────────────────
    total_spans = sum(len(v) for v in all_results.values())
    print(f"\n{'─'*50}")
    print(f"Files processed : {len(txt_files)}")
    print(f"Total spans     : {total_spans}")
    print(f"{'─'*50}")


if __name__ == "__main__":
    main()
