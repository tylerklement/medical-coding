"""
src/pipeline.py

End-to-end medical coding pipeline:
    Raw clinical text (.txt)
        → Stage 1: NER span extraction (bsc-bio-ehr-es)
        → Stage 2: ICD-10 code mapping (FAISS + cross-encoder)
        → Structured output: [{span, type, icd10_code, confidence, ...}]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from src.data_loader import LABEL2ID, ID2LABEL
from src.ner_model import MedicalSpan, NERPredictor, load_ner_model
from src.code_mapper import CodeMapper


class MedicalCodingPipeline:
    """
    High-level API for end-to-end clinical coding.

    Example:
        pipeline = MedicalCodingPipeline()
        pipeline.load()
        results = pipeline.process_text("El paciente presenta hipertensión …")
    """

    def __init__(
        self,
        ner_model_path: str = "PlanTL-GOB-ES/bsc-bio-ehr-es",
        bi_encoder_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        ner_threshold: float = 0.5,
        icd10cm_json: str = "data/icd10cm_descriptions.json",
        icd10pcs_json: str = "data/icd10pcs_descriptions.json",
        top_k_codes: int = 1,
        device: Optional[str] = None,
    ):
        self.ner_model_path = ner_model_path
        self.bi_encoder_model = bi_encoder_model
        self.cross_encoder_model = cross_encoder_model
        self.ner_threshold = ner_threshold
        self.icd10cm_json = icd10cm_json
        self.icd10pcs_json = icd10pcs_json
        self.top_k_codes = top_k_codes
        self.device = device

        self._ner_predictor: Optional[NERPredictor] = None
        self._code_mapper: Optional[CodeMapper] = None

    def load(self) -> "MedicalCodingPipeline":
        """Initialise all model components (lazy-loaded)."""
        print("── Loading NER model ────────────────────────────────────")
        model, tokenizer = load_ner_model(self.ner_model_path, device=self.device)
        self._ner_predictor = NERPredictor(
            model=model,
            tokenizer=tokenizer,
            threshold=self.ner_threshold,
        )

        print("── Loading Code Mapper ──────────────────────────────────")
        self._code_mapper = CodeMapper(
            bi_encoder_model=self.bi_encoder_model,
            cross_encoder_model=self.cross_encoder_model,
            icd10cm_json=self.icd10cm_json,
            icd10pcs_json=self.icd10pcs_json,
        ).load()

        print("── Pipeline ready ───────────────────────────────────────")
        return self

    def process_text(
        self,
        text: str,
        entity_types: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Run the full pipeline on a clinical text string.

        Args:
            text:         Raw Spanish clinical text.
            entity_types: Subset of ["DIAG", "PROC"]. None = both.

        Returns list of dicts:
        [
            {
                "span_text": "hipertensión arterial",
                "entity_type": "DIAG",
                "start": 45,
                "end": 66,
                "confidence": 0.92,
                "icd10_code": "I10",
                "code_description": "Essential hypertension",
                "code_score": 0.87,
            },
            ...
        ]
        """
        entity_types = entity_types or ["DIAG", "PROC"]

        # ── Stage 1: Span extraction ─────────────────────────────────────────
        spans: List[MedicalSpan] = self._ner_predictor.predict(text)
        spans = [s for s in spans if s.entity_type in entity_types]

        if not spans:
            return []

        # ── Stage 2: Code mapping ─────────────────────────────────────────────
        results = []
        for span in spans:
            # Extract surrounding context (up to 100 chars each side)
            ctx_start = max(0, span.start - 100)
            ctx_end = min(len(text), span.end + 100)
            context = text[ctx_start:ctx_end]

            preds = self._code_mapper.map_span(
                span_text=span.text,
                entity_type=span.entity_type,
                context=context,
                return_top_k=self.top_k_codes,
            )

            top_pred = preds[0] if preds else {"code": "UNKNOWN", "description": "", "score": 0.0}
            results.append(
                {
                    "span_text": span.text,
                    "entity_type": span.entity_type,
                    "start": span.start,
                    "end": span.end,
                    "confidence": round(span.confidence, 4),
                    "icd10_code": top_pred["code"],
                    "code_description": top_pred["description"],
                    "code_score": top_pred["score"],
                    "all_candidates": preds,
                }
            )

        return results

    def process_file(
        self,
        txt_path: str | Path,
        entity_types: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Process a .txt file and return coded results."""
        text = Path(txt_path).read_text(encoding="utf-8")
        return self.process_text(text, entity_types=entity_types)

    def to_tsv(self, results: List[Dict], article_id: str = "doc") -> str:
        """
        Format results in CodiEsp-X submission format (TSV):
            articleID  label  ICD10-code  text-reference  reference-position

        Note: CodiEsp uses lowercase codes (e.g. 'i10', 'bw03zzz') — we
        normalise here so predictions match the gold-label format.
        """
        lines = []
        for r in results:
            label = "DIAGNOSTICO" if r["entity_type"] == "DIAG" else "PROCEDIMIENTO"
            lines.append(
                "\t".join([
                    article_id,
                    label,
                    r["icd10_code"].lower(),   # CodiEsp uses lowercase codes
                    r["span_text"],
                    f"{r['start']} {r['end']}",
                ])
            )
        return "\n".join(lines)

