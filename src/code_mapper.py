"""
src/code_mapper.py

Stage 2: ICD-10 Code Mapper.

Maps extracted medical spans to ICD-10-CM (diagnosis) or ICD-10-PCS
(procedure) codes using a two-step retrieval + reranking approach:

  Step 1 — Bi-encoder retrieval (fast):
      Embed candidate span + ICD-10 code descriptions with a multilingual
      sentence-transformer; use FAISS for ANN search → Top-K candidates.

  Step 2 — Cross-encoder reranking (accurate):
      Re-score each (span_context, code_description) pair with a fine-tuned
      cross-encoder; return the top-1 code.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Optional heavy imports (lazy-loaded) ──────────────────────────────────────
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from sentence_transformers import SentenceTransformer, CrossEncoder

# ── Defaults ──────────────────────────────────────────────────────────────────
BI_ENCODER_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Multilingual cross-encoder — handles Spanish input paired with English ICD-10 descriptions.
# Replaces the English-only ms-marco model which scored Spanish/English pairs poorly.
CROSS_ENCODER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# ICD-10-CM descriptions bundled with this repo (downloaded separately or from CMS)
ICD10_CM_DESC_FILE = "data/icd10cm_descriptions.json"   # {code: description}
ICD10_PCS_DESC_FILE = "data/icd10pcs_descriptions.json"

# FAISS index paths (built on first run)
FAISS_CM_INDEX = "models/faiss_cm.index"
FAISS_PCS_INDEX = "models/faiss_pcs.index"
CM_CODES_FILE = "models/cm_codes.pkl"
PCS_CODES_FILE = "models/pcs_codes.pkl"


# ── ICD-10 Description Loader ─────────────────────────────────────────────────

def _load_icd10_descriptions(json_path: str | Path) -> Tuple[List[str], List[str]]:
    """
    Load ICD-10 code descriptions from a JSON file.

    Returns (codes, descriptions) as parallel lists.
    The JSON should be {code: description_string}.
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(
            f"ICD-10 description file not found: {path}\n"
            "Please run `python scripts/build_icd10_index.py` to generate it."
        )
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    
    codes = []
    descriptions = []
    for code, desc_string in data.items():
        # Our custom index joins synonyms with ' | ', we must flatten them
        # so each synonym gets its own distinct FAISS vector.
        synonyms = [s.strip() for s in desc_string.split(" | ")]
        for syn in synonyms:
            codes.append(code)
            descriptions.append(syn)
            
    return codes, descriptions


# ── FAISS Index Builder ───────────────────────────────────────────────────────

def build_faiss_index(
    encoder: SentenceTransformer,
    descriptions: List[str],
    index_path: str | Path,
    codes_path: str | Path,
    batch_size: int = 256,
) -> Tuple["faiss.Index", List[str]]:
    """
    Embed all ICD-10 descriptions and build a FAISS flat L2 index.

    Saves the index and code list to disk for reuse.
    """
    if not FAISS_AVAILABLE:
        raise ImportError("faiss-cpu is not installed. Run: pip install faiss-cpu")

    print(f"Encoding {len(descriptions)} ICD-10 descriptions …")
    embeddings = encoder.encode(
        descriptions,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product = cosine sim (with L2-normed vecs)
    index.add(embeddings.astype(np.float32))

    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    with open(codes_path, "wb") as f:
        pickle.dump(descriptions, f)
    print(f"Saved FAISS index → {index_path}")
    return index, descriptions


def load_or_build_index(
    encoder: SentenceTransformer,
    desc_json: str,
    index_path: str,
    codes_path: str,
) -> Tuple["faiss.Index", List[str], List[str]]:
    """Load pre-built FAISS index or build it from scratch."""
    codes, descriptions = _load_icd10_descriptions(desc_json)
    idx_p = Path(index_path)
    cod_p = Path(codes_path)

    if idx_p.exists() and cod_p.exists():
        print(f"Loading cached FAISS index from {idx_p}")
        index = faiss.read_index(str(idx_p))
        with open(cod_p, "rb") as f:
            descriptions = pickle.load(f)
        return index, codes, descriptions

    index, descriptions = build_faiss_index(encoder, descriptions, idx_p, cod_p)
    return index, codes, descriptions


# ── Main CodeMapper Class ─────────────────────────────────────────────────────

class CodeMapper:
    """
    Maps a medical span (text) to the best ICD-10 code.

    Usage:
        mapper = CodeMapper()
        mapper.load()
        result = mapper.map_span("hipertensión arterial", entity_type="DIAG")
        # result → {"code": "I10", "description": "Essential hypertension", "score": 0.97}
    """

    def __init__(
        self,
        bi_encoder_model: str = BI_ENCODER_MODEL,
        cross_encoder_model: str = CROSS_ENCODER_MODEL,
        top_k: int = 20,
        icd10cm_json: str = ICD10_CM_DESC_FILE,
        icd10pcs_json: str = ICD10_PCS_DESC_FILE,
    ):
        self.top_k = top_k
        self._bi_enc_name = bi_encoder_model
        self._cross_enc_name = cross_encoder_model
        self._icd10cm_json = icd10cm_json
        self._icd10pcs_json = icd10pcs_json

        self.bi_encoder: Optional[SentenceTransformer] = None
        self.cross_encoder: Optional[CrossEncoder] = None

        # Separate indexes for CM (diagnoses) and PCS (procedures)
        self._cm_index = None
        self._cm_codes: List[str] = []
        self._cm_descs: List[str] = []

        self._pcs_index = None
        self._pcs_codes: List[str] = []
        self._pcs_descs: List[str] = []

    def load(self) -> "CodeMapper":
        """Load encoders and FAISS indexes (builds them if not cached)."""
        print("Loading bi-encoder …")
        self.bi_encoder = SentenceTransformer(self._bi_enc_name)

        print("Loading cross-encoder …")
        self.cross_encoder = CrossEncoder(self._cross_enc_name)

        print("Loading ICD-10-CM index …")
        self._cm_index, self._cm_codes, self._cm_descs = load_or_build_index(
            self.bi_encoder,
            self._icd10cm_json,
            FAISS_CM_INDEX,
            CM_CODES_FILE,
        )

        print("Loading ICD-10-PCS index …")
        self._pcs_index, self._pcs_codes, self._pcs_descs = load_or_build_index(
            self.bi_encoder,
            self._icd10pcs_json,
            FAISS_PCS_INDEX,
            PCS_CODES_FILE,
        )

        return self

    def _retrieve_candidates(
        self,
        query_text: str,
        entity_type: str,
    ) -> List[Tuple[str, str, float]]:
        """
        Bi-encoder retrieval → top-K (code, description, bi_score) tuples.
        """
        index = self._cm_index if entity_type == "DIAG" else self._pcs_index
        codes = self._cm_codes if entity_type == "DIAG" else self._pcs_codes
        descs = self._cm_descs if entity_type == "DIAG" else self._pcs_descs

        q_emb = self.bi_encoder.encode(
            [query_text], normalize_embeddings=True, convert_to_numpy=True
        )
        scores, indices = index.search(q_emb.astype(np.float32), self.top_k)

        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            candidates.append((codes[idx], descs[idx], float(score)))
        return candidates

    def _rerank(
        self,
        query_text: str,
        candidates: List[Tuple[str, str, float]],
    ) -> List[Tuple[str, str, float]]:
        """
        Cross-encoder reranking of (query, code_description) pairs.
        Returns candidates sorted by cross-encoder score (descending).
        """
        pairs = [(query_text, desc) for _, desc, _ in candidates]
        cross_scores = self.cross_encoder.predict(pairs)
        reranked = sorted(
            zip([c[0] for c in candidates], [c[1] for c in candidates], cross_scores),
            key=lambda x: x[2],
            reverse=True,
        )
        return reranked

    def map_span(
        self,
        span_text: str,
        entity_type: str,
        context: Optional[str] = None,
        return_top_k: int = 1,
    ) -> List[Dict]:
        """
        Map a medical span to ICD-10 code(s).

        Args:
            span_text:    The extracted span string (e.g., "hipertensión arterial").
            entity_type:  "DIAG" or "PROC".
            context:      Optional surrounding sentence for better retrieval.
            return_top_k: Number of results to return.

        Returns list of dicts:
            [{"code": "I10", "description": "...", "score": 0.97}, ...]
        """
        query = f"{span_text}. {context}" if context else span_text
        candidates = self._retrieve_candidates(query, entity_type)
        if not candidates:
            return []

        reranked = self._rerank(query, candidates)
        return [
            {"code": code, "description": desc, "score": round(float(score), 4)}
            for code, desc, score in reranked[:return_top_k]
        ]

    def map_batch(
        self,
        spans: List[Dict],  # [{"text": ..., "entity_type": ..., "context": ...}]
        return_top_k: int = 1,
    ) -> List[Dict]:
        """
        Batch-map a list of spans to ICD-10 codes.

        Key optimizations vs calling map_span() per span in a loop:
          1. All span texts are encoded by the bi-encoder in ONE batched call.
          2. All (span, candidate_description) pairs are scored by the
             cross-encoder in ONE batched call.

        This cuts GPU round-trips from O(spans) to O(1) per document.
        """
        if not spans:
            return []

        # Build query strings (span text + optional context)
        queries = [
            f"{s['text']}. {s['context']}" if s.get("context") else s["text"]
            for s in spans
        ]

        # Step 1: Encode ALL queries in one bi-encoder call
        query_embs = self.bi_encoder.encode(
            queries,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=64,
            show_progress_bar=False,
        ).astype(np.float32)

        # Step 2: FAISS retrieval for each span (very fast, CPU)
        all_candidates: List[List[Tuple[str, str, float]]] = []
        for i, span in enumerate(spans):
            index = self._cm_index  if span["entity_type"] == "DIAG" else self._pcs_index
            codes = self._cm_codes  if span["entity_type"] == "DIAG" else self._pcs_codes
            descs = self._cm_descs  if span["entity_type"] == "DIAG" else self._pcs_descs

            scores, indices = index.search(query_embs[i : i + 1], self.top_k)
            candidates = [
                (codes[idx], descs[idx], float(score))
                for score, idx in zip(scores[0], indices[0])
                if idx >= 0
            ]
            all_candidates.append(candidates)

        # Step 3: Bypass Cross-Encoder due to severe ontology mismatch
        # US CMS descriptions and CIE-10 (CodiEsp) have suffix mismatches that
        # caused true semantic matches to be treated as hard negatives during
        # reranker training (e.g., E11 vs E11.9). The Bi-Encoder alone is safer.
        results = []
        for span, candidates in zip(spans, all_candidates):
            # Sort by the bi-encoder's score (which is candidate[2])
            reranked = sorted(candidates, key=lambda x: x[2], reverse=True)
            preds = [
                {"code": code, "description": desc, "score": round(cs, 4)}
                for code, desc, cs in reranked[:return_top_k]
            ]
            results.append({**span, "predictions": preds})

        return results
