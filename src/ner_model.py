"""
src/ner_model.py

Spanish Biomedical NER model for CodiEsp span extraction.

Wraps a HuggingFace token-classification model with:
  - Sliding-window inference for long documents
  - Span aggregation (BIO → entity spans)
  - Confidence scoring per span
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)

from src.data_loader import ID2LABEL, LABEL2ID

# ── Default model ──────────────────────────────────────────────────────────────
DEFAULT_MODEL = "PlanTL-GOB-ES/bsc-bio-ehr-es"
# Fallbacks (comment in if the above is unavailable):
# DEFAULT_MODEL = "dccuchile/bert-base-spanish-wwm-cased"
# DEFAULT_MODEL = "xlm-roberta-base"


@dataclass
class MedicalSpan:
    """A medical concept span extracted from clinical text."""

    text: str
    start: int          # character offset in source text
    end: int
    entity_type: str    # "DIAG" | "PROC"
    confidence: float   # mean token probability for this span
    icd10_code: Optional[str] = None   # populated by Stage 2


def load_ner_model(
    model_path: str = DEFAULT_MODEL,
    device: Optional[str] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerFast]:
    """Load (or download) the NER model and tokenizer."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,  # fresh classification head
    )
    model.to(device)
    model.eval()
    return model, tokenizer


class NERPredictor:
    """
    Runs span extraction on a raw clinical text.

    Handles documents longer than the model's max_length via a
    sliding window with majority-vote token aggregation.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = 512,
        stride: int = 256,      # EDA: median span=2 words, 256 stride is safe
        threshold: float = 0.4, # EDA: long-tail codes → favour recall over precision
        device: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.threshold = threshold
        self.device = device or next(model.parameters()).device

    @torch.no_grad()
    def predict(self, text: str) -> List[MedicalSpan]:
        """Extract medical spans from a clinical text string."""
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            stride=self.stride,
            truncation=True,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_tensors="pt",
            padding="max_length",
        )

        # Aggregate per-character label probabilities across windows
        char_probs: Dict[int, List[float]] = {}  # char_idx → list of label_id probs
        char_label_votes: Dict[int, List[int]] = {}

        num_windows = len(encoding["input_ids"])
        for w_idx in range(num_windows):
            input_ids = encoding["input_ids"][w_idx].unsqueeze(0).to(self.device)
            attention_mask = encoding["attention_mask"][w_idx].unsqueeze(0).to(self.device)
            offsets = encoding["offset_mapping"][w_idx]

            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits  # (1, seq_len, num_labels)

            probs = F.softmax(logits[0], dim=-1)  # (seq_len, num_labels)

            for tok_idx, (tok_start, tok_end) in enumerate(offsets):
                tok_start, tok_end = int(tok_start), int(tok_end)
                if tok_start == tok_end:
                    continue  # special token
                tok_probs = probs[tok_idx].cpu().tolist()
                pred_label_id = int(torch.argmax(probs[tok_idx]))

                for char_pos in range(tok_start, tok_end):
                    char_probs.setdefault(char_pos, []).append(tok_probs)
                    char_label_votes.setdefault(char_pos, []).append(pred_label_id)

        # Build per-character majority label
        char_labels: Dict[int, Tuple[int, float]] = {}
        for char_pos, vote_list in char_label_votes.items():
            # majority vote
            majority_label = max(set(vote_list), key=vote_list.count)
            # mean max prob
            mean_conf = sum(max(p) for p in char_probs[char_pos]) / len(char_probs[char_pos])
            char_labels[char_pos] = (majority_label, mean_conf)

        # Convert character labels → entity spans
        return self._decode_spans(text, char_labels)

    def _decode_spans(
        self,
        text: str,
        char_labels: Dict[int, Tuple[int, float]],
    ) -> List[MedicalSpan]:
        """Convert per-character label predictions to MedicalSpan objects."""
        spans: List[MedicalSpan] = []
        chars = sorted(char_labels.keys())
        if not chars:
            return spans

        i = 0
        while i < len(chars):
            char_pos = chars[i]
            label_id, conf = char_labels[char_pos]
            label = ID2LABEL.get(label_id, "O")

            if label.startswith("B-"):
                entity_type = label[2:]  # "DIAG" | "PROC"
                span_start = char_pos
                span_chars = [char_pos]
                confs = [conf]
                # Greedily extend with I- tokens
                j = i + 1
                while j < len(chars):
                    next_pos = chars[j]
                    next_label_id, next_conf = char_labels[next_pos]
                    next_label = ID2LABEL.get(next_label_id, "O")
                    if next_label == f"I-{entity_type}":
                        span_chars.append(next_pos)
                        confs.append(next_conf)
                        j += 1
                    else:
                        break

                span_end = span_chars[-1] + 1
                span_text = text[span_start:span_end].strip()
                mean_conf = sum(confs) / len(confs)

                if span_text and mean_conf >= self.threshold:
                    spans.append(
                        MedicalSpan(
                            text=span_text,
                            start=span_start,
                            end=span_end,
                            entity_type=entity_type,
                            confidence=mean_conf,
                        )
                    )
                i = j
            else:
                i += 1

        return spans


def get_model_config(model_name: str = DEFAULT_MODEL) -> Dict:
    """Return training config dict for a given base model."""
    return {
        "model_name_or_path": model_name,
        "num_labels": len(LABEL2ID),
        "id2label": ID2LABEL,
        "label2id": LABEL2ID,
    }
