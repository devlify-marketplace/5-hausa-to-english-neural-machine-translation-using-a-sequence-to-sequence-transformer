"""Corpus-level translation evaluation with consistent English detokenization."""
from __future__ import annotations

from typing import Sequence

from sacrebleu.metrics import BLEU, CHRF, TER
from sacremoses import MosesDetokenizer


_DETOKENIZER = MosesDetokenizer(lang="en")


def _detokenize(text: str) -> str:
    """Convert the corpus' whitespace-tokenized punctuation back to readable text."""
    return _DETOKENIZER.detokenize(str(text).split()).strip()


def translation_metrics(predictions: Sequence[str], references: Sequence[str]) -> dict:
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same number of examples")
    if not predictions:
        raise ValueError("At least one prediction/reference pair is required")
    # The source corpus has spaces before punctuation. Apply identical detokenization
    # to hypotheses and references before SacreBLEU's own 13a tokenization.
    normalized_predictions = [_detokenize(x) for x in predictions]
    normalized_references = [_detokenize(x) for x in references]
    bleu_metric = BLEU()
    chrf_metric = CHRF(word_order=2)
    ter_metric = TER()
    bleu = bleu_metric.corpus_score(normalized_predictions, [normalized_references])
    chrf = chrf_metric.corpus_score(normalized_predictions, [normalized_references])
    ter = ter_metric.corpus_score(normalized_predictions, [normalized_references])
    exact_match = sum(p == r for p, r in zip(normalized_predictions, normalized_references)) / len(normalized_predictions)
    return {
        "examples": len(normalized_predictions),
        "sacrebleu": round(float(bleu.score), 4),
        "chrf_plus_plus": round(float(chrf.score), 4),
        "ter": round(float(ter.score), 4),
        "exact_match_rate": round(exact_match, 6),
        "metric_text_normalization": "English MosesDetokenizer applied consistently to hypotheses and references before scoring",
        "sacrebleu_signature": str(bleu_metric.get_signature()),
        "chrf_signature": str(chrf_metric.get_signature()),
        "ter_signature": str(ter_metric.get_signature()),
    }
