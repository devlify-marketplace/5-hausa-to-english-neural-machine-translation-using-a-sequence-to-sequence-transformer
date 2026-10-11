#!/usr/bin/env python3
"""Recompute evaluation metadata from stored predictions and current metric code."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from sacremoses import MosesDetokenizer
from hausa_nmt.metrics import translation_metrics


def finalize(root: Path) -> None:
    artifacts = root / "artifacts"
    metrics_path = artifacts / "test_metrics.json"
    prediction_path = artifacts / "test_predictions.csv"
    metadata_path = artifacts / "model_metadata.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(prediction_path, keep_default_na=False)
    base = translation_metrics(frame.base_model_prediction.tolist(), frame.english_reference.tolist())
    tuned = translation_metrics(frame.fine_tuned_prediction.tolist(), frame.english_reference.tolist())
    base["elapsed_seconds"] = metrics["base_model_metrics"].get("elapsed_seconds")
    tuned["elapsed_seconds"] = metrics["fine_tuned_model_metrics"].get("elapsed_seconds")
    metrics["base_model_metrics"] = base
    metrics["fine_tuned_model_metrics"] = tuned
    metrics["metric_deltas_finetuned_minus_base"] = {
        key: round(float(tuned[key]) - float(base[key]), 4)
        for key in ("sacrebleu", "chrf_plus_plus", "ter", "exact_match_rate")
    }
    metrics["metric_note"] = "The English MosesDetokenizer was applied consistently to hypotheses and references before corpus SacreBLEU, chrF++ and TER."
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    detokenizer = MosesDetokenizer(lang="en")
    for source, target in (("english_reference", "english_reference_normalized"),
                           ("base_model_prediction", "base_model_prediction_normalized"),
                           ("fine_tuned_prediction", "fine_tuned_prediction_normalized")):
        frame[target] = [detokenizer.detokenize(str(value).split()).strip() for value in frame[source]]
    frame.to_csv(prediction_path, index=False, encoding="utf-8")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    old_checkpoint = metadata.pop("selected_checkpoint", None)
    metadata["selected_training_checkpoint_name"] = Path(old_checkpoint).name if old_checkpoint else f"checkpoint-{metadata.get('selected_global_step', 'unknown')}"
    metadata["selected_model_artifact"] = "artifacts/model"
    history_path = artifacts / "training_history.csv"
    history = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
    eval_rows = history.dropna(subset=["eval_loss"]) if "eval_loss" in history.columns else pd.DataFrame()
    metadata["best_validation_metrics"] = eval_rows.loc[eval_rows.eval_loss.idxmin()].dropna().to_dict() if not eval_rows.empty else {}
    metadata["test_metric_text_normalization"] = "English MosesDetokenizer applied consistently to hypotheses and references before scoring"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"base": base, "fine_tuned": tuned, "deltas": metrics["metric_deltas_finetuned_minus_base"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    finalize(args.root.resolve())


if __name__ == "__main__":
    main()
