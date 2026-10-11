"""Fine-tune an OPUS Marian Transformer and evaluate on held-out Hausa-English pairs."""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sacremoses import MosesDetokenizer
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

from hausa_nmt.data import load_split, prepare_dataset
from hausa_nmt.metrics import translation_metrics


class ParallelTextDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, tokenizer, source_max_length: int, target_max_length: int):
        self.sources = rows["source"].astype(str).tolist()
        self.targets = rows["target"].astype(str).tolist()
        self.tokenizer = tokenizer
        self.source_max_length = source_max_length
        self.target_max_length = target_max_length

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, index):
        encoded = self.tokenizer(
            self.sources[index], max_length=self.source_max_length, truncation=True,
        )
        target = self.tokenizer(
            text_target=self.targets[index], max_length=self.target_max_length, truncation=True,
        )
        encoded["labels"] = target["input_ids"]
        return encoded


def _generate(model, tokenizer, rows: pd.DataFrame, batch_size: int, max_source_length: int, max_new_tokens: int, num_beams: int) -> list[str]:
    device = next(model.parameters()).device
    outputs: list[str] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            texts = rows.source.iloc[start:start + batch_size].astype(str).tolist()
            batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_source_length)
            batch = {key: value.to(device) for key, value in batch.items()}
            generated = model.generate(
                **batch, num_beams=num_beams, max_new_tokens=max_new_tokens,
                early_stopping=True,
            )
            outputs.extend(x.strip() for x in tokenizer.batch_decode(generated, skip_special_tokens=True))
            done = min(start + len(texts), len(rows))
            if done == len(rows) or done % max(batch_size * 20, batch_size) == 0:
                print(f"  generated {done}/{len(rows)} translations", flush=True)
    return outputs


def _evaluate_model(model, tokenizer, rows: pd.DataFrame, cfg: dict, label: str) -> tuple[dict, list[str]]:
    started = time.time()
    predictions = _generate(
        model, tokenizer, rows, int(cfg["inference_batch_size"]), int(cfg["max_source_length"]),
        int(cfg["max_new_tokens"]), int(cfg["num_beams"]),
    )
    metrics = translation_metrics(predictions, rows.target.astype(str).tolist())
    metrics["elapsed_seconds"] = round(time.time() - started, 2)
    print(f"{label} evaluation: BLEU={metrics['sacrebleu']:.2f}, chrF++={metrics['chrf_plus_plus']:.2f}, TER={metrics['ter']:.2f}", flush=True)
    return metrics, predictions


def _read_config(root: Path) -> dict:
    with (root / "configs" / "default.json").open(encoding="utf-8") as f:
        return json.load(f)


def run(root: Path, max_train_samples: int | None = None, max_eval_samples: int | None = None) -> dict:
    root = root.resolve()
    cfg = _read_config(root)
    if max_train_samples is not None:
        cfg["max_train_samples"] = max_train_samples
    if max_eval_samples is not None:
        cfg["max_eval_samples"] = max_eval_samples
    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    raw_path = root / cfg["raw_data_path"]
    processed_dir = root / cfg["processed_data_dir"]
    split_files = {name: processed_dir / f"ha_en_{name}.csv" for name in ("train", "validation", "test")}
    if not all(path.exists() for path in split_files.values()):
        manifest = prepare_dataset(raw_path, processed_dir, seed=seed)
    else:
        manifest = json.loads((processed_dir / "manifest.json").read_text(encoding="utf-8"))

    train_rows = load_split(split_files["train"], int(cfg["max_train_samples"]), seed)
    val_rows = load_split(split_files["validation"], int(cfg["max_eval_samples"]), seed + 1)
    test_rows = load_split(split_files["test"], int(cfg["max_eval_samples"]), seed + 2)
    print(f"Corpus pairs: {manifest['usable_unique_pair_count']}; run rows: train={len(train_rows)}, validation={len(val_rows)}, test={len(test_rows)}", flush=True)
    if min(len(train_rows), len(val_rows), len(test_rows)) == 0:
        raise ValueError("Train, validation and test sets must all be non-empty")

    model_id = cfg["base_model"]
    print(f"Loading base sequence-to-sequence Transformer: {model_id}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    # Establish the pretrained baseline on the same test sample before fine-tuning.
    baseline_model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    baseline_model.to(device)
    baseline_metrics, baseline_predictions = _evaluate_model(baseline_model, tokenizer, test_rows, cfg, "Base model")
    del baseline_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    train_dataset = ParallelTextDataset(train_rows, tokenizer, int(cfg["max_source_length"]), int(cfg["max_target_length"]))
    validation_dataset = ParallelTextDataset(val_rows, tokenizer, int(cfg["max_source_length"]), int(cfg["max_target_length"]))
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
    run_dir = root / "artifacts" / "training_run"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    args = Seq2SeqTrainingArguments(
        output_dir=str(run_dir),
        learning_rate=float(cfg["learning_rate"]),
        per_device_train_batch_size=int(cfg["train_batch_size"]),
        per_device_eval_batch_size=int(cfg["eval_batch_size"]),
        num_train_epochs=float(cfg["epochs"]),
        weight_decay=float(cfg["weight_decay"]),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=max(1, min(25, len(train_dataset) // max(1, int(cfg["train_batch_size"]) * 10))),
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=False,
        report_to="none",
        seed=seed,
        data_seed=seed,
        use_cpu=(device == "cpu"),
        fp16=(device == "cuda"),
        dataloader_num_workers=0,
        save_safetensors=True,
        disable_tqdm=False,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )
    print("Fine-tuning started; the test split is not used for checkpoint selection.", flush=True)
    train_result = trainer.train()
    best_checkpoint = trainer.state.best_model_checkpoint
    selected_epoch = trainer.state.best_global_step
    eval_history = [entry for entry in trainer.state.log_history if "eval_loss" in entry]
    best_validation_metrics = min(eval_history, key=lambda entry: entry["eval_loss"]) if eval_history else {}
    model_dir = root / "artifacts" / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(model_dir)

    tuned_metrics, tuned_predictions = _evaluate_model(trainer.model, tokenizer, test_rows, cfg, "Fine-tuned model")
    results = {
        "project": "Topic 5 - Hausa-to-English NMT",
        "evaluation_protocol": "same deterministic held-out test sample for base and fine-tuned models; grouped by unique Hausa source string before sampling",
        "selection_metric": "validation loss (lower is better)",
        "base_model": model_id,
        "dataset": cfg["dataset_id"],
        "seed": seed,
        "run_sample_counts": {"train": len(train_rows), "validation": len(val_rows), "test": len(test_rows)},
        "base_model_metrics": baseline_metrics,
        "fine_tuned_model_metrics": tuned_metrics,
        "metric_deltas_finetuned_minus_base": {
            metric: round(float(tuned_metrics[metric]) - float(baseline_metrics[metric]), 4)
            for metric in ("sacrebleu", "chrf_plus_plus", "ter", "exact_match_rate")
        },
        "note": "This is a held-out split of the selected MT560 corpus. The upstream OPUS-MT checkpoint is itself trained on OPUS resources; corpus overlap in pretraining cannot be ruled out, and results do not represent an independent benchmark or every Hausa domain.",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    artifacts = root / "artifacts"
    (artifacts / "test_metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    predictions_frame = pd.DataFrame({
        "hausa_source": test_rows.source.astype(str).tolist(),
        "english_reference": test_rows.target.astype(str).tolist(),
        "base_model_prediction": baseline_predictions,
        "fine_tuned_prediction": tuned_predictions,
    })
    detokenizer = MosesDetokenizer(lang="en")
    for raw_column, normalized_column in (("english_reference", "english_reference_normalized"),
                                         ("base_model_prediction", "base_model_prediction_normalized"),
                                         ("fine_tuned_prediction", "fine_tuned_prediction_normalized")):
        predictions_frame[normalized_column] = [detokenizer.detokenize(str(value).split()).strip()
                                                for value in predictions_frame[raw_column]]
    predictions_frame.to_csv(artifacts / "test_predictions.csv", index=False, encoding="utf-8")
    history = pd.DataFrame(trainer.state.log_history)
    history.to_csv(artifacts / "training_history.csv", index=False)
    metadata = {
        "model_name": "Topic 5 Hausa-to-English fine-tuned Marian Transformer",
        "model_path": "artifacts/model",
        "base_model": model_id,
        "architecture": "MarianMTModel (Transformer encoder-decoder)",
        "source_language": "Hausa (hau)",
        "target_language": "English (eng)",
        "training_config": cfg,
        "selected_training_checkpoint_name": Path(best_checkpoint).name if best_checkpoint else None,
        "selected_model_artifact": "artifacts/model",
        "selected_global_step": selected_epoch,
        "best_validation_metrics": best_validation_metrics,
        "corpus_manifest": manifest,
        "run_sample_counts": results["run_sample_counts"],
        "device": device,
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "test_metrics_file": "artifacts/test_metrics.json",
        "test_set_used_for_training_or_selection": False,
    }
    (artifacts / "model_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    if run_dir.exists():
        shutil.rmtree(run_dir)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2], help="Project root")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Training rows; 0 means use every train row")
    parser.add_argument("--max-eval-samples", type=int, default=None, help="Rows in each validation/test run; 0 means use every row")
    args = parser.parse_args()
    result = run(args.root, args.max_train_samples, args.max_eval_samples)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
