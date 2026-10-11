#!/usr/bin/env python3
"""Build a cautious human-readable result summary and comparison plot from run artifacts."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"


def main() -> None:
    metrics = json.loads((ART / "test_metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "data" / "processed" / "manifest.json").read_text(encoding="utf-8"))
    base, tuned = metrics["base_model_metrics"], metrics["fine_tuned_model_metrics"]
    run_counts = metrics["run_sample_counts"]
    metric_names = [("SacreBLEU", "sacrebleu", True), ("chrF++", "chrf_plus_plus", True), ("TER", "ter", False)]
    table_rows = []
    for label, key, higher in metric_names:
        table_rows.append(f"| {label} | {base[key]:.4f} | {tuned[key]:.4f} | {metrics['metric_deltas_finetuned_minus_base'][key]:+.4f} | {'Higher' if higher else 'Lower'} is better |")
    text = f'''# Experimental results

## Run and split summary

- **Run sample:** {run_counts['train']:,} training pairs, {run_counts['validation']:,} validation pairs and {run_counts['test']:,} held-out test pairs.
- **Available processed corpus:** {manifest['usable_unique_pair_count']:,} unique pairs; train {manifest['counts']['train']:,}, validation {manifest['counts']['validation']:,}, test {manifest['counts']['test']:,}.
- **Selection:** checkpoint with lowest validation loss; the test split was not used for parameter updates or checkpoint selection.
- **Decoding:** beam search, `{metrics.get('base_model', 'Helsinki-NLP/opus-mt-ha-en')}` base checkpoint.
- **Seed:** {metrics['seed']}.

## Same-sample model comparison

| Metric | Base model | Fine-tuned | Fine-tuned − base | Direction |
| --- | ---: | ---: | ---: | --- |
{chr(10).join(table_rows)}

Exact-match rate: base **{base['exact_match_rate']:.4f}**, fine-tuned **{tuned['exact_match_rate']:.4f}**.

![Base and fine-tuned test metric comparison](../artifacts/metrics_comparison.png)

Before scoring, Moses English detokenization is applied in the same way to hypotheses and references because MT560 punctuation is whitespace-tokenized. SacreBLEU and chrF++ are higher-is-better corpus scores; TER is lower-is-better. The corpus-level measurements may change with the evaluation subset, tokenization signature and software versions; see `artifacts/test_metrics.json` for full metric signatures and elapsed time.

## Interpretation limits

The reported test is a deterministic, bounded sample from an internal 10% partition of the MT560-derived corpus. The partition keeps identical Hausa source strings together, but the dataset mirror does not provide document provenance. Helsinki OPUS-MT was trained on OPUS material and MT560 is an OPUS-aggregated collection, so overlap with the pretrained model's training data cannot be ruled out. The corpus is heterogeneous and domain-skewed; the scores are not independent benchmark results and must not be generalized to other domains or all Hausa speakers. No bilingual human evaluation has been performed. A metric increase does not guarantee preserved meaning.

The complete example-level outputs and actual model/run metadata are in `../artifacts/test_predictions.csv` and `../artifacts/model_metadata.json`.
'''
    (ROOT / "docs" / "RESULTS.md").write_text(text, encoding="utf-8")

    labels = [x[0] for x in metric_names]
    keys = [x[1] for x in metric_names]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), constrained_layout=True)
    colors = ["#8ea59a", "#176b50"]
    for ax, label, key, higher in zip(axes, labels, keys, [True, True, False]):
        values = [base[key], tuned[key]]
        bars = ax.bar(["Base", "Fine-tuned"], values, color=colors, width=0.62)
        ax.set_title(label, fontweight="bold")
        ax.set_ylabel("Score")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        lo, hi = min(values), max(values)
        span = max(hi - lo, max(abs(hi), 1) * 0.08)
        ax.set_ylim(max(0, min(values) - span * .4), max(values) + span * .55)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
        ax.text(.5, -.22, "higher is better" if higher else "lower is better", transform=ax.transAxes, ha="center", fontsize=8, color="#60716b")
    fig.suptitle(f"Hausa → English test sample (n={run_counts['test']})", fontsize=13, fontweight="bold")
    fig.savefig(ART / "metrics_comparison.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    history_path = ART / "training_history.csv"
    if history_path.exists():
        history = pd.read_csv(history_path)
        if "loss" in history.columns and "step" in history.columns:
            training = history.dropna(subset=["loss"])
            validation = history.dropna(subset=["eval_loss"])
            if not training.empty or not validation.empty:
                fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
                if not training.empty:
                    ax.plot(training.step, training.loss, marker="o", ms=3, label="Training loss", color="#176b50")
                if not validation.empty:
                    ax.scatter(validation.step, validation.eval_loss, marker="D", s=40, label="Validation loss", color="#c18729", zorder=3)
                ax.set(xlabel="Training step", ylabel="Loss", title="Training and validation loss")
                ax.grid(alpha=.2); ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
                fig.savefig(ART / "training_history.png", dpi=180, bbox_inches="tight", facecolor="white")
                plt.close(fig)
    print(f"Wrote {ROOT / 'docs' / 'RESULTS.md'} and {ART / 'metrics_comparison.png'}")


if __name__ == "__main__":
    main()
