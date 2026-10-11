# Hausa-to-English Neural Machine Translation Using a Sequence-to-Sequence Transformer

**Topic 5** from the supplied final-year project list. This is a complete, runnable Hausa (`hau`) → English (`eng`) research prototype built around a pretrained Marian encoder–decoder Transformer and fine-tuned on a public Hausa-English parallel corpus. It includes data acquisition and provenance, leakage-aware splits, baseline and fine-tuned evaluation, a saved local model, a REST API, a browser demo, tests and an academic report.

> **Important evaluation scope.** The corpus is an MT560-derived, mixed-source parallel collection, whose preview includes religious material, while corpus-wide domain proportions are undocumented. The pretrained OPUS-MT model was also trained with OPUS resources, so overlap with held-out examples cannot be ruled out. The delivered held-out scores are an internal split estimate, not an independent benchmark or a measure of translation quality for all Hausa speakers and domains. The bundled local fine-tuning run uses 3,000 training rows and 256 validation/test rows for practical CPU reproducibility; all-row training is supported by the code.

## Included

- `scripts/download_data.py`: downloads the public 295k-row parquet mirror; records retrieval time and SHA-256; executes no remote code.
- `scripts/prepare_data.py` and `src/hausa_nmt/data.py`: orient examples Hausa→English, preserve diacritics, remove exact duplicate pairs and split by unique Hausa source sentence so duplicate inputs do not cross partitions.
- `src/hausa_nmt/train.py`: evaluates the original Helsinki OPUS-MT checkpoint, fine-tunes a Marian sequence-to-sequence Transformer, chooses the checkpoint by validation loss, and evaluates both versions on the same held-out sample using SacreBLEU, chrF++ and TER.
- `artifacts/model/`: local fine-tuned model and tokenizer, usable without a model download.
- `artifacts/test_metrics.json`, `test_predictions.csv`, `training_history.csv`, `model_metadata.json`: measured run outputs and configuration.
- `src/hausa_nmt/api.py` and `web/index.html`: local FastAPI API and responsive translation demo.
- `tests/`: offline unit/API tests; no model download needed.
- `docs/`: project report, results, data provenance, model card and architecture diagram.

## Requirements and setup

Python 3.10–3.12 and approximately 2 GB of free disk space for the environment and model. CPU is supported; an NVIDIA GPU is optional. The completed bundle includes the processed data and fine-tuned model, so local inference does not need to download model weights. Internet is needed to install Python packages or refresh the corpus/base checkpoint for training.

```bash
cd /workspace/topic-05-hausa-to-english-nmt
python3 -m venv .venv
source .venv/bin/activate                 # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e '.[test]'
```

Alternatively, use `pip install -r requirements.txt && pip install -e .`.

## Data and training

The bundle includes the raw MT560-derived parquet file and deterministic processed partitions. To retrieve a fresh copy and rebuild the partitions:

```bash
python scripts/download_data.py --root . --overwrite
python scripts/prepare_data.py --root . --seed 42
```

Fine-tune and reproduce the delivered CPU-sized run:

```bash
python -m hausa_nmt.train --root .
python scripts/summarize_results.py
```

Default run: seed 42, 3,000 randomly sampled training pairs, 256 validation pairs, 256 held-out test pairs, one epoch, batch size 8, learning rate `2e-5`, max source/target length 128, and beam size 4. Validation loss selects the checkpoint. The test split is not used for parameter updates or checkpoint selection.

To train on every prepared training row and evaluate on every validation/test row:

```bash
python -m hausa_nmt.train --root . --max-train-samples 0 --max-eval-samples 0
```

Full-data fine-tuning is much more compute-intensive. You may also adjust values in `configs/default.json`. For a meaningful experiment, tune only on the training/validation partitions and preserve the test partition for final reporting.

## Run the browser demo and API

The project includes the local fine-tuned checkpoint. From the project root with the environment activated:

```bash
uvicorn hausa_nmt.api:app --app-dir src --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. Interactive docs: <http://127.0.0.1:8000/docs>.

- `GET /api/health` — model readiness.
- `POST /api/translate` — translate one Hausa text.
- `POST /api/translate-batch` — translate up to 20 texts.

Example:

```bash
curl -X POST http://127.0.0.1:8000/api/translate \
  -H 'Content-Type: application/json' \
  -d '{"text":"Na gode sosai."}'
```

The API processes text in memory and does not persist requests. Do not submit private or sensitive material. This research prototype must not be used as a certified translation or for high-impact decisions.

## Tests

```bash
pytest -q
```

Tests cover source/target orientation, Unicode/diacritic preservation, deterministic source-grouped splits, translation metric output, input validation, service readiness and API response shape without downloading model weights.

## Dataset and model citations

- Vaswani, A. et al. (2017). *Attention Is All You Need*. Advances in Neural Information Processing Systems.
- Junczys-Dowmunt, M. et al. (2018). *Marian: Fast Neural Machine Translation in C++*. ACL System Demonstrations.
- Gowda, T., Zhang, Z., Mattmann, C. A., & May, J. (2021). *Many-to-English Machine Translation Tools, Data, and Pretrained Models*. ACL System Demonstrations. <https://doi.org/10.18653/v1/2021.acl-demo.37>
- Tiedemann, J. (2009). *News from OPUS — A Collection of Multilingual Parallel Corpora with Tools and Interfaces*. Recent Advances in Natural Language Processing. <https://opus.nlpl.eu/>
- Helsinki-NLP. *Helsinki-NLP/opus-mt-ha-en model card*. <https://huggingface.co/Helsinki-NLP/opus-mt-ha-en>
- Hugging Face. *English-Hausa Parallel Dataset (MT560-derived) card*. <https://huggingface.co/datasets/michsethowusu/english-hausa_sentence-pairs_mt560>
- Post, M. (2018). *A Call for Clarity in Reporting BLEU Scores*. WMT.
- Popović, M. (2015). *chrF: character n-gram F-score for automatic MT evaluation*. WMT.
- Hugging Face Transformers. *Translation task guide*. <https://huggingface.co/docs/transformers/en/tasks/translation>

Code is MIT licensed. The model checkpoint is listed as Apache-2.0 by its publisher. The dataset mirror reports CC BY 4.0, but MT560 aggregates material from multiple corpora with potentially distinct source conditions; consult `docs/DATA_PROVENANCE.md` and upstream notices before redistribution or commercial use.
