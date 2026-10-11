# Model card: Hausa-to-English NMT

## Summary

- **Project:** Topic 5, Hausa-to-English Neural Machine Translation Using a Sequence-to-Sequence Transformer
- **Direction:** Hausa (`hau`) → English (`eng`)
- **Architecture:** `MarianMTModel`, encoder–decoder Transformer; six encoder and six decoder layers, hidden width 512, eight attention heads, 2,048-unit feed-forward blocks as reported in the upstream model configuration
- **Base checkpoint:** [Helsinki-NLP/opus-mt-ha-en](https://huggingface.co/Helsinki-NLP/opus-mt-ha-en), model card reports Apache-2.0
- **Fine-tuning corpus:** MT560-derived Hausa-English parallel pairs; see [`DATA_PROVENANCE.md`](DATA_PROVENANCE.md)
- **Tokenizer:** upstream SentencePiece-based Marian tokenizer
- **Primary artifact:** `artifacts/model/`

## Intended use

Educational demonstration and controlled research on low-resource machine translation. It can translate short Hausa text into English for qualitative inspection. The user should compare output with the original and obtain human review where correctness matters.

## Out of scope

The model is not suitable for certified translation, medical/legal/safety advice, official records, high-impact decisions, public-facing production translation or replacing a qualified Hausa-English translator. The upstream Helsinki model is a pretrained baseline; this project fine-tunes it but does not claim to have trained a new architecture from random initialization.

## Training and evaluation

The workflow uses a seed-fixed 80/10/10 source-grouped split of the MT560-derived pair collection; identical Hausa source strings remain in one partition. The default local run fine-tunes on 3,000 randomly sampled training pairs for one epoch, using validation loss to select the checkpoint. The test split is not used for gradient updates or checkpoint selection. The pipeline evaluates the base and fine-tuned checkpoints on the same held-out sample using corpus SacreBLEU, chrF++ (character n-gram F-score with word order 2), TER (translation edit rate) and exact-match rate. Actual run settings and scores are recorded in `artifacts/model_metadata.json` and `artifacts/test_metrics.json`.

**Important:** the upstream OPUS-MT checkpoint was trained on OPUS resources. MT560 itself incorporates OPUS and related sources. The Hugging Face mirror lacks document-level source provenance, so pretraining overlap with this evaluation data cannot be ruled out. The scores are therefore internal held-out split measurements, not independent external benchmark results. See `docs/RESULTS.md` for the exact sampled evaluation size and cautious interpretation.

## Limitations and risks

- Data-source domain bias, especially possible overrepresentation of religious or translated written material, may lead to poor performance on informal conversation, news, education, health, or other domains.
- Hausa dialects, orthographic variation, diacritics, code-switching and transliteration may be underrepresented.
- Parallel-corpus references may be inconsistent; an alternative but valid translation can score poorly against a single reference, while overlap-based scores do not prove meaning preservation.
- Translations can omit, add or alter meaning. The model may emit fluent but incorrect English.
- Confidence is not calibrated or exposed as a probability.
- The system is single-sentence oriented and input is truncated at the configured maximum length.
- No human evaluation or community safety assessment is included in the current automated run.

## Privacy

The local FastAPI demo does not persist submitted text. Running the demo on a shared/public host would require separate security, privacy and rate-limiting review. Avoid entering sensitive personal information.

## Licenses

The upstream model card reports Apache-2.0. The Hugging Face mirror reports CC BY 4.0; source corpora within MT560 may carry their own conditions. Project source code is MIT licensed. See the provenance note before redistribution or commercial use.
