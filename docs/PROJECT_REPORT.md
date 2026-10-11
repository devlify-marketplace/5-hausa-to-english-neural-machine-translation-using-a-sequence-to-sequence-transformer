# Project Report

## Title

**Hausa-to-English Neural Machine Translation Using a Sequence-to-Sequence Transformer**

## Abstract

This project develops a reproducible Hausa-to-English machine-translation prototype using a sequence-to-sequence Transformer. It fine-tunes the pretrained Marian encoder–decoder checkpoint `Helsinki-NLP/opus-mt-ha-en` on an English–Hausa parallel corpus derived from OPUS MT560. A deterministic 80/10/10 partition keeps identical Hausa source sentences within the same split; training uses the train partition, validation loss selects the checkpoint, and the test partition is reserved for final evaluation. The pipeline compares the base model with the fine-tuned checkpoint on the same held-out sample using corpus SacreBLEU, chrF++ and TER, records predictions and metadata, and exposes the saved model through a local REST API and browser interface. The bundled CPU-sized run fine-tunes a reproducible sample of the training split; the software also supports training and evaluation on every available row. The corpus is a heterogeneous mixed-source collection and the base checkpoint was trained using OPUS resources, so evaluation overlap cannot be ruled out. Results are interpreted as internal split measurements rather than independent benchmark estimates. The prototype is for coursework and research inspection, not certified or high-impact translation.

## 1. Introduction

Neural machine translation (NMT) maps a source-language sequence into a target-language sequence. Encoder–decoder Transformers model contextual interactions with self-attention and can generate target tokens autoregressively. Hausa is widely spoken in Nigeria and neighbouring countries, but parallel data, evaluation resources and high-quality digital tools remain more limited than those available for English and other high-resource languages. Transfer learning from a pretrained translation model provides a practical way to build an academic prototype with modest compute.

This work operationalizes the selected topic as sentence-level Hausa (`hau`) to English (`eng`) translation. It uses an existing Marian Transformer checkpoint and fine-tunes it on a public parallel corpus, then evaluates base and adapted models on a held-out subset.

## 2. Problem statement

A Hausa speaker who encounters English-only digital material may face language barriers, while a general-purpose translation service may not expose its data, model or evaluation process. The project studies a reproducible, locally runnable NMT model using documented public resources. The primary technical problem is to adapt a sequence-to-sequence Transformer to Hausa-English sentence pairs, evaluate its output transparently, and provide a simple interface while clearly bounding claims about its quality.

## 3. Aim and objectives

**Aim:** design, implement and evaluate a Hausa-to-English neural machine-translation prototype based on a sequence-to-sequence Transformer.

**Objectives:**
1. Acquire a public Hausa-English parallel corpus and document its source, schema and licensing statements.
2. Validate, orient, de-duplicate and partition aligned sentence pairs reproducibly.
3. Fine-tune a pretrained Marian encoder–decoder model using the training partition and validation-only checkpoint selection.
4. Compare base and fine-tuned translation quality on the same held-out test sample using SacreBLEU, chrF++ and TER.
5. Deliver a locally loadable checkpoint, batch inference module, API, browser demo and automated tests.
6. Discuss dataset bias, potential training/evaluation overlap, licensing, privacy and translation risks.

## 4. Research questions

1. What corpus-level SacreBLEU, chrF++ and TER scores does the base model obtain on the selected held-out sample?
2. How do the scores change after fine-tuning on a bounded subset of the available training partition?
3. What methodological limits arise from mixed-source data and a pretrained checkpoint whose training sources may overlap the corpus collection?

## 5. Scope

The project covers text translation from Hausa into English, one direction only. It does not support English-to-Hausa, speech recognition, speech synthesis, document layout preservation, terminology control, human post-editing, or continuous online learning. It does not establish performance for every Hausa dialect or use case. It is an offline-capable coursework demonstration after installation and model bundling.

## 6. Data acquisition and preprocessing

The project uses the Hugging Face dataset mirror `michsethowusu/english-hausa_sentence-pairs_mt560`. The card reports 295,829 examples in a single train split with `eng` and `hau` fields and names OPUS MT560 as upstream. The MT560 project aggregates multiple parallel corpora and directs users to its source acknowledgements and papers. The mirror reports CC BY 4.0, but upstream component terms may vary; `DATA_PROVENANCE.md` details the cautious reuse position.

The project orients Hausa text as the source and English as the target. It trims whitespace, collapses repeated whitespace, removes empty pairs and exact duplicate pairs, while retaining punctuation and Hausa diacritics. A seeded group assignment splits unique Hausa source strings into train/validation/test groups at nominal 80/10/10 proportions. Identical Hausa source strings cannot cross partitions. Because source groups may differ in size, final example counts are measured rather than assumed. The default experiment uses 3,000 training records and 256 records from each evaluation partition, selected reproducibly. The full available split can be used with caps set to zero.

This split is not a fully independent external test: the available mirror lacks document provenance, and the pretrained checkpoint uses OPUS sources. Exact-source grouping prevents one simple form of leakage but cannot prove independence from pretraining or remove document-level overlap.

## 7. Model and training procedure

The base checkpoint is `Helsinki-NLP/opus-mt-ha-en`, a Marian model with an encoder and an autoregressive decoder. The Hugging Face configuration specifies six encoder and six decoder layers, width 512, eight attention heads, and 2,048-dimensional feed-forward blocks. SentencePiece-based subword tokenization supports the model's vocabulary. The model card lists Hausa as source and English as target and reports Apache-2.0.

Fine-tuning uses `Seq2SeqTrainer`, dynamic batch padding, cross-entropy next-token loss, AdamW optimization and a deterministic random seed. Defaults are learning rate `2e-5`, weight decay `0.01`, batch size 8, one epoch and maximum source/target lengths 128. The checkpoint with the lowest validation loss is retained. The test partition is used only after checkpoint selection for final model comparison. The exact installed library versions, caps, counts, model identity and seed are stored in the artifact metadata.

## 8. Evaluation design

The base model is evaluated before updates, then the best validation-selected fine-tuned checkpoint is evaluated on the identical held-out test sample. Decoding uses deterministic beam search with four beams. Because English target text in the corpus includes whitespace-separated punctuation, Moses English detokenization is applied consistently to each hypothesis and reference before metric scoring. Metrics are computed corpus-wide:

- **SacreBLEU:** n-gram precision with brevity penalty and a serialized tokenization signature.
- **chrF++:** character n-gram F-score with word-order features; useful where word segmentation differs.
- **TER:** translation edit rate; lower is better.
- **Exact-match rate:** fraction whose generated string is an exact reference match after trimming.

Automated reference-based metrics provide reproducible lexical comparison but do not establish semantic adequacy. A single valid translation may differ from the selected reference; human bilingual evaluation is required for robust quality judgments. The test sample is a subset in the packaged run and has sampling uncertainty. Model scores must not be generalized to other domains or user populations.

> Final measured scores and sample counts are reported in `RESULTS.md` and `../artifacts/test_metrics.json`; those artifact values take precedence if a later rerun changes them.

## 9. System implementation

The deliverable is a Python package divided into data preparation, metrics, training, prediction and FastAPI modules. The demo accepts a Hausa input and shows a generated English translation. The API includes readiness, one-text and batch endpoints. Submitted text is handled in memory and is not persisted by the application. Tests validate data orientation, Unicode retention, reproducible grouped splitting, metric output, request constraints and API output without fetching pretrained weights.

## 10. Ethical considerations and limitations

The corpus is aggregated, potentially domain-skewed, and not a representative sample of Hausa language use. The pretrained model's OPUS training provenance means evaluation overlap cannot be ruled out. The experiment is a transfer-learning prototype and does not demonstrate training a new Transformer from random initialization. The training and evaluation caps are small relative to the corpus, and no expert human evaluation has been completed. The corpus license is reported by a mirror; component rights and attribution requirements may vary.

The system may hallucinate, omit, mistranslate or produce unsafe wording. A fluent result can still be wrong. Do not use it as a certified translation, for medical/legal/safety decisions, immigration or official records, or without a qualified human reviewer. Avoid submitting personal or confidential content. Public deployment would require security, access control, rate limiting, logging review and data-protection assessment.

## 11. Conclusion and future work

The project delivers a complete, runnable Hausa→English sequence-to-sequence Transformer fine-tuning workflow, saved model, test artifacts, API and browser demo. It provides a transparent educational starting point rather than evidence of production-quality translation. Future work should establish a source-documented, document-disjoint evaluation set, compare against independently trained and multilingual baselines, add bilingual human adequacy/fluency judgments, measure domain and orthographic variation, and obtain community review. Larger-scale training should be performed only with a compute budget and validation methodology appropriate to the full corpus.

## References

1. Vaswani, A. et al. (2017). *Attention Is All You Need*. Advances in Neural Information Processing Systems.
2. Junczys-Dowmunt, M. et al. (2018). *Marian: Fast Neural Machine Translation in C++*. Proceedings of ACL System Demonstrations.
3. Gowda, T., Zhang, Z., Mattmann, C. A., & May, J. (2021). *Many-to-English Machine Translation Tools, Data, and Pretrained Models*. Proceedings of the 2021 Conference of the Association for Computational Linguistics: System Demonstrations. <https://doi.org/10.18653/v1/2021.acl-demo.37>.
4. Tiedemann, J. (2009). *News from OPUS — A Collection of Multilingual Parallel Corpora with Tools and Interfaces*. Recent Advances in Natural Language Processing. <https://opus.nlpl.eu/>.
5. Helsinki-NLP. *Helsinki-NLP/opus-mt-ha-en*. Model card and OPUS-MT training note. <https://huggingface.co/Helsinki-NLP/opus-mt-ha-en>.
6. OPUS. *MT560: A Many-to-English Machine Translation Dataset*. <https://opus.nlpl.eu/MT560>.
7. michsethowusu. *English-Hausa Parallel Dataset*. Hugging Face dataset card. <https://huggingface.co/datasets/michsethowusu/english-hausa_sentence-pairs_mt560>.
8. Post, M. (2018). *A Call for Clarity in Reporting BLEU Scores*. Proceedings of WMT.
9. Popović, M. (2015). *chrF: character n-gram F-score for automatic MT evaluation*. Proceedings of WMT.
10. Hugging Face. *Translation task guide*. <https://huggingface.co/docs/transformers/en/tasks/translation>.
