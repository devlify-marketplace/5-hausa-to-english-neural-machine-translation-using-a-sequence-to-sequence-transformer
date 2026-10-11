# Data provenance and use notes

## Selected corpus

This project uses the English–Hausa parallel dataset mirrored at [Hugging Face: michsethowusu/english-hausa_sentence-pairs_mt560](https://huggingface.co/datasets/michsethowusu/english-hausa_sentence-pairs_mt560). Its dataset card reports **295,829 rows**, `eng` and `hau` columns, a single `train` split, and the [OPUS MT560](https://opus.nlpl.eu/MT560) collection as the original source. MT560 is a many-to-English dataset consolidated from multiple resources; its page links to the original [Gowda et al. paper](https://doi.org/10.18653/v1/2021.acl-demo.37), source acknowledgements and raw/tokenized files.

The selected Hugging Face card reports **CC BY 4.0**. OPUS MT560 describes itself as derived from numerous sources including JW300, OPUS100, Tatoeba, Tanzil, Wikipedia/Wikimedia and others, and asks users to cite source articles and notify the maintainers about rights concerns. These are separate layers of licensing: a mirror-level license statement does not necessarily erase obligations attached to the source materials. Review the current source and dataset terms before redistribution, publication or commercial use. This student project is not legal advice.

## Retrieval and reproducibility

`scripts/download_data.py` retrieves the public Parquet file from the Hugging Face dataset repository using HTTPS. It does not execute dataset loading scripts or remote code. It records the source URL, UTC retrieval time, downloaded byte count and SHA-256 digest in `data/raw/download_manifest.json`. The processed manifest at `data/processed/manifest.json` records the raw hash, usable pair count, split counts, unique-source counts, seed and split method.

The bundled data columns are oriented as **Hausa source (`hau`) → English reference (`eng`)**. Cleaning trims each string and collapses whitespace, removes empty records, and drops exact duplicate sentence pairs. Diacritics, punctuation and spelling are otherwise retained. The split assigns all rows with the same Hausa source sentence to a single partition using a seeded random assignment: 80% train, 10% validation and 10% test by source-group assignment. Since groups can contain different numbers of rows, final pair counts may not match those ratios exactly. This guards against exact repeated Hausa source sentences crossing the data partitions; it does not remove all semantic/document-level overlap.

The default run takes a deterministic random sample of training, validation and test rows from these partitions. It reports the actual sample counts and uses validation loss for model selection. The test sample is evaluated after selection only. For all-row training/evaluation, set both `--max-train-samples` and `--max-eval-samples` to `0`.

## Known data limitations

- MT560 is aggregated from heterogeneous sources, domains, registers and collection processes. The visible sample includes religious texts, so domain balance should not be assumed.
- The dataset-card row total is not a count of unique independent documents or human-reviewed translations. Alignment noise, duplicated contexts, non-literal translations, mistranslations and inconsistent orthography may be present.
- Splitting by identical source sentence reduces exact input leakage, but no document provenance is available in the Hugging Face mirror, so document-level leakage cannot be ruled out.
- The corpus is not representative of the Hausa language community, Nigerian regions, dialects, education levels, informal chat, speech, code-switching or specialized domains.
- No demographic annotations or community-led quality audit are supplied.

For these reasons, automated corpus metrics are not proof of semantic adequacy or community usefulness. A real-world or public release would need a separately sourced, documented, independently held-out evaluation set and bilingual human review.
