"""Local inference wrapper for the trained Hausa -> English model."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from sacremoses import MosesDetokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class HausaTranslator:
    def __init__(self, model_path: str | Path, device: str | None = None, max_source_length: int = 128):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}; train the model first.")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()
        self.max_source_length = max_source_length
        self.output_detokenizer = MosesDetokenizer(lang="en")

    @torch.inference_mode()
    def translate_batch(self, texts: Sequence[str], num_beams: int = 4, max_new_tokens: int = 128) -> list[str]:
        if not texts:
            return []
        normalized = [" ".join(str(text).split()) for text in texts]
        batch = self.tokenizer(
            normalized, return_tensors="pt", padding=True, truncation=True,
            max_length=self.max_source_length,
        ).to(self.device)
        generated = self.model.generate(
            **batch, num_beams=num_beams, max_new_tokens=max_new_tokens,
            early_stopping=True,
        )
        raw = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return [self.output_detokenizer.detokenize(text.split()).strip() for text in raw]

    def translate(self, text: str, **kwargs) -> str:
        return self.translate_batch([text], **kwargs)[0]
