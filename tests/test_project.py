from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from hausa_nmt import api
from hausa_nmt.data import load_parallel_pairs, split_by_source_group
from hausa_nmt.metrics import translation_metrics


def test_load_pairs_orients_hausa_to_english_and_preserves_diacritics(tmp_path):
    source = tmp_path / "source.csv"
    pd.DataFrame({"eng": ["Thank you", "Thank you", "Welcome"], "hau": ["Na gode sosai", "Na gode sosai", "Barka da zuwa"]}).to_csv(source, index=False)
    pairs = load_parallel_pairs(source)
    assert list(pairs.columns) == ["source", "target"]
    assert len(pairs) == 2
    assert pairs.iloc[0].source == "Na gode sosai"
    assert pairs.iloc[0].target == "Thank you"


def test_loader_keeps_hausa_diacritics(tmp_path):
    source = tmp_path / "tone.csv"
    pd.DataFrame({"eng": ["Thank you"], "hau": ["Na gode, Ɗan Adam."]}).to_csv(source, index=False)
    pairs = load_parallel_pairs(source)
    assert pairs.iloc[0].source == "Na gode, Ɗan Adam."


def test_grouped_split_is_deterministic_and_source_disjoint():
    frame = pd.DataFrame({"source": [f"hausa {i}" for i in range(200)], "target": [f"english {i}" for i in range(200)]})
    frame = pd.concat([frame, pd.DataFrame({"source": ["hausa 0"], "target": ["another English wording"]})], ignore_index=True)
    first = split_by_source_group(frame, seed=42)
    second = split_by_source_group(frame, seed=42)
    assert {k: list(v.index) for k, v in first.items()} == {k: list(v.index) for k, v in second.items()}
    source_sets = [set(first[k].source) for k in ("train", "validation", "test")]
    assert not (source_sets[0] & source_sets[1] or source_sets[0] & source_sets[2] or source_sets[1] & source_sets[2])
    assert sum(map(len, first.values())) == len(frame)


def test_translation_metrics_return_sacrebleu_chrf_and_ter():
    result = translation_metrics(["The quick brown fox jumps over the lazy dog."], ["The quick brown fox jumps over the lazy dog."])
    assert result["examples"] == 1
    assert result["sacrebleu"] > 0
    assert result["chrf_plus_plus"] > 0
    assert result["ter"] == 0
    assert "sacrebleu_signature" in result


def test_health_status_and_single_text_schema(monkeypatch):
    client = TestClient(api.app)
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["direction"] == "hau-en"

    class FakeTranslator:
        def translate(self, text):
            return "Thank you very much."
        def translate_batch(self, texts):
            return ["Thank you very much." for _ in texts]

    monkeypatch.setattr(api, "get_translator", lambda: FakeTranslator())
    response = client.post("/api/translate", json={"text": "Na gode sosai."})
    assert response.status_code == 200
    assert response.json()["translation"] == "Thank you very much."
    assert response.json()["source_language"] == "Hausa"


def test_batch_validation_and_translation(monkeypatch):
    class FakeTranslator:
        def translate_batch(self, texts):
            return [f"English {i}" for i, _ in enumerate(texts)]
    monkeypatch.setattr(api, "get_translator", lambda: FakeTranslator())
    client = TestClient(api.app)
    response = client.post("/api/translate-batch", json={"texts": ["Sannu", "Na gode"]})
    assert response.status_code == 200
    assert response.json()["translations"] == ["English 0", "English 1"]
    assert client.post("/api/translate", json={"text": "   "}).status_code == 422
