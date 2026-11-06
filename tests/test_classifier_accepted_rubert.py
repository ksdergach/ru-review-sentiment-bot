"""Shared interface on the accepted T06 RuBERT bundle installed by tools/fetch_transformer.py.

Skipped when models/transformer is absent; the classifier itself rejects any bundle
whose files differ from reports/transformer/run_metadata.json. Uses only the fixed
smoke texts and synthetic boundary texts, never the final test split.
"""
import json
import math
import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BUNDLE = ROOT / "models" / "transformer"
if not BUNDLE.is_dir():
    pytest.skip("accepted RuBERT bundle is not installed: python tools/fetch_transformer.py",
                allow_module_level=True)

from src import bot, classifier as c, train_baseline as base, train_transformer as tt  # noqa: E402
from tools import predict_text  # noqa: E402

TEXTS = base._smoke_texts()
RECORDED = json.loads((ROOT / "reports/transformer/bundle_roundtrip.json").read_text(encoding="utf-8"))["fixed_texts"]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("inference must not use the network")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


@pytest.fixture(autouse=True)
def no_training(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("inference must not start training")
    for name in ("run_training", "load_pretrained", "make_loader"):
        monkeypatch.setattr(tt, name, refuse)


@pytest.fixture(scope="module")
def accepted():
    model = c.RuBERTClassifier()
    model.predict(TEXTS[0])
    return model


def test_accepted_bundle_matches_recorded_and_training_predict(accepted):
    model, tokenizer, config = accepted._get_bundle()
    raw = tt.predict_texts(model, tokenizer, TEXTS, config)
    assert raw["probability_labels"] == c.EXPECTED_CLASS_NAMES == ["negative", "neutral", "positive"]
    assert dict(model.config.label2id) == {"negative": 0, "neutral": 1, "positive": 2}
    for i, text in enumerate(TEXTS):
        result = accepted.predict(text)
        assert set(result) == {"label", "label_id", "confidence", "probabilities", "truncated"}
        assert list(result["probabilities"]) == c.EXPECTED_CLASS_NAMES
        values = list(result["probabilities"].values())
        assert all(math.isfinite(v) and 0 <= v <= 1 for v in values)
        assert math.isclose(sum(values), 1, abs_tol=1e-6)
        assert result["label_id"] == raw["label_ids"][i] == RECORDED["label_ids"][i]
        assert result["label"] == RECORDED["labels"][i]
        assert result["confidence"] == result["probabilities"][result["label"]] == max(values)
        assert values == pytest.approx(raw["probabilities"][i], abs=1e-7)
        assert values == pytest.approx(RECORDED["probabilities"][i], abs=1e-6)
        assert result["truncated"] is False
    assert accepted.data_version["final_test_used"] is False


def test_bot_cli_and_default_predict_agree(accepted, monkeypatch, capsys):
    monkeypatch.setattr(c, "_default_classifier", accepted)
    for text in TEXTS:
        direct = c.predict(text)
        assert bot.classify_text_for_bot(text) == bot.format_prediction(direct)
        assert predict_text.main([text]) == 0
        assert json.loads(capsys.readouterr().out) == direct


def test_truncation_boundary_on_real_tokenizer(accepted):
    model, tokenizer, config = accepted._get_bundle()
    def length(text):
        return len(tokenizer(text, add_special_tokens=True)["input_ids"])
    words = 1
    while length(" ".join(["фильм"] * (words + 1))) <= config["max_length"]:
        words += 1
    at_limit, over_limit = " ".join(["фильм"] * words), " ".join(["фильм"] * (words + 1))
    assert length(at_limit) == config["max_length"] < length(over_limit)
    for text, truncated in ((at_limit, False), (over_limit, True), ((TEXTS[0] + " ") * 12, True)):
        result = accepted.predict(text)
        raw = tt.predict_texts(model, tokenizer, [text], config)
        assert result["truncated"] is truncated
        assert list(result["probabilities"].values()) == pytest.approx(raw["probabilities"][0], abs=1e-7)
        assert ("усечён" in bot.format_prediction(result)) is truncated


def test_accepted_model_is_reused(accepted, monkeypatch):
    def reload(*args, **kwargs):
        raise AssertionError("loaded model must be reused")
    monkeypatch.setattr(tt, "load_bundle", reload)
    for text in TEXTS * 2:
        accepted.predict(text)


def test_changed_copy_of_accepted_bundle_is_rejected(tmp_path):
    copy = tmp_path / "bundle"
    copy.mkdir()
    for path in BUNDLE.iterdir():
        (copy / path.name).write_bytes(path.read_bytes())
    (copy / "label_mapping.json").write_text('{"positive": 0, "neutral": 1, "negative": 2}', encoding="utf-8")
    with pytest.raises(c.ModelUnavailableError, match="checksum mismatch: label_mapping.json"):
        c.RuBERTClassifier(copy).predict(TEXTS[0])
