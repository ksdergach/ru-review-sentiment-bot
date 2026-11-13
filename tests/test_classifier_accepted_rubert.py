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
# The same texts in another batch are padded to another length: float32 kernels then sum
# in another order. Measured on all 10 396 validation texts, one text vs the evaluation batch:
# up to 8.9e-7 (1 CPU thread) and 1.1e-6 (4 threads), labels and truncation equal.
PADDING_ATOL = 2e-6
# bundle_roundtrip.json was written on the training machine (another CPU and BLAS).
RECORDED_ATOL = 1e-6


@pytest.fixture(scope="module", autouse=True)
def guards():
    """Network and training are refused for the whole module, before the first load.

    Attempts are recorded as well as refused: the classifier wraps load errors, so a
    swallowed attempt must still fail the test.
    """
    import torch
    attempts = []
    loads = []

    def refuse(name):
        def call(*args, **kwargs):
            attempts.append(name)
            raise AssertionError(f"inference must not use {name}")
        return call

    original_load = tt.load_bundle

    def counting_load(*args, **kwargs):
        loads.append(args[0])
        return original_load(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        for name in ("connect", "connect_ex"):
            patch.setattr(socket.socket, name, refuse(f"socket.{name}"))
        patch.setattr(socket, "create_connection", refuse("socket.create_connection"))
        patch.setattr(socket, "getaddrinfo", refuse("socket.getaddrinfo"))
        for name in ("run_training", "load_pretrained", "make_loader"):
            patch.setattr(tt, name, refuse(f"train_transformer.{name}"))
        patch.setattr(torch.Tensor, "backward", refuse("Tensor.backward"))
        patch.setattr(torch.optim.Optimizer, "step", refuse("Optimizer.step"))
        patch.setattr(tt, "load_bundle", counting_load)
        yield {"attempts": attempts, "loads": loads}
    assert attempts == []


@pytest.fixture(scope="module")
def accepted(guards):
    # Loads are counted only while the guards are installed; other tests may load first.
    before = len(guards["loads"])
    model = c.RuBERTClassifier()
    assert model._bundle is None
    first = model.predict(TEXTS[0])
    guards["accepted_first_load"] = guards["loads"][before:]
    assert guards["accepted_first_load"] == [BUNDLE] and guards["attempts"] == []
    assert first["label_id"] == RECORDED["label_ids"][0]
    return model


def test_first_load_ran_under_guards(accepted, guards):
    assert guards["accepted_first_load"] == [BUNDLE] and guards["attempts"] == []
    assert accepted.data_version["final_test_used"] is False


def test_concurrent_first_load_of_accepted_bundle(guards):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    before = len(guards["loads"])
    model = c.RuBERTClassifier()
    gate = Barrier(4)

    def run(_):
        gate.wait(timeout=30)
        return model.predict(TEXTS[1])

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run, range(4)))
    assert len(guards["loads"]) - before == 1
    assert all(r == results[0] for r in results) and results[0]["label_id"] == RECORDED["label_ids"][1]
    assert guards["attempts"] == []


def test_accepted_bundle_matches_recorded_and_training_predict(accepted):
    model, tokenizer, config = accepted._get_bundle()
    raw = tt.predict_texts(model, tokenizer, TEXTS, config)
    # Same input, same computation: the interface adds no numerical difference.
    assert [list(r["probabilities"].values()) for r in accepted.predict_batch(TEXTS)] == raw["probabilities"]
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
        assert values == tt.predict_texts(model, tokenizer, [text], config)["probabilities"][0]
        assert values == pytest.approx(raw["probabilities"][i], abs=PADDING_ATOL)
        assert values == pytest.approx(RECORDED["probabilities"][i], abs=RECORDED_ATOL)
        assert result["truncated"] is False
    assert accepted.data_version["final_test_used"] is False


def test_bot_cli_and_default_predict_agree(accepted, monkeypatch, capsys):
    monkeypatch.setattr(c, "_default_classifier", accepted)
    batch = c.predict_batch(TEXTS)
    for text, evaluated in zip(TEXTS, batch):
        direct = c.predict(text)
        assert bot.classify_text_for_bot(text) == bot.format_prediction(direct)
        assert predict_text.main([text]) == 0
        assert json.loads(capsys.readouterr().out) == direct
        # Evaluation path: same fields; batch padding may change probabilities by float noise.
        assert {k: evaluated[k] for k in ("label", "label_id", "truncated")} == \
            {k: direct[k] for k in ("label", "label_id", "truncated")}
        assert list(evaluated["probabilities"]) == c.EXPECTED_CLASS_NAMES
        assert list(evaluated["probabilities"].values()) == pytest.approx(list(direct["probabilities"].values()),
                                                                          abs=PADDING_ATOL)
        assert evaluated["confidence"] == pytest.approx(direct["confidence"], abs=PADDING_ATOL)


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
        assert list(result["probabilities"].values()) == raw["probabilities"][0]
        assert ("усечён" in bot.format_prediction(result)) is truncated


def test_accepted_model_is_reused(accepted, guards):
    before = len(guards["loads"])
    for text in TEXTS * 2:
        accepted.predict(text)
    accepted.predict_batch(TEXTS)
    assert len(guards["loads"]) == before


def test_changed_copy_of_accepted_bundle_is_rejected(tmp_path):
    copy = tmp_path / "bundle"
    copy.mkdir()
    for path in BUNDLE.iterdir():
        (copy / path.name).write_bytes(path.read_bytes())
    (copy / "label_mapping.json").write_text('{"positive": 0, "neutral": 1, "negative": 2}', encoding="utf-8")
    with pytest.raises(c.ModelUnavailableError, match="checksum mismatch: label_mapping.json"):
        c.RuBERTClassifier(copy).predict(TEXTS[0])
