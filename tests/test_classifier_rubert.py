"""Real offline serialization/inference on tiny BERT; not trained RuBERT metrics."""
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np
import pytest

from src import classifier as c, bot, train_baseline as base, train_transformer as tt
from tools import predict_text
from test_train_transformer import tiny_model, config  # noqa: F401
from test_classifier import fake_prediction, make_real_bundle


@pytest.fixture
def saved_transformer(tmp_path, tiny_model, config):
    model, tokenizer = tiny_model
    config.update(max_length=8, num_threads=1)
    tt.setup(config)
    version = {"train_ids_equal_baseline": True, "final_test_used": False,
               "train_ids_sha256": "a" * 64, "validation_ids_sha256": "b" * 64}
    metadata = {"config_sha256": base.sha256_json(config), "data_version": version}
    bundle = tmp_path / "bundle"
    tt.save_bundle(bundle, model, tokenizer, config, metadata)
    report = tmp_path / "run.json"
    base.write_json(report, {**metadata, "bundle_files_sha256": {
        p.name: base.sha256_file(p) for p in bundle.iterdir()}})
    return bundle, report


def test_rubert_matches_training_and_detects_truncation(saved_transformer):
    bundle, report = saved_transformer
    adapter = c.RuBERTClassifier(bundle, report_path=report)
    loaded, tok, cfg = tt.load_bundle(bundle)
    for count in (1, 6, 7, 30):
        text = " ".join(["фильм"] * count)
        result = adapter.predict(text)
        raw = tt.predict_texts(loaded, tok, [text], cfg)
        assert result["label_id"] == raw["label_ids"][0]
        assert list(result["probabilities"]) == base.CLASS_NAMES
        np.testing.assert_allclose(list(result["probabilities"].values()), raw["probabilities"][0], atol=1e-7)
        assert result["truncated"] is (count + 2 > 8)
        assert result["confidence"] == max(result["probabilities"].values())
    assert adapter.data_version["train_ids_sha256"] == "a" * 64


def test_concurrent_rubert_calls_load_once(saved_transformer, monkeypatch):
    bundle, report = saved_transformer
    original = tt.load_bundle
    calls = []
    def load(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(tt, "load_bundle", load)
    adapter = c.RuBERTClassifier(bundle, report_path=report)
    gate = Barrier(4)
    def run(_):
        gate.wait(timeout=5)
        return adapter.predict("фильм оценка")
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run, range(4)))
    assert len(calls) == 1
    assert all(r == results[0] for r in results)


@pytest.mark.parametrize("tamper", ["weights", "mapping", "config", "metadata", "missing", "extra"])
def test_rejects_changed_bundle_before_loading(saved_transformer, monkeypatch, tamper):
    bundle, report = saved_transformer
    names = {"weights": "model.safetensors", "mapping": "label_mapping.json",
             "config": "training_config.json", "metadata": "metadata.json"}
    if tamper == "missing":
        (bundle / "model.safetensors").unlink()
    elif tamper == "extra":
        (bundle / "unexpected.txt").write_text("extra")
    else:
        with (bundle / names[tamper]).open("ab") as stream:
            stream.write(b" ")
    def forbidden(*a, **k):
        pytest.fail("Corrupt bundle must not be loaded")
    monkeypatch.setattr(tt, "load_bundle", forbidden)
    with pytest.raises(c.ModelUnavailableError):
        c.RuBERTClassifier(bundle, report_path=report).predict("фильм")


@pytest.mark.parametrize("backend", ["baseline", "rubert"])
def test_bot_direct_and_cli_share_result(backend, saved_transformer, tmp_path, monkeypatch, capsys):
    if backend == "rubert":
        bundle, report = saved_transformer
        model = c.create_classifier(backend, bundle, report_path=report)
    else:
        model = c.create_classifier(backend, make_real_bundle(tmp_path))
    monkeypatch.setattr(c, "_default_classifier", model)
    text = "фильм " * 20
    direct = c.predict(text)
    assert bot.classify_text_for_bot(text) == bot.format_prediction(direct)
    assert predict_text.predict is c.predict
    assert predict_text.main([text]) == 0
    assert json.loads(capsys.readouterr().out) == direct
    assert ("усечён" in bot.classify_text_for_bot(text)) is direct["truncated"]


@pytest.mark.parametrize("probabilities", [[float("nan"), .2, .8], [float("inf"), 0, 0],
                                           [-.1, .2, .9], [0, 0, 0], [.1, .2, 1.1]])
def test_invalid_probabilities_rejected(probabilities):
    raw = fake_prediction()
    raw["probabilities"] = [probabilities]
    with pytest.raises(c.ClassifierError):
        c._format_prediction(raw, truncated=False)


def test_label_must_maximize_probability():
    raw = fake_prediction(label_id=0, label="negative")
    with pytest.raises(c.ClassifierError):
        c._format_prediction(raw, truncated=False)


def test_environment_selects_rubert_once(saved_transformer, monkeypatch):
    bundle, report = saved_transformer
    monkeypatch.setattr(c, "_default_classifier", None)
    monkeypatch.setenv("CLASSIFIER_BACKEND", "rubert")
    monkeypatch.setenv("CLASSIFIER_MODEL_PATH", str(bundle))
    monkeypatch.setenv("CLASSIFIER_REPORT_PATH", str(report))
    result = c.predict("фильм")
    monkeypatch.setenv("CLASSIFIER_BACKEND", "invalid")
    assert c.predict("фильм") == result
    assert isinstance(c._default_classifier, c.RuBERTClassifier)


def test_unknown_backend_has_clear_error(monkeypatch):
    monkeypatch.setattr(c, "_default_classifier", None)
    monkeypatch.setenv("CLASSIFIER_BACKEND", "unknown")
    with pytest.raises(c.ModelUnavailableError):
        c.predict("фильм")


@pytest.mark.parametrize("text", ["", " " * 3000, "а" * 2001, None])
def test_rubert_validates_before_load(text, tmp_path):
    with pytest.raises(c.InvalidTextError):
        c.RuBERTClassifier(tmp_path / "missing").predict(text)


@pytest.mark.parametrize("change", ["mapping", "truncation", "provenance", "positions"])
def test_incompatible_self_consistent_bundle_rejected(saved_transformer, change):
    bundle, report = saved_transformer
    def edit(name, update):
        value = base.read_json(bundle / name)
        update(value)
        base.write_json(bundle / name, value)
    if change == "mapping":
        edit("config.json", lambda value: value.update(id2label={"0": "positive", "1": "neutral", "2": "negative"}))
    elif change == "truncation":
        edit("training_config.json", lambda value: value.update(truncation_side="left"))
    elif change == "positions":
        edit("training_config.json", lambda value: value.update(max_length=128))
    else:
        edit("metadata.json", lambda value: value["data_version"].update(train_ids_equal_baseline=False))
    edit("metadata.json", lambda value: value.update(config_sha256=base.sha256_json(base.read_json(bundle / "training_config.json"))))
    metadata = base.read_json(bundle / "metadata.json")
    base.write_json(report, {**metadata, "bundle_files_sha256": {p.name: base.sha256_file(p) for p in bundle.iterdir()}})
    with pytest.raises(c.ModelUnavailableError):
        c.RuBERTClassifier(bundle, report_path=report).predict("фильм")
