"""Offline regression tests. Tiny random BERT fixtures are not RuBERT results."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from safetensors.torch import load_file
from transformers import BertConfig, BertForSequenceClassification, BertModel, BertTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import train_baseline as base
from src import train_transformer as tt


@pytest.fixture
def config():
    return base.read_json(ROOT / "configs/transformer.json")


@pytest.fixture
def project(tmp_path, config):
    frames = {}
    metadata = {"winner_validation_macro_f1": 0.72}
    for split in ("train", "validation"):
        frame = pd.DataFrame({
            "record_id": [f"{split}-{i}" for i in range(6)],
            "review_text": [f"{split} фильм оценка {i}" for i in range(6)],
            "label_id": [0, 1, 2, 0, 1, 2],
            "label_name": base.CLASS_NAMES * 2,
            "review_language": ["ru"] * 6,
            "movie_id": [f"{split}-movie-{i}" for i in range(6)],
        })
        path = tmp_path / config[f"{split}_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        metadata[f"{split}_parquet_sha256"] = base.sha256_file(path)
        metadata[f"{split}_ids_sha256"] = base.ids_sha256(frame, id_column="record_id", label_column="label_id")
        metadata[f"{split}_rows"] = len(frame)
        frames[split] = frame
    audit = {"status": "completed", "final_summary": {
        split: {"rows": len(frame), **base.class_distribution(frame, "label_id")}
        for split, frame in frames.items()}}
    base.write_json(tmp_path / config["data_prep_audit_path"], audit)
    manifest = pd.concat([frame.assign(split=split)[["split", "record_id", "label_id"]]
                          for split, frame in frames.items()], ignore_index=True)
    manifest = pd.concat([manifest, pd.DataFrame([{"split": "test", "record_id": "held-out", "label_id": 0}])])
    manifest.to_csv(tmp_path / config["data_prep_split_ids_path"], index=False)
    for key, expected in (("data_prep_audit_path", "t03_audit_sha256"),
                          ("data_prep_split_ids_path", "t03_split_ids_sha256")):
        metadata[expected] = base.sha256_file(tmp_path / config[key])
    base._write_train_ids(frames["train"], tmp_path / config["baseline_train_ids_path"],
                          id_column="record_id", label_column="label_id")
    base.write_json(tmp_path / config["baseline_metadata_path"], metadata)
    return tmp_path, frames


@pytest.fixture
def tiny_model(tmp_path):
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nфильм\nоценка\ntrain\nvalidation\n", encoding="utf-8")
    tokenizer = BertTokenizer(vocab=str(vocab), do_lower_case=True)
    model = BertForSequenceClassification(BertConfig(
        vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=64,
        num_labels=3, id2label=tt.ID2LABEL, label2id=tt.MAPPING))
    return model, tokenizer


@pytest.mark.parametrize("key,value", [
    ("test_path", "data/processed/test.parquet"),
    ("train_path", "data/processed/test.parquet"),
    ("revision", "main"), ("epochs", 4), ("batch_size", True),
    ("max_length", 513), ("truncation_side", "left"),
    ("learning_rate", float("nan")), ("label_mapping", {"negative": 2, "neutral": 1, "positive": 0}),
])
def test_config_rejects_invalid(config, key, value):
    config[key] = value
    with pytest.raises(ValueError):
        tt.validate_config(config)


def test_data_only_reads_train_validation(project, config, monkeypatch):
    root, frames = project
    original = pd.read_parquet
    opened = []
    def guarded(path, *args, **kwargs):
        assert Path(path).name in ("train.parquet", "validation.parquet")
        opened.append(Path(path).name)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(pd, "read_parquet", guarded)
    train, validation, version = tt.load_data(config, root)
    assert opened == ["train.parquet", "validation.parquet"]
    assert train.record_id.tolist() == frames["train"].record_id.tolist()
    assert version["train_ids_equal_baseline"] and not version["final_test_used"]
    assert not set(train.record_id) & set(validation.record_id)


@pytest.mark.parametrize("mutation", ["order", "label", "unknown", "parquet", "audit"])
def test_data_rejects_different_baseline_version(project, config, mutation):
    root, _ = project
    path = root / config["baseline_train_ids_path"]
    ids = pd.read_csv(path)
    if mutation == "order":
        ids = ids.iloc[::-1]
    elif mutation == "label":
        ids.loc[0, "label_id"] = 2
    elif mutation == "unknown":
        ids.loc[0, "record_id"] = "held-out"
    elif mutation == "parquet":
        with (root / config["train_path"]).open("ab") as fh:
            fh.write(b"changed")
    else:
        base.write_json(root / config["data_prep_audit_path"], {"status": "completed"})
    ids.to_csv(path, index=False)
    with pytest.raises(ValueError):
        tt.load_data(config, root)


def test_selection_uses_macro_f1_and_earlier_step():
    rows = [{"step": 5, "macro_f1": 0.71, "accuracy": 0.9},
            {"step": 10, "macro_f1": 0.72, "accuracy": 0.8},
            {"step": 15, "macro_f1": 0.72, "accuracy": 0.99}]
    assert tt.select_checkpoint(rows) is rows[1]
    with pytest.raises(ValueError):
        tt.select_checkpoint([{"step": 1, "macro_f1": float("nan")}])


def test_right_truncation_keeps_beginning(config, tiny_model):
    _, tokenizer = tiny_model
    config.update(max_length=8)
    ids = lambda *words: tokenizer.convert_tokens_to_ids(list(words))
    # Distinct start/end tokens: left truncation would keep "validation" and drop "train".
    text = "train оценка " + "фильм " * 20 + "validation"
    tokenizer.truncation_side = "left"
    left = tokenizer([text], truncation=True, max_length=8)["input_ids"][0]
    assert ids("validation")[0] in left and ids("train")[0] not in left
    inputs = tt.encode(tokenizer, [text], config)
    assert tokenizer.truncation_side == "right"
    expected = [tokenizer.cls_token_id, *ids("train", "оценка", *["фильм"] * 4), tokenizer.sep_token_id]
    assert inputs["input_ids"][0].tolist() == expected


def test_local_bundle_roundtrip(tmp_path, config, tiny_model):
    model, tokenizer = tiny_model
    config.update(max_length=8, num_threads=1)
    tt.setup(config)
    before = tt.predict_texts(model, tokenizer, tt.FIXED_TEXTS, config)
    bundle = tmp_path / "bundle"
    tt.save_bundle(bundle, model, tokenizer, config, {"fixture_only": True})
    loaded, tok, cfg = tt.load_bundle(bundle)
    after = tt.predict_texts(loaded, tok, tt.FIXED_TEXTS, cfg)
    np.testing.assert_allclose(before["probabilities"], after["probabilities"], atol=0, rtol=0)
    assert after["probability_labels"] == ["negative", "neutral", "positive"]
    np.testing.assert_allclose(np.sum(after["probabilities"], axis=1), 1, atol=1e-6)
    base.write_json(bundle / "label_mapping.json", {"bad": 0})
    with pytest.raises(ValueError, match="mapping"):
        tt.load_bundle(bundle)


def test_end_to_end_training_updates_encoder_and_head(project, config, tiny_model, monkeypatch):
    root, frames = project
    model, tokenizer = tiny_model
    config.update(epochs=2, batch_size=3, max_length=16, num_threads=1,
                  eval_every_steps=2, log_every_steps=1)
    path = root / "config.json"
    base.write_json(path, config)
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    monkeypatch.setattr(tt, "load_pretrained", lambda c: (tokenizer, model, {"fixture_only": True}))
    result = tt.run_training(path, root)
    assert result["steps"] == result["steps_budget"] == 4
    assert result["examples_seen"] == 2 * len(frames["train"])
    assert result["parameter_updates"]["encoder_updated"]
    assert result["parameter_updates"]["head_updated"]
    report = root / config["report_dir"]
    roundtrip = base.read_json(report / "bundle_roundtrip.json")
    assert roundtrip["validation_predictions_identical_after_load"]
    rows = pd.read_csv(report / "validation_results.csv")
    assert len(rows) == 2
    assert result["winner"]["macro_f1"] == pytest.approx(rows.macro_f1.max())
    assert result["delta_vs_baseline"] == result["winner"]["macro_f1"] - 0.72
    # The saved winner holds fine-tuned weights in every encoder block and the head, read back
    # directly from model.safetensors rather than from the in-memory training model.
    saved = load_file(root / config["model_dir"] / "model.safetensors")
    changed = {k for k, v in saved.items() if v.is_floating_point() and not torch.equal(v, initial[k])}
    for prefix in ("bert.embeddings.", "bert.encoder.layer.0.", "bert.encoder.layer.1.", "classifier."):
        assert any(k.startswith(prefix) for k in changed), prefix
    with pytest.raises(FileExistsError):
        tt.run_training(path, root)


def test_pilot_discards_weights_without_validation(project, config, tiny_model, monkeypatch):
    root, _ = project
    model, tokenizer = tiny_model
    config.update(batch_size=2, max_length=16, num_threads=1)
    path = root / "config.json"
    base.write_json(path, config)
    monkeypatch.setattr(tt, "load_pretrained", lambda c: (tokenizer, model, {"fixture_only": True}))
    def forbidden(*args, **kwargs):
        raise AssertionError("Pilot must not evaluate validation")
    monkeypatch.setattr(tt, "evaluate", forbidden)
    result = tt.run_training(path, root, pilot=True)
    assert result["steps"] == 3 and result["examples_seen"] == 6
    assert result["weights_discarded"]
    assert not (root / config["model_dir"]).exists()
    assert not (root / config["report_dir"] / "validation_results.csv").exists()


def _save_encoder(path, tokenizer, layers=2, declared_layers=None):
    """A local 'pretrained' encoder checkpoint without a classification head."""
    encoder = BertModel(BertConfig(
        vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32, num_hidden_layers=layers,
        num_attention_heads=2, max_position_embeddings=64))
    encoder.save_pretrained(path)
    tokenizer.save_pretrained(path)
    if declared_layers is not None:
        cfg = base.read_json(path / "config.json")
        cfg["num_hidden_layers"] = declared_layers
        base.write_json(path / "config.json", cfg)
    return encoder


def test_load_pretrained_real_encoder_is_complete_and_trainable(tmp_path, config, tiny_model):
    """No substitution of load_pretrained: real from_pretrained on a local encoder directory."""
    _, tokenizer = tiny_model
    encoder = _save_encoder(tmp_path / "encoder", tokenizer)
    config.update(model_name=str(tmp_path / "encoder"))
    tok, model, loading = tt.load_pretrained(config)
    assert loading["only_new_head_missing"] and loading["mismatched_keys_count"] == 0
    assert sorted(loading["missing_keys"]) == ["classifier.bias", "classifier.weight"]
    assert all(p.requires_grad for p in model.parameters())
    assert model.config.id2label == tt.ID2LABEL and model.config.label2id == tt.MAPPING
    for key, value in encoder.state_dict().items():
        assert torch.equal(model.bert.state_dict()[key], value), key
    assert tok.convert_tokens_to_ids("фильм") == tokenizer.convert_tokens_to_ids("фильм")


def test_load_pretrained_rejects_incomplete_encoder(tmp_path, config, tiny_model):
    _, tokenizer = tiny_model
    # The config declares 3 layers but the checkpoint has 2: layer 2 would be randomly initialized.
    _save_encoder(tmp_path / "encoder", tokenizer, layers=2, declared_layers=3)
    config.update(model_name=str(tmp_path / "encoder"))
    with pytest.raises(RuntimeError, match="not loaded completely"):
        tt.load_pretrained(config)
