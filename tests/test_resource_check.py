"""Tests for T05 (src/check_resources.py).

These tests are fast and do not download models, do not need a GPU and never read the final test:
heavy parts (RuBERT, faster-whisper) are replaced by small fakes that mimic their *contracts*
(for example faster-whisper's lazy segment generator). A green run of this file therefore proves
the logic and the guards of the check, NOT that the defence machine has enough resources: that is
proved only by the real run of ``src/check_resources.py`` (see README_T05.md).

Tests that need PyTorch use ``pytest.importorskip("torch")`` and are skipped when it is missing.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import time
import types
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "check_resources.py"
REAL_CONFIG_PATH = ROOT / "configs" / "resource_check.json"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_resources", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_resources"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


cr = _load_module()
MIB = cr.MIB


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def real_config() -> dict:
    return json.loads(REAL_CONFIG_PATH.read_text(encoding="utf-8"))


def make_train(n: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    names = {0: "negative", 1: "neutral", 2: "positive"}
    rows = []
    for i in range(n):
        label = i % 3
        rows.append(
            {
                "record_id": f"train-00000-of-00001.parquet#row={i:08d}",
                "review_text": f"Уникальный отзыв номер {i}: длинный текст для проверки утечки",
                "label_id": label,
                "label_name": names[label],
            }
        )
    train = pd.DataFrame(rows)
    ids = pd.DataFrame(
        {"train_order": range(n), "record_id": train["record_id"], "label_id": train["label_id"]}
    )
    return train, ids


def write_wav(path: Path, seconds: float = 1.0, rate: int = 8000) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * int(seconds * rate))
    return path


def make_repo(tmp_path: Path, n: int = 30) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    """A miniature repository with all files the check reads (parquet is a placeholder file)."""
    root = tmp_path / "repo"
    cfg = real_config()
    train, ids = make_train(n)
    (root / "configs").mkdir(parents=True)
    (root / "configs" / "resource_check.json").write_text(json.dumps(cfg), encoding="utf-8")
    parquet = root / cfg["paths"]["train_parquet"]
    parquet.parent.mkdir(parents=True)
    parquet.write_bytes(b"placeholder-parquet-bytes")
    ids_path = root / cfg["paths"]["train_ids_csv"]
    ids_path.parent.mkdir(parents=True)
    ids.to_csv(ids_path, index=False)
    metadata_path = root / cfg["paths"]["baseline_run_metadata"]
    metadata_path.write_text(
        json.dumps(
            {
                "train_parquet_sha256": hashlib.sha256(parquet.read_bytes()).hexdigest(),
                "train_ids_sha256": cr.ids_sha256(ids),
                "final_test_used": False,
            }
        ),
        encoding="utf-8",
    )
    split_path = root / cfg["paths"]["split_ids_csv"]
    split_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {"record_id": train["record_id"], "split": "train", "label_name": train["label_name"], "label_id": train["label_id"]}
    ).to_csv(split_path, index=False)
    return root, train, ids


class FakePsutil:
    """Minimal psutil look-alike with controllable numbers."""

    def __init__(self, rss_mib: float = 100.0, available_mib: float = 5000.0, total_mib: float = 14000.0):
        self.rss = rss_mib * MIB
        self.available = available_mib * MIB
        self.total = total_mib * MIB
        outer = self

        class _Process:
            def __init__(self, pid=None):
                pass

            def memory_info(self):
                return SimpleNamespace(rss=outer.rss)

        self.Process = _Process

    def virtual_memory(self):
        return SimpleNamespace(available=self.available, total=self.total)


def fake_env() -> dict:
    return {
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "os": "Windows-10-test",
        "os_edition": "Pro",
        "python": "3.13.7",
        "cpu": "test cpu",
        "logical_cpu_count": 12,
        "physical_cpu_count": 8,
        "ram_total_mib": 14039.0,
        "ram_available_mib_at_start": 4000.0,
        "process_rss_mib_at_start": 300.0,
        "torch": "2.10.0+cu126",
        "torch_cuda_runtime": "12.6",
        "cuda_available": True,
        "gpu": {"name": "RTX 4050 Laptop", "total_vram_mib": 6140.0},
        "versions": {
            "transformers": "5.5.0",
            "faster-whisper": "1.2.1",
            "ctranslate2": "4.8.2",
            "av": "18.1.0",
            "psutil": "7.1.0",
        },
    }


def fake_rubert_metrics(batch_size: int = 8, max_length: int = 256) -> dict:
    return {
        "status": "PASS",
        "model_name": "cointegrated/rubert-tiny2",
        "device": "cuda",
        "batch_size": batch_size,
        "max_length": max_length,
        "mixed_precision": "fp16",
        "training_steps": 3,
        "encoded_sequence_length": max_length,
        "non_pad_tokens_min": 10,
        "non_pad_tokens_mean": 50.0,
        "non_pad_tokens_max": 120,
        "losses": [1.2, 1.1, 1.0],
        "step_seconds_first_cold": 1.4,
        "step_seconds_steady_mean": 0.2,
        "encoder_updated": True,
        "head_updated": True,
        "encoder_sentinel_parameters": ["bert.encoder.layer.0.x", "bert.encoder.layer.2.x"],
        "head_sentinel_parameters": ["classifier.weight", "classifier.bias"],
        "encoder_max_abs_update": 1e-4,
        "head_max_abs_update": 2e-4,
        "encoder_tensors_changed": 30,
        "encoder_tensors_snapshotted": 30,
        "head_tensors_changed": 2,
        "head_tensors_snapshotted": 2,
        "optimizer_steps_skipped_by_grad_scaler": 0,
        "cuda_peak_allocated_after_backward_mib": 500.0,
        "cuda_memory_at_peak_window": {
            "torch_peak_allocated_mib": 576.0,
            "torch_peak_reserved_mib": 880.0,
            "device_used_mib": 1500.0,
        },
        "cuda_device_used_mib_after_step": 1500.0,
        "process_rss_peak_sampled_mib": 1700.0,
        "tokenizer_load_seconds": 0.3,
        "model_load_seconds": 2.0,
        "loading_info": {"available": True, "only_new_head_missing": True, "missing_keys": ["classifier.bias"]},
    }


def fake_whisper_metrics(text: str) -> dict:
    return {
        "status": "PASS",
        "model_size": "small",
        "device": "cpu",
        "compute_type": "int8",
        "backend_reported_device": "cpu",
        "model_load_seconds": 40.0,
        "transcription_seconds": 3.6,
        "audio_duration_seconds": 5.9,
        "real_time_factor": 0.61,
        "process_rss_peak_sampled_mib": 2300.0,
        "segment_count": 2,
        "transcript_chars": len(text),
        "transcript_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def fake_joint_metrics() -> dict:
    return {
        "status": "PASS",
        "operation_order": ["rubert_forward", "whisper_transcribe", "rubert_forward"],
        "rubert_and_whisper_loaded_together": True,
        "rubert_logits_identical_before_after_whisper": True,
        "rubert_forward_seconds_before_whisper": 0.02,
        "rubert_forward_seconds_after_whisper": 0.01,
        "whisper_while_rubert_loaded": {"transcription_seconds_warm": 2.0},
        "process_rss_mib": 2100.0,
    }


def install_fake_faster_whisper(monkeypatch, segments, *, backend_device="cpu", delay=0.0, fail_load=False):
    """Fake ``faster_whisper`` whose ``transcribe`` returns a LAZY generator, like the real one."""
    state = {"consumed": 0, "created": 0}

    class FakeModel:
        def __init__(self, size, device, compute_type, cpu_threads):
            if fail_load:
                raise RuntimeError("Library cublas64_12.dll is not found")
            state["created"] += 1
            self.model = SimpleNamespace(device=backend_device, compute_type=compute_type)

        def transcribe(self, path, language, beam_size, vad_filter):
            def generator():
                for segment in segments:
                    time.sleep(delay)
                    state["consumed"] += 1
                    yield segment

            return generator(), SimpleNamespace(language=language)

    module = types.ModuleType("faster_whisper")
    module.WhisperModel = FakeModel
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    return state


def segment(text: str, start: float = 0.0, end: float = 1.0):
    return SimpleNamespace(text=f" {text} ", start=start, end=end)


TRANSCRIPT_A = "это секретная расшифровка голосового отзыва"
TRANSCRIPT_B = "второй сегмент тоже приватный"


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def test_real_config_is_valid_and_has_expected_shape():
    config = cr.load_config(REAL_CONFIG_PATH)
    assert config["rubert"]["model_name"] == "cointegrated/rubert-tiny2"
    assert set(config["profiles"]) == {"primary", "reserve"}
    assert config["remote_fallback"]["confirmed_by_human"] is False
    assert 1 <= config["rubert"]["training_steps"] <= 3
    assert config["debug_audio"]["language"] == "ru"


def test_real_config_reserve_is_strictly_lighter_than_primary():
    profiles = real_config()["profiles"]
    cr.validate_reserve_is_lighter(profiles["primary"], profiles["reserve"])
    assert profiles["primary"]["rubert"]["max_length"] >= 128
    assert profiles["reserve"]["rubert"]["batch_size"] <= profiles["primary"]["rubert"]["batch_size"]


def test_real_config_local_artifacts_are_outside_reports():
    paths = real_config()["paths"]
    assert not paths["local_artifacts_dir"].startswith(paths["reports_dir"])


def _set(path, value):
    def apply(config):
        node = config
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value

    return apply


def _drop(path):
    def apply(config):
        node = config
        for key in path[:-1]:
            node = node[key]
        del node[path[-1]]

    return apply


@pytest.mark.parametrize(
    "mutation",
    [
        _set(["profiles", "primary", "rubert", "batch_size"], 2),
        _set(["profiles", "primary", "rubert", "batch_size"], True),
        _set(["profiles", "primary", "rubert", "max_length"], 8),
        _set(["profiles", "primary", "rubert", "max_length"], 2048),
        _set(["profiles", "primary", "rubert", "device"], "tpu"),
        _set(["profiles", "primary", "rubert", "mixed_precision"], "fp8"),
        _set(["profiles", "primary", "whisper", "device"], "gpu"),
        _set(["profiles", "primary", "whisper", "compute_type"], "float16"),  # not valid on CPU
        _set(["profiles", "primary", "whisper", "beam_size"], 0),
        _set(["profiles", "primary", "whisper", "cpu_threads"], 0),
        _set(["profiles", "primary", "whisper", "vad_filter"], "yes"),
        _set(["rubert", "training_steps"], 0),
        _set(["rubert", "training_steps"], 4),
        _set(["rubert", "learning_rate"], 0),
        _set(["rubert", "model_name"], "DeepPavlov/rubert-base-cased"),
        _set(["debug_audio", "language"], "en"),
        _set(["debug_audio", "max_seconds"], 0),
        _set(["debug_audio", "max_bytes"], -1),
        _set(["labels"], {"negative": 0, "neutral": 1}),
        _set(["seed"], "42"),
        _set(["machine_label"], " "),
        _set(["paths", "train_parquet"], "C:/data/train.parquet"),
        _set(["paths", "train_parquet"], "/abs/train.parquet"),
        _set(["paths", "train_parquet"], "../outside/train.parquet"),
        _set(["paths", "train_parquet"], "data/processed/validation.parquet"),
        _set(["paths", "local_artifacts_dir"], "reports/resources/local"),
        _set(["remote_fallback", "confirmed_by_human"], True),  # provider is null
        _set(["remote_fallback", "confirmed_by_human"], "true"),
        _drop(["paths", "baseline_run_metadata"]),
        _drop(["remote_fallback"]),
        _set(["unexpected_section"], {}),
        _set(["rubert", "unexpected"], 1),
        _set(["profiles", "primary", "whisper", "unexpected"], 1),
    ],
)
def test_invalid_config_is_rejected(mutation):
    config = real_config()
    mutation(config)
    with pytest.raises(cr.T05Error):
        cr.validate_config(config)


def test_fp16_on_cpu_is_rejected():
    config = real_config()
    config["profiles"]["primary"]["rubert"]["device"] = "cpu"
    with pytest.raises(cr.T05Error, match="fp16"):
        cr.validate_config(config)


def test_reserve_not_lighter_is_rejected():
    config = real_config()
    config["profiles"]["reserve"] = copy.deepcopy(config["profiles"]["primary"])
    with pytest.raises(cr.T05Error, match="lighter"):
        cr.validate_config(config)


def test_reserve_heavier_is_rejected():
    config = real_config()
    config["profiles"]["reserve"]["rubert"]["batch_size"] = 16
    with pytest.raises(cr.T05Error):
        cr.validate_config(config)


def test_reserve_larger_whisper_is_rejected():
    config = real_config()
    config["profiles"]["reserve"]["whisper"]["model_size"] = "medium"
    with pytest.raises(cr.T05Error, match="larger"):
        cr.validate_config(config)


def test_remote_fallback_confirmed_needs_provider():
    config = real_config()
    config["remote_fallback"]["confirmed_by_human"] = True
    config["remote_fallback"]["provider"] = "colab-team-account"
    cr.validate_config(config)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: c["paths"].update({"test_path": "x"}),
        lambda c: c["paths"].update({"testPath": "x"}),
        lambda c: c["paths"].update({"final_test_file": "x"}),
        lambda c: c.update({"final_audio_test": "x"}),
        lambda c: c["paths"].update({"train_parquet": "data/processed/test.parquet"}),
        lambda c: c["paths"].update({"train_parquet": "data/raw/test-00000-of-00001.parquet"}),
        lambda c: c["paths"].update({"train_ids_csv": "tests\\final\\test.csv"}),
        lambda c: c["paths"].update({"reports_dir": "test/reports"}),
        lambda c: c["debug_audio"].update({"extra": "data/final_test/list.csv"}),
    ],
)
def test_final_test_references_are_rejected(mutation):
    config = real_config()
    mutation(config)
    with pytest.raises(cr.T05Error):
        cr.reject_final_test_references(config)


def test_harmless_names_are_not_mistaken_for_test_data():
    config = real_config()
    config["remote_fallback"]["note"] = "latest contest attestation"
    config["paths"]["reports_dir"] = "reports/protests_free/resources"  # 'protests_free' is not a 'test' token
    cr.reject_final_test_references(config)


def test_load_config_reports_bad_json_and_missing_file(tmp_path):
    with pytest.raises(cr.T05Error, match="not found"):
        cr.load_config(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{oops", encoding="utf-8")
    with pytest.raises(cr.T05Error, match="Invalid JSON"):
        cr.load_config(bad)
    arr = tmp_path / "arr.json"
    arr.write_text("[]", encoding="utf-8")
    with pytest.raises(cr.T05Error, match="object"):
        cr.load_config(arr)


def test_resolve_config_path_stays_inside_repo(tmp_path):
    assert cr.resolve_config_path(tmp_path, "a/b.txt") == (tmp_path / "a" / "b.txt").resolve()
    with pytest.raises(cr.T05Error):
        cr.resolve_config_path(tmp_path, "../elsewhere")


def test_parse_args_defaults_and_flags():
    args = cr.parse_args([])
    assert args.profile == "primary"
    assert args.audio is None
    assert args.confirm_debug_audio is False
    assert args.preflight is False
    assert args.offline is False
    args = cr.parse_args(["--profile", "reserve", "--audio", "x.m4a", "--confirm-debug-audio", "--offline"])
    assert (args.profile, args.audio, args.confirm_debug_audio, args.offline) == ("reserve", "x.m4a", True, True)


# --------------------------------------------------------------------------- #
# Train contract, provenance, smoke batch
# --------------------------------------------------------------------------- #
def test_train_contract_accepts_valid_data():
    train, ids = make_train()
    cr.validate_train_contract(train, ids)


@pytest.mark.parametrize(
    "case",
    ["missing_column", "empty_text", "null_text", "bad_label_name", "dup_record", "two_classes"],
)
def test_train_contract_rejects_bad_train(case):
    train, ids = make_train()
    if case == "missing_column":
        train = train.drop(columns=["label_name"])
    elif case == "empty_text":
        train.loc[0, "review_text"] = "   "
    elif case == "null_text":
        train.loc[0, "review_text"] = None
    elif case == "bad_label_name":
        train.loc[0, "label_name"] = "positive"
    elif case == "dup_record":
        train.loc[1, "record_id"] = train.loc[0, "record_id"]
    elif case == "two_classes":
        train = train[train["label_id"] != 2].reset_index(drop=True)
        ids = ids[ids["label_id"] != 2].reset_index(drop=True)
    with pytest.raises(cr.T05Error):
        cr.validate_train_contract(train, ids)


@pytest.mark.parametrize(
    "case", ["text_column", "unknown_id", "wrong_label", "dup_id", "empty", "bad_order", "two_classes", "missing_column"]
)
def test_train_contract_rejects_bad_ids(case):
    train, ids = make_train()
    if case == "text_column":
        ids["review_text"] = train["review_text"]
    elif case == "unknown_id":
        ids.loc[0, "record_id"] = "somewhere-else#row=0"
    elif case == "wrong_label":
        ids.loc[0, "label_id"] = (int(ids.loc[0, "label_id"]) + 1) % 3
    elif case == "dup_id":
        ids.loc[1, "record_id"] = ids.loc[0, "record_id"]
    elif case == "empty":
        ids = ids.iloc[0:0]
    elif case == "bad_order":
        ids.loc[1, "train_order"] = ids.loc[0, "train_order"]
    elif case == "two_classes":
        ids = ids[ids["label_id"] != 2]
    elif case == "missing_column":
        ids = ids.drop(columns=["label_id"])
    with pytest.raises(cr.T05Error):
        cr.validate_train_contract(train, ids)


def test_ids_sha256_matches_t04_formula_and_respects_train_order():
    _, ids = make_train(6)
    expected = hashlib.sha256()
    for rid, label in zip(ids["record_id"], ids["label_id"]):
        expected.update(f"{rid}\t{int(label)}\n".encode("utf-8"))
    assert cr.ids_sha256(ids) == expected.hexdigest()
    shuffled = ids.sample(frac=1.0, random_state=1)
    assert cr.ids_sha256(shuffled) == expected.hexdigest()  # train_order restores the order
    swapped = ids.copy()
    swapped.loc[[0, 1], "train_order"] = [1, 0]
    assert cr.ids_sha256(swapped) != expected.hexdigest()


def test_provenance_accepts_matching_t04_files(tmp_path):
    root, _, ids = make_repo(tmp_path)
    cfg = real_config()["paths"]
    result = cr.verify_train_provenance(
        train_path=root / cfg["train_parquet"],
        train_ids=ids,
        metadata_path=root / cfg["baseline_run_metadata"],
        split_ids_path=root / cfg["split_ids_csv"],
    )
    assert result["train_parquet_sha256_matches_t04"] is True
    assert result["train_ids_sha256_matches_t04"] is True
    assert result["all_ids_in_t03_train_split"] is True
    assert result["t04_final_test_used"] is False
    assert result["train_ids_rows"] == len(ids)


def _provenance_call(root, ids):
    cfg = real_config()["paths"]
    return cr.verify_train_provenance(
        train_path=root / cfg["train_parquet"],
        train_ids=ids,
        metadata_path=root / cfg["baseline_run_metadata"],
        split_ids_path=root / cfg["split_ids_csv"],
    )


def test_provenance_rejects_changed_parquet(tmp_path):
    root, _, ids = make_repo(tmp_path)
    (root / real_config()["paths"]["train_parquet"]).write_bytes(b"another parquet")
    with pytest.raises(cr.T05Error, match="train.parquet does not match"):
        _provenance_call(root, ids)


def test_provenance_rejects_changed_id_list(tmp_path):
    root, _, ids = make_repo(tmp_path)
    reordered = ids.copy()
    reordered.loc[[0, 1], "train_order"] = [1, 0]
    with pytest.raises(cr.T05Error, match="train_ids.csv does not match"):
        _provenance_call(root, reordered)


def test_provenance_rejects_ids_from_other_splits(tmp_path):
    root, _, ids = make_repo(tmp_path)
    split_path = root / real_config()["paths"]["split_ids_csv"]
    split = pd.read_csv(split_path, dtype=str, keep_default_na=False)
    split.loc[5, "split"] = "validation"
    split.to_csv(split_path, index=False)
    with pytest.raises(cr.T05Error, match="not in the T03 'train' split"):
        _provenance_call(root, ids)


@pytest.mark.parametrize("value", [True, None, "missing"])
def test_provenance_requires_final_test_used_false(tmp_path, value):
    root, _, ids = make_repo(tmp_path)
    meta_path = root / real_config()["paths"]["baseline_run_metadata"]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if value == "missing":
        del meta["final_test_used"]
    else:
        meta["final_test_used"] = value
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(cr.T05Error, match="final_test_used"):
        _provenance_call(root, ids)


def test_provenance_reports_missing_files(tmp_path):
    root, _, ids = make_repo(tmp_path)
    cfg = real_config()["paths"]
    (root / cfg["baseline_run_metadata"]).unlink()
    with pytest.raises(cr.T05Error, match="run_metadata.json not found"):
        _provenance_call(root, ids)


def test_fixed_batch_is_deterministic_stratified_and_from_ids_only():
    train, ids = make_train(40)
    ids = ids.sample(frac=1.0, random_state=3)  # file order differs from train_order
    first = cr.select_fixed_batch(train, ids, 8)
    second = cr.select_fixed_batch(train, ids, 8)
    assert first["record_id"].tolist() == second["record_id"].tolist()
    assert len(first) == 8
    assert set(first["label_id"]) == {0, 1, 2}
    assert set(first["record_id"]) <= set(ids["record_id"])
    assert first["record_id"].tolist()[:3] == [
        "train-00000-of-00001.parquet#row=00000000",
        "train-00000-of-00001.parquet#row=00000001",
        "train-00000-of-00001.parquet#row=00000002",
    ]
    assert first["record_id"].is_unique


def test_fixed_batch_uses_only_listed_ids():
    train, ids = make_train(40)
    subset = ids[ids["train_order"] >= 10]
    batch = cr.select_fixed_batch(train, subset, 6)
    assert set(batch["record_id"]) <= set(subset["record_id"])


def test_fixed_batch_rejects_tiny_or_too_large_batch():
    train, ids = make_train(6)
    with pytest.raises(cr.T05Error):
        cr.select_fixed_batch(train, ids, 2)
    with pytest.raises(cr.T05Error, match="Not enough"):
        cr.select_fixed_batch(train, ids, 7)


def test_sample_ids_export_has_no_text(tmp_path):
    train, ids = make_train(20)
    batch = cr.select_fixed_batch(train, ids, 5)
    path = tmp_path / "out" / "rubert_sample_ids.csv"
    cr.write_sample_ids(path, batch)
    exported = pd.read_csv(path)
    assert list(exported.columns) == ["batch_order", "record_id", "label_id"]
    assert "Уникальный отзыв" not in path.read_text(encoding="utf-8")
    assert len(exported) == 5


# --------------------------------------------------------------------------- #
# Debug audio
# --------------------------------------------------------------------------- #
def test_debug_audio_ok_for_real_wav(tmp_path):
    audio = write_wav(tmp_path / "debug_ru.wav", seconds=1.5)
    meta = cr.validate_debug_audio(audio, real_config())
    assert meta["audio_bytes"] == audio.stat().st_size
    assert meta["audio_duration_seconds"] == pytest.approx(1.5, abs=0.01)
    assert meta["audio_sha256"] == hashlib.sha256(audio.read_bytes()).hexdigest()


def test_debug_audio_too_long_is_rejected(tmp_path):
    audio = write_wav(tmp_path / "debug_ru.wav", seconds=1.5)
    config = real_config()
    config["debug_audio"]["max_seconds"] = 1.0
    with pytest.raises(cr.T05Error, match="too long"):
        cr.validate_debug_audio(audio, config)


def test_debug_audio_too_large_is_rejected(tmp_path):
    audio = write_wav(tmp_path / "debug_ru.wav", seconds=1.5)
    config = real_config()
    config["debug_audio"]["max_bytes"] = 100
    with pytest.raises(cr.T05Error, match="too large"):
        cr.validate_debug_audio(audio, config)


def test_debug_audio_missing_and_empty_are_rejected(tmp_path):
    with pytest.raises(cr.T05Error, match="not found"):
        cr.validate_debug_audio(tmp_path / "none.wav", real_config())
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    with pytest.raises(cr.T05Error, match="empty"):
        cr.validate_debug_audio(empty, real_config())


def test_debug_audio_unknown_duration_fails_closed(tmp_path, monkeypatch):
    audio = tmp_path / "debug_ru.m4a"
    audio.write_bytes(b"not really audio")
    monkeypatch.setattr(cr, "probe_audio_duration_seconds", lambda path: None)
    with pytest.raises(cr.T05Error, match="Cannot determine"):
        cr.validate_debug_audio(audio, real_config())


def test_debug_audio_zero_duration_is_rejected(tmp_path, monkeypatch):
    audio = tmp_path / "debug_ru.m4a"
    audio.write_bytes(b"x")
    monkeypatch.setattr(cr, "probe_audio_duration_seconds", lambda path: 0.0)
    with pytest.raises(cr.T05Error, match="zero or invalid"):
        cr.validate_debug_audio(audio, real_config())


@pytest.mark.parametrize(
    "relative",
    ["final_01.wav", "recording_FINAL.m4a", "финал_1.wav", "final_36/rec01.wav", "Финальные записи/rec01.wav"],
)
def test_debug_audio_rejects_names_of_final_recordings(tmp_path, relative):
    audio = write_wav(_mk(tmp_path, relative))
    assert cr.looks_like_final_recording(audio)
    with pytest.raises(cr.T05Error, match="final"):
        cr.validate_debug_audio(audio, real_config())


def _mk(tmp_path: Path, relative: str) -> Path:
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@pytest.mark.parametrize("name", ["debug_ru.m4a", "debug_ru.wav", "smoke_voice.ogg"])
def test_debug_audio_normal_names_are_not_flagged(tmp_path, name):
    assert not cr.looks_like_final_recording(tmp_path / name)


def test_probe_wav_duration_handles_garbage(tmp_path):
    garbage = tmp_path / "broken.wav"
    garbage.write_bytes(b"RIFFxxxx")
    assert cr._probe_wav_duration_seconds(garbage) is None


# --------------------------------------------------------------------------- #
# Git safety
# --------------------------------------------------------------------------- #
def _completed(code: int):
    return subprocess.CompletedProcess(args=["git"], returncode=code, stdout="", stderr="")


@pytest.mark.parametrize("code,expected", [(0, True), (1, False), (128, None)])
def test_git_path_ignored_maps_exit_codes(tmp_path, monkeypatch, code, expected):
    monkeypatch.setattr(cr, "_run_git", lambda root, *args: _completed(code))
    assert cr.git_path_ignored(tmp_path, tmp_path / "artifacts" / "x.json") is expected


def test_git_path_ignored_none_when_git_missing_and_true_outside_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(cr, "_run_git", lambda root, *args: None)
    assert cr.git_path_ignored(tmp_path, tmp_path / "a.json") is None
    other = tmp_path.parent / "elsewhere.m4a"
    assert cr.git_path_ignored(tmp_path, other) is True  # cannot be committed from this repository


def test_git_safety_status_values(tmp_path, monkeypatch):
    target = tmp_path / "artifacts" / "x.json"
    monkeypatch.setattr(cr, "_run_git", lambda root, *args: _completed(0))
    assert cr.git_safety_status(tmp_path, target, "label") == "git_ignored"
    monkeypatch.setattr(cr, "_run_git", lambda root, *args: _completed(128))
    assert cr.git_safety_status(tmp_path, target, "label") == "git_unavailable_not_checked"
    monkeypatch.setattr(cr, "_run_git", lambda root, *args: _completed(1))
    with pytest.raises(cr.T05Error, match="not ignored by Git"):
        cr.git_safety_status(tmp_path, target, "Local transcript folder")
    monkeypatch.setattr(cr, "_run_git", lambda root, *args: _completed(0))
    assert cr.git_safety_status(tmp_path, tmp_path.parent / "voice.m4a", "audio") == "outside_repository"


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_git_guard_with_a_real_repository(tmp_path):
    repo = tmp_path / "gitrepo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / ".gitignore").write_text("artifacts/\n*.m4a\n", encoding="utf-8")
    assert cr.git_path_ignored(repo, repo / "artifacts" / "resource_check" / "primary" / "whisper_result.json") is True
    assert cr.git_path_ignored(repo, repo / "voice" / "debug_ru.m4a") is True
    assert cr.git_path_ignored(repo, repo / "reports" / "resources" / "resource_report.json") is False
    (repo / ".gitignore").write_text("", encoding="utf-8")
    with pytest.raises(cr.T05Error, match="not ignored by Git"):
        cr.git_safety_status(repo, repo / "artifacts" / "whisper_result.json", "Local transcript folder")


# --------------------------------------------------------------------------- #
# Parameter-update evidence (pure numpy)
# --------------------------------------------------------------------------- #
BERT_LIKE_NDIMS = {
    "bert.embeddings.word_embeddings.weight": 2,
    "bert.embeddings.position_embeddings.weight": 2,
    "bert.embeddings.LayerNorm.weight": 1,
    "bert.encoder.layer.0.attention.self.query.weight": 2,
    "bert.encoder.layer.0.attention.self.query.bias": 1,
    "bert.encoder.layer.1.output.dense.weight": 2,
    "bert.encoder.layer.2.output.dense.weight": 2,
    "bert.pooler.dense.weight": 2,
    "classifier.weight": 2,
    "classifier.bias": 1,
}


def test_sentinels_cover_first_and_last_transformer_layer_and_head_not_embeddings():
    sentinels = cr.select_sentinel_parameters(BERT_LIKE_NDIMS)
    assert any(".encoder.layer.0." in name for name in sentinels["encoder"])
    assert any(".encoder.layer.2." in name for name in sentinels["encoder"])
    assert not any("embeddings" in name for name in sentinels["encoder"])
    assert all(param.ndim >= 2 for param in [SimpleNamespace(ndim=BERT_LIKE_NDIMS[n]) for n in sentinels["encoder"]])
    assert sentinels["head"] == ["classifier.bias", "classifier.weight"]


def test_sentinels_fail_without_transformer_layers_or_head():
    only_embeddings = {k: v for k, v in BERT_LIKE_NDIMS.items() if "encoder.layer" not in k}
    with pytest.raises(cr.T05Error, match="transformer-layer"):
        cr.select_sentinel_parameters(only_embeddings)
    no_head = {k: v for k, v in BERT_LIKE_NDIMS.items() if not k.startswith("classifier")}
    with pytest.raises(cr.T05Error, match="classifier-head"):
        cr.select_sentinel_parameters(no_head)


def _snapshots(changed: set[str]):
    before = {name: np.ones((2, 2)) for name in BERT_LIKE_NDIMS}
    after = {name: (before[name] + 1e-3 if name in changed else before[name].copy()) for name in before}
    return before, after


def _grad_norms(value=0.5):
    return {name: value for name in BERT_LIKE_NDIMS}


def test_update_evidence_accepts_real_update():
    sentinels = cr.select_sentinel_parameters(BERT_LIKE_NDIMS)
    changed = set(sentinels["encoder"]) | set(sentinels["head"])
    before, after = _snapshots(changed)
    evidence = cr.evaluate_parameter_updates(before, after, _grad_norms(), sentinels)
    assert evidence["encoder_updated"] and evidence["head_updated"]
    assert evidence["encoder_max_abs_update"] == pytest.approx(1e-3)
    assert evidence["head_tensors_changed"] == 2
    assert evidence["encoder_tensors_changed"] == len(sentinels["encoder"])


def test_update_evidence_rejects_when_only_embeddings_changed():
    sentinels = cr.select_sentinel_parameters(BERT_LIKE_NDIMS)
    before, after = _snapshots({"bert.embeddings.position_embeddings.weight", *sentinels["head"]})
    with pytest.raises(cr.T05Error, match="encoder parameter"):
        cr.evaluate_parameter_updates(before, after, _grad_norms(), sentinels)


def test_update_evidence_rejects_unchanged_head():
    sentinels = cr.select_sentinel_parameters(BERT_LIKE_NDIMS)
    before, after = _snapshots(set(sentinels["encoder"]))
    with pytest.raises(cr.T05Error, match="head parameter"):
        cr.evaluate_parameter_updates(before, after, _grad_norms(), sentinels)


@pytest.mark.parametrize("bad_grad", [0.0, float("nan"), float("inf"), None])
def test_update_evidence_rejects_missing_zero_or_nonfinite_gradients(bad_grad):
    sentinels = cr.select_sentinel_parameters(BERT_LIKE_NDIMS)
    changed = set(sentinels["encoder"]) | set(sentinels["head"])
    before, after = _snapshots(changed)
    norms = _grad_norms()
    norms[sentinels["encoder"][0]] = bad_grad
    if bad_grad is None:
        del norms[sentinels["encoder"][0]]
    with pytest.raises(cr.T05Error, match="gradient"):
        cr.evaluate_parameter_updates(before, after, norms, sentinels)


def test_update_evidence_requires_snapshots_of_sentinels():
    sentinels = cr.select_sentinel_parameters(BERT_LIKE_NDIMS)
    changed = set(sentinels["encoder"]) | set(sentinels["head"])
    before, after = _snapshots(changed)
    del after[sentinels["head"][0]]
    with pytest.raises(cr.T05Error, match="snapshot"):
        cr.evaluate_parameter_updates(before, after, _grad_norms(), sentinels)


def test_summarize_loading_info():
    ok = cr.summarize_loading_info(
        {"missing_keys": {"classifier.weight", "classifier.bias"}, "unexpected_keys": {"cls.a", "cls.b"}, "mismatched_keys": []}
    )
    assert ok["available"] and ok["only_new_head_missing"] is True
    assert ok["unexpected_keys_count"] == 2
    bad = cr.summarize_loading_info(
        {"missing_keys": ["classifier.weight", "bert.encoder.layer.0.attention.self.query.weight"], "unexpected_keys": []}
    )
    assert bad["only_new_head_missing"] is False
    assert cr.summarize_loading_info(None) == {"available": False}


def test_synchronize_only_touches_cuda():
    calls: list[str] = []
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: calls.append("sync")))
    cr._synchronize(fake_torch, "cpu")
    assert calls == []
    cr._synchronize(fake_torch, "cuda")
    assert calls == ["sync"]


# --------------------------------------------------------------------------- #
# Memory monitor
# --------------------------------------------------------------------------- #
def test_monitor_tracks_peak_rss_and_minimum_available_ram():
    fake = FakePsutil(rss_mib=100, available_mib=5000)
    monitor = cr.PeakRSSMonitor(fake)
    assert monitor.peak_mib == pytest.approx(100)
    fake.rss = 700 * MIB
    fake.available = 800 * MIB
    monitor._sample_once(fake.Process(0))
    fake.rss = 300 * MIB
    fake.available = 2000 * MIB
    monitor._sample_once(fake.Process(0))
    assert monitor.peak_mib == pytest.approx(700)
    assert monitor.min_available_mib == pytest.approx(800)


def test_monitor_thread_start_stop_with_real_psutil():
    psutil = pytest.importorskip("psutil")
    monitor = cr.PeakRSSMonitor(psutil, interval_seconds=0.005).start()
    time.sleep(0.03)
    peak = monitor.stop()
    assert peak > 0
    assert monitor.min_available_mib is not None and monitor.min_available_mib > 0


def test_process_peak_rss_prefers_windows_peak_working_set():
    class Proc:
        def __init__(self, pid=None):
            pass

        def memory_info(self):
            return SimpleNamespace(rss=10 * MIB, peak_wset=2048 * MIB)

    assert cr.process_peak_rss_mib(SimpleNamespace(Process=Proc)) == pytest.approx(2048)


def test_process_peak_rss_falls_back_to_os_counter():
    psutil = pytest.importorskip("psutil")
    value = cr.process_peak_rss_mib(psutil)
    assert value is None or value > 0


def test_collect_environment_is_json_serialisable(tmp_path):
    psutil = pytest.importorskip("psutil")
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False), __version__="0.0", version=SimpleNamespace(cuda=None)
    )
    env = cr.collect_environment(fake_torch, psutil, tmp_path)
    assert env["cuda_available"] is False and env["gpu"] is None
    assert env["ram_total_mib"] > 0
    assert "faster-whisper" in env["versions"]
    json.loads(cr.dumps_json(env))


# --------------------------------------------------------------------------- #
# Whisper: lazy generator must be consumed inside the measured section
# --------------------------------------------------------------------------- #
def test_transcribe_fully_consumes_the_generator_inside_the_timer(monkeypatch):
    state = install_fake_faster_whisper(
        monkeypatch, [segment("a"), segment("b"), segment("c")], delay=0.03
    )
    from faster_whisper import WhisperModel

    model = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=1)
    segments, info, seconds = cr.transcribe_fully(model, Path("x.wav"), language="ru", beam_size=5, vad_filter=True)
    assert state["consumed"] == 3 and len(segments) == 3
    assert seconds >= 0.08  # 3 segments * 0.03 s of decoding happened INSIDE the timed section
    assert info.language == "ru"


def test_whisper_run_writes_transcript_only_to_local_artifact(tmp_path, monkeypatch):
    psutil = pytest.importorskip("psutil")
    audio = write_wav(tmp_path / "debug_ru.wav", seconds=2.0)
    install_fake_faster_whisper(
        monkeypatch, [segment(TRANSCRIPT_A, 0.0, 1.0), segment(TRANSCRIPT_B, 1.0, 2.0)], delay=0.02
    )
    local = tmp_path / "artifacts" / "primary" / "whisper_result.json"
    profile = real_config()["profiles"]["primary"]
    model, metrics, text = cr.run_whisper_transcription(
        profile=profile, audio_path=audio, local_artifact_path=local, psutil_module=psutil
    )
    assert text == f"{TRANSCRIPT_A} {TRANSCRIPT_B}"
    payload = json.loads(local.read_text(encoding="utf-8"))
    assert payload["text"] == text and len(payload["segments"]) == 2
    assert metrics["segment_count"] == 2
    assert metrics["transcript_chars"] == len(text)
    assert metrics["transcript_sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert metrics["transcription_measured_with_all_segments_consumed"] is True
    assert metrics["transcription_seconds"] >= 0.04
    assert metrics["audio_duration_seconds"] == pytest.approx(2.0, abs=0.01)
    assert metrics["real_time_factor"] is not None
    assert metrics["backend_reported_device"] == "cpu"
    assert metrics["transcript_in_git_report"] is False
    dumped = cr.dumps_json(metrics)
    assert TRANSCRIPT_A not in dumped and TRANSCRIPT_B not in dumped
    assert not (set(metrics) & cr.FORBIDDEN_REPORT_KEYS)


def test_whisper_empty_transcript_is_a_failure(tmp_path, monkeypatch):
    psutil = pytest.importorskip("psutil")
    audio = write_wav(tmp_path / "debug_ru.wav")
    install_fake_faster_whisper(monkeypatch, [segment("   ")])
    with pytest.raises(cr.T05Error, match="empty transcription"):
        cr.run_whisper_transcription(
            profile=real_config()["profiles"]["primary"],
            audio_path=audio,
            local_artifact_path=tmp_path / "a" / "w.json",
            psutil_module=psutil,
        )
    assert not (tmp_path / "a" / "w.json").exists()


def test_whisper_no_segments_is_a_failure(tmp_path, monkeypatch):
    psutil = pytest.importorskip("psutil")
    audio = write_wav(tmp_path / "debug_ru.wav")
    install_fake_faster_whisper(monkeypatch, [])
    with pytest.raises(cr.T05Error, match="empty transcription"):
        cr.run_whisper_transcription(
            profile=real_config()["profiles"]["primary"],
            audio_path=audio,
            local_artifact_path=tmp_path / "a" / "w.json",
            psutil_module=psutil,
        )


def test_whisper_backend_device_mismatch_is_detected(tmp_path, monkeypatch):
    psutil = pytest.importorskip("psutil")
    audio = write_wav(tmp_path / "debug_ru.wav")
    install_fake_faster_whisper(monkeypatch, [segment("x y z")], backend_device="cuda")
    with pytest.raises(cr.T05Error, match="reports device"):
        cr.run_whisper_transcription(
            profile=real_config()["profiles"]["primary"],  # expects cpu
            audio_path=audio,
            local_artifact_path=tmp_path / "a" / "w.json",
            psutil_module=psutil,
        )


def test_whisper_load_failure_gives_actionable_error(tmp_path, monkeypatch):
    psutil = pytest.importorskip("psutil")
    audio = write_wav(tmp_path / "debug_ru.wav")
    install_fake_faster_whisper(monkeypatch, [segment("x y z")], fail_load=True)
    with pytest.raises(cr.T05Error, match="failed to load"):
        cr.run_whisper_transcription(
            profile=real_config()["profiles"]["primary"],
            audio_path=audio,
            local_artifact_path=tmp_path / "a" / "w.json",
            psutil_module=psutil,
        )


def test_whisper_missing_library_is_actionable(tmp_path, monkeypatch):
    psutil = pytest.importorskip("psutil")
    monkeypatch.setitem(sys.modules, "faster_whisper", None)  # import raises ImportError
    with pytest.raises(cr.T05Error, match="not installed"):
        cr.run_whisper_transcription(
            profile=real_config()["profiles"]["primary"],
            audio_path=tmp_path / "x.wav",
            local_artifact_path=tmp_path / "w.json",
            psutil_module=psutil,
        )


def test_describe_whisper_backend():
    model = SimpleNamespace(model=SimpleNamespace(device="cpu", compute_type="int8"))
    assert cr.describe_whisper_backend(model, "cpu")["backend_reported_device"] == "cpu"
    with pytest.raises(cr.T05Error):
        cr.describe_whisper_backend(model, "cuda")
    assert cr.describe_whisper_backend(SimpleNamespace(), "cpu")["backend_reported_device"] is None


# --------------------------------------------------------------------------- #
# Joint load
# --------------------------------------------------------------------------- #
def test_joint_core_runs_rubert_whisper_rubert_in_order_and_syncs():
    events: list[str] = []
    syncs: list[int] = []

    def rubert():
        events.append("rubert")
        return np.array([[0.1, 0.9, 0.0]])

    def whisper():
        events.append("whisper")
        return {"segments_consumed": 2}

    result = cr.run_joint_core(
        rubert_forward=rubert, whisper_pass=whisper, psutil_module=FakePsutil(), sync=lambda: syncs.append(1)
    )
    assert events == ["rubert", "whisper", "rubert"]
    assert result["operation_order"] == ["rubert_forward", "whisper_transcribe", "rubert_forward"]
    assert result["status"] == "PASS"
    assert result["predicted_label"] == "neutral"
    assert result["whisper_while_rubert_loaded"] == {"segments_consumed": 2}
    assert len(syncs) == 2


def test_joint_core_detects_interference_between_models():
    outputs = iter([np.array([[0.1, 0.2, 0.3]]), np.array([[0.9, 0.2, 0.3]])])
    with pytest.raises(cr.T05Error, match="changed after Whisper"):
        cr.run_joint_core(
            rubert_forward=lambda: next(outputs), whisper_pass=lambda: {}, psutil_module=FakePsutil()
        )


@pytest.mark.parametrize("bad", [np.array([[np.nan, 0.0, 0.0]]), np.array([[0.0, 0.0]])])
def test_joint_core_rejects_nonfinite_or_wrong_class_count(bad):
    with pytest.raises(cr.T05Error):
        cr.run_joint_core(rubert_forward=lambda: bad, whisper_pass=lambda: {}, psutil_module=FakePsutil())


def test_joint_core_propagates_whisper_failure():
    def broken():
        raise cr.T05Error("Whisper transcription failed")

    with pytest.raises(cr.T05Error, match="Whisper transcription failed"):
        cr.run_joint_core(
            rubert_forward=lambda: np.zeros((1, 3)), whisper_pass=broken, psutil_module=FakePsutil()
        )


def test_verify_joint_requires_both_models_loaded():
    runtime = cr.RuntimeObjects(torch=SimpleNamespace(), tokenizer=None, model=None, whisper_model=None)
    with pytest.raises(cr.T05Error, match="at the same time"):
        cr.verify_joint_operation(
            runtime=runtime,
            profile=real_config()["profiles"]["primary"],
            batch_df=make_train(3)[0],
            audio_path=Path("x.wav"),
            language="ru",
            cold_transcript_sha256="0",
            psutil_module=FakePsutil(),
        )


# --------------------------------------------------------------------------- #
# Findings and report privacy
# --------------------------------------------------------------------------- #
def _findings(memory=None, rubert=None, whisper=None, config=None):
    return cr.build_resource_findings(
        environment=fake_env(),
        memory=memory or {"system_available_mib_min_sampled": 4000.0, "system_available_mib_at_start": 5000.0, "system_total_mib": 14000.0},
        rubert=rubert or fake_rubert_metrics(),
        whisper=whisper or fake_whisper_metrics("текст расшифровки для отчёта"),
        config=config or real_config(),
    )


def _codes(findings, level=None):
    return {f["code"] for f in findings if level is None or f["level"] == level}


def test_findings_warn_about_thin_ram_headroom():
    thin = {"system_available_mib_min_sampled": 700.0, "system_available_mib_at_start": 2586.0, "system_total_mib": 14039.0}
    assert "ram_headroom" in _codes(_findings(memory=thin), "WARNING")
    assert "ram_headroom" not in _codes(_findings(), "WARNING")


def test_findings_warn_about_low_vram_headroom_slow_whisper_and_missing_weights():
    rubert = fake_rubert_metrics()
    rubert["cuda_device_used_mib_after_step"] = 5500.0
    rubert["loading_info"] = {"available": True, "only_new_head_missing": False, "missing_keys": ["bert.x"]}
    whisper = fake_whisper_metrics("текст расшифровки для отчёта")
    whisper["real_time_factor"] = 1.7
    warnings = _codes(_findings(rubert=rubert, whisper=whisper), "WARNING")
    assert {"vram_headroom", "whisper_slower_than_realtime", "rubert_missing_weights"} <= warnings


def test_findings_explain_expected_head_initialisation_and_backup_status():
    codes = _codes(_findings(), "INFO")
    assert {"rubert_head_initialised_from_scratch", "external_backup_compute", "first_call_timings", "mixed_precision"} <= codes
    assert cr.external_backup_status(real_config())["status"] == "NOT_VERIFIED_NOT_CLAIMED"
    confirmed = real_config()
    confirmed["remote_fallback"].update({"confirmed_by_human": True, "provider": "team-vds"})
    status = cr.external_backup_status(confirmed)
    assert status["status"] == "CONFIRMED_BY_HUMAN_NOT_CHECKED_BY_SCRIPT"


def test_assert_no_text_leak_finds_text_under_any_key_and_in_escaped_form():
    with pytest.raises(cr.T05Error, match="leaked"):
        cr.assert_no_text_leak({"r.json": json.dumps({"comment": f"see: {TRANSCRIPT_A}"}, ensure_ascii=False)}, [TRANSCRIPT_A])
    with pytest.raises(cr.T05Error, match="leaked"):
        cr.assert_no_text_leak({"r.json": json.dumps({"comment": TRANSCRIPT_A}, ensure_ascii=True)}, [TRANSCRIPT_A])
    cr.assert_no_text_leak({"r.json": "clean"}, [TRANSCRIPT_A, "short"])
    cr.assert_no_text_leak({"r.json": "short inside"}, ["short"])  # too short to be a meaningful needle


def test_assert_no_text_leak_finds_a_fragment_of_a_long_text():
    long_text = "первое предложение записи. " + TRANSCRIPT_A + ". и ещё одно длинное предложение в конце записи"
    fragment_report = json.dumps({"note": f"кусок: {TRANSCRIPT_A}"}, ensure_ascii=False)
    with pytest.raises(cr.T05Error, match="leaked"):
        cr.assert_no_text_leak({"r.json": fragment_report}, [long_text])
    cr.assert_no_text_leak({"r.json": "совсем другой текст отчёта без совпадений"}, [long_text])


def test_sanitize_report_rejects_text_keys_anywhere():
    cr.sanitize_report_for_git({"a": {"b": [{"ok": 1}]}})
    with pytest.raises(cr.T05Error):
        cr.sanitize_report_for_git({"a": {"b": [{"transcript": "x"}]}})
    with pytest.raises(cr.T05Error):
        cr.sanitize_report_for_git({"review_text": "x"})


def _build(tmp_path: Path, joint=None, rubert=None, whisper=None):
    train, ids = make_train(20)
    batch = cr.select_fixed_batch(train, ids, 8)
    train_path = tmp_path / "train.parquet"
    train_path.write_bytes(b"p")
    ids_path = tmp_path / "train_ids.csv"
    ids.to_csv(ids_path, index=False)
    text = f"{TRANSCRIPT_A} {TRANSCRIPT_B}"
    config = real_config()
    findings = _findings()
    report = cr.build_report(
        config=config,
        profile_name="primary",
        environment=fake_env(),
        train_path=train_path,
        train_ids_path=ids_path,
        provenance={
            "train_parquet_sha256_matches_t04": True,
            "train_ids_sha256_matches_t04": True,
            "all_ids_in_t03_train_split": True,
        },
        batch_df=batch,
        audio_meta={"audio_bytes": 22793, "audio_duration_seconds": 5.9, "audio_sha256": "ab" * 32},
        audio_location="outside_repository",
        artifacts_location="git_ignored",
        rubert_metrics=rubert or fake_rubert_metrics(),
        whisper_metrics=whisper or fake_whisper_metrics(text),
        joint_metrics=joint or fake_joint_metrics(),
        memory={
            "process_peak_rss_mib_os": 2600.0,
            "process_peak_rss_mib_sampled": 2500.0,
            "system_available_mib_min_sampled": 4000.0,
            "system_available_mib_at_start": 5000.0,
        },
        findings=findings,
    )
    return report, batch, text


def test_report_status_is_derived_from_stage_results(tmp_path):
    report, _, _ = _build(tmp_path)
    assert report["status"] == "PASS"
    failing_joint = fake_joint_metrics()
    failing_joint["status"] = "FAIL"
    report, _, _ = _build(tmp_path, joint=failing_joint)
    assert report["status"] == "FAIL"


def test_git_safe_outputs_contain_no_transcript_and_no_review_text(tmp_path):
    report, batch, text = _build(tmp_path)
    cr.sanitize_report_for_git(report)
    outputs = {
        "resource_report.json": cr.dumps_json(report),
        "resource_summary.md": cr.render_summary(report),
        "resource_table.csv": cr.render_resource_table(report),
    }
    cr.assert_no_text_leak(outputs, [text, TRANSCRIPT_A, TRANSCRIPT_B, *batch["review_text"].tolist()])
    assert report["constraints"]["final_36_audio_used"] is False
    assert report["constraints"]["final_text_test_used"] is False
    assert report["local_only_artifacts"]["must_not_commit"] is True
    assert report["resource_plan"]["external_backup_compute"]["status"] == "NOT_VERIFIED_NOT_CLAIMED"
    assert report["methodology"] == cr.METHODOLOGY


def test_summary_and_table_expose_the_required_evidence(tmp_path):
    report, _, _ = _build(tmp_path)
    summary = cr.render_summary(report)
    for needle in ("encoder updated: True", "head updated: True", "max_length: 256", "batch_size: 8", "peak allocated", "RSS"):
        assert needle in summary
    rows = {(section, metric): value for section, metric, value, _unit in cr.resource_table_rows(report)}
    assert rows[("rubert", "encoder_updated")] is True
    assert rows[("rubert", "head_updated")] is True
    assert rows[("whisper", "transcription_seconds")] == 3.6
    assert rows[("joint", "both_loaded")] is True
    assert rows[("constraints", "final_36_audio_used")] is False
    assert rows[("constraints", "external_backup_compute")] == "NOT_VERIFIED_NOT_CLAIMED"
    table = cr.render_resource_table(report)
    assert table.splitlines()[0] == "section,metric,value,unit"


# --------------------------------------------------------------------------- #
# Pre-flight and main() with fakes
# --------------------------------------------------------------------------- #
def install_fake_torch(monkeypatch, cuda: bool = True):
    fake = types.ModuleType("torch")
    fake.cuda = SimpleNamespace(is_available=lambda: cuda, get_device_name=lambda idx: "Fake RTX")
    fake.__version__ = "0.0-fake"
    monkeypatch.setitem(sys.modules, "torch", fake)
    return fake


def patch_heavy_parts(monkeypatch, train, ids, *, transcript=f"{TRANSCRIPT_A} {TRANSCRIPT_B}"):
    """Replace everything heavy; keep validation, provenance, batch selection and report writing real."""
    install_fake_torch(monkeypatch)
    monkeypatch.setattr(cr, "package_version", lambda name: "1.0")
    monkeypatch.setattr(cr, "_read_train_and_ids", lambda train_path, ids_path: (train, ids))
    monkeypatch.setattr(cr, "collect_environment", lambda torch_module, psutil_module, root: fake_env())
    calls = {"rubert": 0, "whisper": 0, "joint": 0}

    def fake_rubert(*, config, profile_name, batch_df, psutil_module):
        calls["rubert"] += 1
        profile = config["profiles"][profile_name]["rubert"]
        return cr.RuntimeObjects(torch=None, tokenizer=None, model=None), fake_rubert_metrics(
            profile["batch_size"], profile["max_length"]
        )

    def fake_whisper(*, profile, audio_path, local_artifact_path, psutil_module, language="ru"):
        calls["whisper"] += 1
        cr.write_text_lf(local_artifact_path, json.dumps({"text": transcript}, ensure_ascii=False))
        return object(), fake_whisper_metrics(transcript), transcript

    def fake_joint(**kwargs):
        calls["joint"] += 1
        return fake_joint_metrics()

    monkeypatch.setattr(cr, "run_rubert_training_step", fake_rubert)
    monkeypatch.setattr(cr, "run_whisper_transcription", fake_whisper)
    monkeypatch.setattr(cr, "verify_joint_operation", fake_joint)
    return calls


def test_preflight_is_cheap_and_does_not_run_models(tmp_path, monkeypatch, capsys):
    root, train, ids = make_repo(tmp_path)
    calls = patch_heavy_parts(monkeypatch, train, ids)
    audio = write_wav(tmp_path / "debug_ru.wav")
    code = cr.main(["--preflight", "--profile", "reserve", "--audio", str(audio)], root=root)
    out = json.loads(capsys.readouterr().out)
    assert code == 0 and out["status"] == "PASS"
    assert out["details"]["train_provenance"]["train_parquet_sha256_matches_t04"] is True
    assert out["details"]["audio_duration_seconds"] == pytest.approx(1.0, abs=0.01)
    assert calls == {"rubert": 0, "whisper": 0, "joint": 0}
    assert not (root / "reports" / "resources").exists()


def test_preflight_fails_with_readable_issues_when_data_is_missing(tmp_path, monkeypatch, capsys):
    root, train, ids = make_repo(tmp_path)
    patch_heavy_parts(monkeypatch, train, ids)
    (root / real_config()["paths"]["train_ids_csv"]).unlink()
    code = cr.main(["--preflight"], root=root)
    out = json.loads(capsys.readouterr().out)
    assert code == 2 and out["status"] == "FAIL"
    assert any("train IDs" in issue for issue in out["issues"])


def test_preflight_reports_missing_libraries(tmp_path, monkeypatch, capsys):
    root, train, ids = make_repo(tmp_path)
    patch_heavy_parts(monkeypatch, train, ids)
    monkeypatch.setattr(cr, "package_version", lambda name: None if name == "faster-whisper" else "1.0")
    code = cr.main(["--preflight"], root=root)
    out = json.loads(capsys.readouterr().out)
    assert code == 2
    assert any("faster-whisper" in issue for issue in out["issues"])


def test_preflight_fails_when_cuda_profile_has_no_cuda(tmp_path, monkeypatch, capsys):
    root, train, ids = make_repo(tmp_path)
    patch_heavy_parts(monkeypatch, train, ids)
    install_fake_torch(monkeypatch, cuda=False)
    code = cr.main(["--preflight"], root=root)
    out = json.loads(capsys.readouterr().out)
    assert code == 2 and any("CUDA" in issue for issue in out["issues"])


def test_real_run_requires_audio_and_confirmation_before_touching_anything(tmp_path, monkeypatch, capsys):
    root, train, ids = make_repo(tmp_path)
    calls = patch_heavy_parts(monkeypatch, train, ids)
    assert cr.main([], root=root) == 2
    assert "--audio is required" in capsys.readouterr().err
    audio = write_wav(tmp_path / "debug_ru.wav")
    monkeypatch.setattr(cr, "validate_debug_audio", lambda *a, **k: (_ for _ in ()).throw(AssertionError("audio opened")))
    assert cr.main(["--audio", str(audio)], root=root) == 2
    assert "--confirm-debug-audio" in capsys.readouterr().err
    assert calls == {"rubert": 0, "whisper": 0, "joint": 0}


def test_real_run_refuses_audio_that_looks_like_final_recording(tmp_path, monkeypatch, capsys):
    root, train, ids = make_repo(tmp_path)
    calls = patch_heavy_parts(monkeypatch, train, ids)
    audio = write_wav(tmp_path / "final_recording_07.wav")
    assert cr.main(["--audio", str(audio), "--confirm-debug-audio"], root=root) == 2
    assert "final" in capsys.readouterr().err.lower()
    assert calls["rubert"] == 0 and calls["whisper"] == 0


def test_unknown_profile_is_rejected(tmp_path, monkeypatch, capsys):
    root, train, ids = make_repo(tmp_path)
    patch_heavy_parts(monkeypatch, train, ids)
    assert cr.main(["--profile", "turbo", "--preflight"], root=root) == 2
    assert "Unknown profile" in capsys.readouterr().err


@pytest.mark.parametrize("profile", ["primary", "reserve"])
def test_full_run_with_fakes_writes_git_safe_reports(tmp_path, monkeypatch, capsys, profile):
    root, train, ids = make_repo(tmp_path)
    calls = patch_heavy_parts(monkeypatch, train, ids)
    audio = write_wav(tmp_path / "debug_ru.wav")
    code = cr.main(["--profile", profile, "--audio", str(audio), "--confirm-debug-audio"], root=root)
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert calls == {"rubert": 1, "whisper": 1, "joint": 1}

    reports = root / "reports" / "resources" / profile
    assert {p.name for p in reports.iterdir()} == {
        "resource_report.json", "resource_summary.md", "resource_table.csv", "rubert_sample_ids.csv"
    }
    report = json.loads((reports / "resource_report.json").read_text(encoding="utf-8"))
    assert report["profile"] == profile and report["status"] == "PASS"
    assert report["methodology"] == cr.METHODOLOGY
    batch_size = real_config()["profiles"][profile]["rubert"]["batch_size"]
    assert report["inputs"]["rubert_smoke_batch_size"] == batch_size
    assert sum(report["inputs"]["rubert_smoke_batch_label_counts"].values()) == batch_size
    assert report["train_provenance"]["all_ids_in_t03_train_split"] is True

    for path in reports.iterdir():
        content = path.read_text(encoding="utf-8")
        assert TRANSCRIPT_A not in content and TRANSCRIPT_B not in content, path.name
        for review in train["review_text"].head(batch_size):
            assert review not in content, path.name

    local = root / "artifacts" / "resource_check" / profile / "whisper_result.json"
    assert TRANSCRIPT_A in local.read_text(encoding="utf-8")  # the transcript exists, but only locally
    assert "Final 36 audio recordings were NOT used" in captured.out


def test_failed_rerun_removes_stale_reports(tmp_path, monkeypatch, capsys):
    root, train, ids = make_repo(tmp_path)
    patch_heavy_parts(monkeypatch, train, ids)
    audio = write_wav(tmp_path / "debug_ru.wav")
    reports = root / "reports" / "resources" / "primary"
    reports.mkdir(parents=True)
    for name in ("resource_report.json", "resource_summary.md", "resource_table.csv"):
        (reports / name).write_text('{"status": "PASS", "methodology": "old"}', encoding="utf-8")

    def failing_rubert(**kwargs):
        raise cr.T05Error("RuBERT training step did not update encoder and head")

    monkeypatch.setattr(cr, "run_rubert_training_step", failing_rubert)
    code = cr.main(["--audio", str(audio), "--confirm-debug-audio"], root=root)
    assert code == 2
    assert "did not update" in capsys.readouterr().err
    assert not (reports / "resource_report.json").exists()
    assert not (reports / "resource_summary.md").exists()
    assert not (reports / "resource_table.csv").exists()


def test_leak_into_report_aborts_and_writes_nothing(tmp_path, monkeypatch, capsys):
    root, train, ids = make_repo(tmp_path)
    patch_heavy_parts(monkeypatch, train, ids)
    audio = write_wav(tmp_path / "debug_ru.wav")
    monkeypatch.setattr(
        cr,
        "build_resource_findings",
        lambda **kwargs: [{"level": "INFO", "code": "leak", "message": f"heard: {TRANSCRIPT_A}"}],
    )
    code = cr.main(["--audio", str(audio), "--confirm-debug-audio"], root=root)
    assert code == 2
    assert "leaked" in capsys.readouterr().err
    reports = root / "reports" / "resources" / "primary"
    assert not (reports / "resource_report.json").exists()
    assert not (reports / "resource_summary.md").exists()


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_full_run_refuses_when_local_artifacts_are_not_git_ignored(tmp_path, monkeypatch, capsys):
    root, train, ids = make_repo(tmp_path)
    patch_heavy_parts(monkeypatch, train, ids)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    audio = write_wav(tmp_path / "debug_ru.wav")
    code = cr.main(["--audio", str(audio), "--confirm-debug-audio"], root=root)
    assert code == 2
    assert "not ignored by Git" in capsys.readouterr().err
    assert not (root / "artifacts").exists()
    (root / ".gitignore").write_text("artifacts/\n", encoding="utf-8")
    assert cr.main(["--audio", str(audio), "--confirm-debug-audio"], root=root) == 0


def test_cuda_oom_gets_a_dedicated_exit_code(tmp_path, monkeypatch, capsys):
    root, train, ids = make_repo(tmp_path)
    patch_heavy_parts(monkeypatch, train, ids)
    audio = write_wav(tmp_path / "debug_ru.wav")

    def oom(**kwargs):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    monkeypatch.setattr(cr, "run_rubert_training_step", oom)
    code = cr.main(["--audio", str(audio), "--confirm-debug-audio"], root=root)
    assert code == 4
    assert "reserve profile" in capsys.readouterr().err


def test_offline_flag_sets_hub_offline_environment(tmp_path, monkeypatch):
    root, train, ids = make_repo(tmp_path)
    patch_heavy_parts(monkeypatch, train, ids)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    import os

    cr.main(["--preflight", "--offline"], root=root)
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)


# --------------------------------------------------------------------------- #
# PyTorch-dependent tests (skipped when torch is missing)
# --------------------------------------------------------------------------- #
def _tiny_model_and_tokenizer(freeze_encoder: bool = False, freeze_head: bool = False):
    torch = pytest.importorskip("torch")
    nn = torch.nn

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.bert = nn.Module()
            self.bert.embeddings = nn.Embedding(200, 8)
            self.bert.encoder = nn.Module()
            self.bert.encoder.layer = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])
            self.classifier = nn.Linear(8, 3)

        def forward(self, input_ids, attention_mask=None, labels=None):
            hidden = self.bert.embeddings(input_ids).mean(dim=1)
            for layer in self.bert.encoder.layer:
                hidden = torch.tanh(layer(hidden))
            logits = self.classifier(hidden)
            loss = torch.nn.functional.cross_entropy(logits, labels) if labels is not None else None
            return SimpleNamespace(loss=loss, logits=logits)

    torch.manual_seed(0)
    model = Tiny()
    if freeze_encoder:
        for param in model.bert.encoder.parameters():
            param.requires_grad = False
    if freeze_head:
        for param in model.classifier.parameters():
            param.requires_grad = False

    def tokenizer(texts, padding, truncation, max_length, return_tensors):
        length = max_length if padding == "max_length" else 5
        ids = torch.tensor([[(hash(text) + i) % 199 + 1 for i in range(length)] for text in texts])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    return torch, model, tokenizer


def _profile(max_length: int = 32):
    return {"device": "cpu", "batch_size": 6, "max_length": max_length, "mixed_precision": "none"}


def test_training_steps_prove_real_backward_and_updates_on_tiny_model():
    torch, model, tokenizer = _tiny_model_and_tokenizer()
    train, ids = make_train(12)
    batch = cr.select_fixed_batch(train, ids, 6)
    result = cr.run_training_steps(
        model=model,
        tokenizer=tokenizer,
        batch_df=batch,
        profile_rubert=_profile(),
        rubert_cfg={"learning_rate": 1e-2, "weight_decay": 0.0, "training_steps": 3},
        torch_module=torch,
        psutil_module=FakePsutil(),
    )
    assert result["encoder_updated"] and result["head_updated"]
    assert len(result["step_seconds"]) == 3 and result["step_seconds_steady_mean"] is not None
    assert result["encoded_sequence_length"] == 32
    assert all(np.isfinite(result["losses"]))
    assert result["optimizer_steps_skipped_by_grad_scaler"] == 0
    assert all(p.grad is None for p in model.parameters())  # gradients released after the check


def test_training_steps_reject_frozen_encoder():
    torch, model, tokenizer = _tiny_model_and_tokenizer(freeze_encoder=True)
    train, ids = make_train(12)
    batch = cr.select_fixed_batch(train, ids, 6)
    with pytest.raises(cr.T05Error, match="transformer-layer"):
        cr.run_training_steps(
            model=model, tokenizer=tokenizer, batch_df=batch, profile_rubert=_profile(),
            rubert_cfg={"learning_rate": 1e-2, "weight_decay": 0.0, "training_steps": 1},
            torch_module=torch, psutil_module=FakePsutil(),
        )


def test_training_steps_reject_frozen_head():
    torch, model, tokenizer = _tiny_model_and_tokenizer(freeze_head=True)
    train, ids = make_train(12)
    batch = cr.select_fixed_batch(train, ids, 6)
    with pytest.raises(cr.T05Error, match="classifier-head"):
        cr.run_training_steps(
            model=model, tokenizer=tokenizer, batch_df=batch, profile_rubert=_profile(),
            rubert_cfg={"learning_rate": 1e-2, "weight_decay": 0.0, "training_steps": 1},
            torch_module=torch, psutil_module=FakePsutil(),
        )


def test_training_steps_reject_batch_that_was_not_padded_to_max_length():
    torch, model, tokenizer = _tiny_model_and_tokenizer()
    train, ids = make_train(12)
    batch = cr.select_fixed_batch(train, ids, 6)

    def short_tokenizer(texts, padding, truncation, max_length, return_tensors):
        return tokenizer(texts, padding="longest", truncation=truncation, max_length=max_length, return_tensors=return_tensors)

    with pytest.raises(cr.T05Error, match="padding to max_length"):
        cr.run_training_steps(
            model=model, tokenizer=short_tokenizer, batch_df=batch, profile_rubert=_profile(),
            rubert_cfg={"learning_rate": 1e-2, "weight_decay": 0.0, "training_steps": 1},
            torch_module=torch, psutil_module=FakePsutil(),
        )


def test_verify_module_device_detects_wrong_device():
    torch, model, _ = _tiny_model_and_tokenizer()
    cr.verify_module_device(model, "cpu")
    with pytest.raises(cr.T05Error, match="expected only 'cuda'"):
        cr.verify_module_device(model, "cuda")


def test_joint_operation_on_tiny_model_with_fake_whisper(tmp_path, monkeypatch):
    torch, model, tokenizer = _tiny_model_and_tokenizer()
    audio = write_wav(tmp_path / "debug_ru.wav")
    install_fake_faster_whisper(monkeypatch, [segment(TRANSCRIPT_A), segment(TRANSCRIPT_B)])
    from faster_whisper import WhisperModel

    whisper_model = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=1)
    runtime = cr.RuntimeObjects(torch=torch, tokenizer=tokenizer, model=model, whisper_model=whisper_model)
    profile = {"rubert": _profile(), "whisper": real_config()["profiles"]["primary"]["whisper"]}
    train, _ = make_train(6)
    cold_hash = hashlib.sha256(f"{TRANSCRIPT_A} {TRANSCRIPT_B}".encode("utf-8")).hexdigest()
    result = cr.verify_joint_operation(
        runtime=runtime, profile=profile, batch_df=train, audio_path=audio, language="ru",
        cold_transcript_sha256=cold_hash, psutil_module=FakePsutil(),
    )
    assert result["status"] == "PASS"
    assert result["operation_order"] == ["rubert_forward", "whisper_transcribe", "rubert_forward"]
    assert result["whisper_while_rubert_loaded"]["segments_consumed"] == 2
    assert result["whisper_while_rubert_loaded"]["transcript_matches_cold_run"] is True
    assert result["rubert_and_whisper_loaded_together"] is True
    assert TRANSCRIPT_A not in cr.dumps_json(result)
